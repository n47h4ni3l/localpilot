param(
    [ValidateSet('Fresh', 'Resume', 'Restart')]
    [string]$Mode = 'Fresh',

    [ValidateSet('MonitorOnly', 'CriticalOnly', 'Protective')]
    [string]$MemoryPolicy = 'MonitorOnly'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'training_memory_policy.ps1')
$policy = Get-TrainingMemoryPolicy
$history = @{}
$decision = $null
$stopReason = $null
$monitorError = $null
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$reportDir = Join-Path $repoRoot 'training\reports'
$pidPath = Join-Path $reportDir 'adapter_v1_eager_20g_20260922.pid'
$statusPath = Join-Path $reportDir 'adapter_v1_guard_status.json'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $reportDir "adapter_v1_$stamp.stdout.log"
$stderrPath = Join-Path $reportDir "adapter_v1_$stamp.stderr.log"
$telemetryPath = Join-Path $reportDir "adapter_v1_$stamp.memory.jsonl"
$wslConfig = Join-Path $env:USERPROFILE '.wslconfig'
$configPath = Join-Path $repoRoot 'training\configs\qlora_v1.yaml'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$outputPath = Join-Path $repoRoot ([string]$config.output.directory)
$checkpointRoot = Join-Path $outputPath 'checkpoints'

if (-not (Test-Path -LiteralPath $wslConfig)) {
    throw 'The measured WSL memory configuration is missing.'
}
$wslSettings = Get-Content -LiteralPath $wslConfig -Raw
if ($wslSettings -notmatch '(?m)^\s*memory=20GB\s*$' -or $wslSettings -notmatch '(?m)^\s*swap=16GB\s*$') {
    throw 'This run requires the measured WSL memory=20GB and swap=16GB settings.'
}
if (-not (Test-Path -LiteralPath (Join-Path $reportDir 'adapter_v1_eager_20g_20260922_dry_run.json'))) {
    throw 'The passing target-machine dry-run report is missing.'
}

$worker = Get-ScheduledTask -TaskName 'LocalPilot Background Worker' -TaskPath '\' -ErrorAction Stop
if ($worker.State -ne 'Disabled') {
    throw 'Pause LocalPilot Background Worker before starting adapter training.'
}
if (Get-Process -Name 'ollama', 'ollama app' -ErrorAction SilentlyContinue) {
    throw 'Close the Ollama app and server before starting adapter training.'
}
if (Get-CimInstance Win32_Process -Filter "name = 'pythonw.exe' OR name = 'python.exe'" |
    Where-Object { $_.CommandLine -match ' -m localpilot\.(broker|runtime_worker)(\s|$)' }) {
    throw 'Close the live LocalPilot broker and runtime worker before training; disabling scheduled tasks alone does not stop them.'
}

$wslArguments = @(
    '-d', 'LocalPilot-Training',
    '--cd', '/mnt/e/LLM_HOME/src/localpilot',
    '--', 'bash', 'training/scripts/launch_adapter_v1.sh'
)
if ($Mode -eq 'Resume') { $wslArguments += '--resume' }
if ($Mode -eq 'Restart') { $wslArguments += '--restart' }

function Get-LatestCompleteCheckpoint {
    if (-not (Test-Path -LiteralPath $checkpointRoot -PathType Container)) {
        return $null
    }

    $candidates = @(
        Get-ChildItem -LiteralPath $checkpointRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $match = [regex]::Match($_.Name, '^checkpoint-([1-9][0-9]*)$')
            if (-not $match.Success) { return }
            [pscustomobject]@{
                step = [int]$match.Groups[1].Value
                path = $_.FullName
                marker = Join-Path $_.FullName 'localpilot_checkpoint_complete.json'
            }
        } |
        Where-Object { Test-Path -LiteralPath $_.marker -PathType Leaf } |
        Sort-Object step -Descending
    )

    if ($candidates.Count -eq 0) { return $null }
    return $candidates[0]
}

function Get-TopMemoryProcesses {
    $rows = @()
    Get-Process -ErrorAction SilentlyContinue |
        Sort-Object WorkingSet64 -Descending |
        Select-Object -First 15 |
        ForEach-Object {
            $rows += [ordered]@{
                pid = $_.Id
                name = $_.ProcessName
                working_set_gib = [math]::Round($_.WorkingSet64 / 1GB, 3)
                private_gib = [math]::Round($_.PrivateMemorySize64 / 1GB, 3)
                paged_gib = [math]::Round($_.PagedMemorySize64 / 1GB, 3)
                handles = $_.HandleCount
            }
        }
    return @($rows)
}

function Get-GpuProcessMemory {
    try {
        $counters = Get-Counter @(
            '\GPU Process Memory(*)\Dedicated Usage',
            '\GPU Process Memory(*)\Shared Usage'
        ) -ErrorAction Stop
        $byPid = @{}
        foreach ($sample in $counters.CounterSamples) {
            $path = [string]$sample.Path
            if ($sample.Status -ne 0 -or $path -notmatch 'pid_(\d+)') { continue }
            $processId = [int]$Matches[1]
            if (-not $byPid.ContainsKey($processId)) {
                $byPid[$processId] = [ordered]@{
                    pid = $processId
                    name = $null
                    dedicated_gib = 0.0
                    shared_gib = 0.0
                }
            }
            if ($path -like '*dedicated usage') {
                $byPid[$processId].dedicated_gib += $sample.CookedValue / 1GB
            }
            elseif ($path -like '*shared usage') {
                $byPid[$processId].shared_gib += $sample.CookedValue / 1GB
            }
        }
        foreach ($processId in @($byPid.Keys)) {
            try {
                $byPid[$processId].name = (
                    Get-Process -Id $processId -ErrorAction Stop
                ).ProcessName
            }
            catch {
                $byPid[$processId].name = $null
            }
            $byPid[$processId].dedicated_gib = [math]::Round(
                $byPid[$processId].dedicated_gib,
                3
            )
            $byPid[$processId].shared_gib = [math]::Round(
                $byPid[$processId].shared_gib,
                3
            )
        }
        return @(
            $byPid.Values |
            Sort-Object {
                [double]$_.dedicated_gib + [double]$_.shared_gib
            } -Descending |
            Select-Object -First 15
        )
    }
    catch {
        return @()
    }
}

function Get-WindowsMemoryBreakdown {
    try {
        $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction Stop
        function GiB($value) {
            if ($null -eq $value) { return $null }
            return [math]::Round(([double]$value) / 1GB, 3)
        }
        return [ordered]@{
            available_gib = if ($null -ne $memory.AvailableMBytes) {
                [math]::Round(([double]$memory.AvailableMBytes) / 1024.0, 3)
            } else { $null }
            cache_gib = GiB $memory.CacheBytes
            system_cache_resident_gib = GiB $memory.SystemCacheResidentBytes
            paged_pool_gib = GiB $memory.PoolPagedBytes
            nonpaged_pool_gib = GiB $memory.PoolNonpagedBytes
            modified_page_list_gib = GiB $memory.ModifiedPageListBytes
            standby_core_gib = GiB $memory.StandbyCacheCoreBytes
            standby_normal_gib = GiB $memory.StandbyCacheNormalPriorityBytes
            standby_reserve_gib = GiB $memory.StandbyCacheReserveBytes
            committed_gib = GiB $memory.CommittedBytes
            commit_limit_gib = GiB $memory.CommitLimit
        }
    }
    catch {
        return [ordered]@{
            error = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
        }
    }
}

function Get-WslMemoryState {
    try {
        $meminfo = @(wsl.exe -d LocalPilot-Training -- cat /proc/meminfo 2>$null)
        if ($LASTEXITCODE -ne 0 -or $meminfo.Count -eq 0) {
            return $null
        }
        $values = @{}
        foreach ($line in $meminfo) {
            if ($line -match '^([A-Za-z_()]+):\s+([0-9]+)\s+kB') {
                $values[$Matches[1]] = [double]$Matches[2] * 1KB
            }
        }
        $top = @()
        $processLines = @(
            wsl.exe -d LocalPilot-Training -- ps -eo pid=,ppid=,comm=,rss=,vsz=,pcpu= --sort=-rss 2>$null |
            Select-Object -First 15
        )
        foreach ($line in $processLines) {
            $parts = @($line.Trim() -split '\s+')
            if ($parts.Count -lt 6) { continue }
            $rssKiB = 0.0
            $vszKiB = 0.0
            $cpu = 0.0
            if (-not [double]::TryParse($parts[3], [ref]$rssKiB)) { continue }
            [void][double]::TryParse($parts[4], [ref]$vszKiB)
            [void][double]::TryParse($parts[5], [ref]$cpu)
            $top += [ordered]@{
                pid = [int]$parts[0]
                ppid = [int]$parts[1]
                name = $parts[2]
                rss_gib = [math]::Round(($rssKiB * 1KB) / 1GB, 3)
                vsz_gib = [math]::Round(($vszKiB * 1KB) / 1GB, 3)
                cpu_percent = [math]::Round($cpu, 2)
            }
        }
        $swapTotal = if ($values.ContainsKey('SwapTotal')) { $values.SwapTotal } else { 0.0 }
        $swapFree = if ($values.ContainsKey('SwapFree')) { $values.SwapFree } else { 0.0 }
        return [ordered]@{
            mem_total_gib = if ($values.ContainsKey('MemTotal')) {
                [math]::Round($values.MemTotal / 1GB, 3)
            } else { $null }
            mem_available_gib = if ($values.ContainsKey('MemAvailable')) {
                [math]::Round($values.MemAvailable / 1GB, 3)
            } else { $null }
            swap_total_gib = [math]::Round($swapTotal / 1GB, 3)
            swap_used_gib = [math]::Round([math]::Max(0.0, $swapTotal - $swapFree) / 1GB, 3)
            top_processes = $top
        }
    }
    catch {
        return [ordered]@{
            error = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
            top_processes = @()
        }
    }
}

function Write-GuardStatus {
    param(
        [string]$State,
        [double]$AvailableGiB,
        [double]$CommitHeadroomGiB,
        [int]$ExitCode = -1,
        $LatestCheckpoint = $null
    )
    $warning = if ($null -ne $decision) { [bool]$decision.warning } else { $false }
    $pressureReason = if ($null -ne $decision) { $decision.pressure_reason } else { $null }
    $lowRamSeconds = if ($null -ne $decision) { $decision.low_ram_seconds } else { 0.0 }
    $lowCommitSeconds = if ($null -ne $decision) { $decision.low_commit_seconds } else { 0.0 }

    [ordered]@{
        state = $State
        mode = $Mode
        memory_policy_mode = $MemoryPolicy
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        available_gib = $AvailableGiB
        commit_headroom_gib = $CommitHeadroomGiB
        windows_process_id = $trainingProcess.Id
        exit_code = $ExitCode
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        memory_log = $telemetryPath
        memory_policy = $policy
        memory_warning = $warning
        pressure_reason = $pressureReason
        stop_reason = $stopReason
        monitor_error = $monitorError
        low_ram_seconds = $lowRamSeconds
        low_commit_seconds = $lowCommitSeconds
        latest_complete_checkpoint_step = if ($null -ne $LatestCheckpoint) { $LatestCheckpoint.step } else { $null }
        latest_complete_checkpoint_path = if ($null -ne $LatestCheckpoint) { $LatestCheckpoint.path } else { $null }
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$statusPath.tmp" -Encoding utf8
    [System.IO.File]::Move("$statusPath.tmp", $statusPath, $true)
}

$available = 0.0
$headroom = 0.0
$latestCheckpoint = Get-LatestCompleteCheckpoint
$topMemory = @()
$gpuMemory = @()
$windowsMemoryBreakdown = $null
$wslMemory = $null
$trainingProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

$safetyStop = $false
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$lastDetailElapsed = -999.0
$lastCheckpointElapsed = -999.0
$lastMemoryWarning = $false

while (-not $trainingProcess.HasExited) {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $available = $os.FreePhysicalMemory / 1MB
        $counters = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
        $committed = @($counters.CounterSamples | Where-Object Path -Like '*committed bytes')
        $limit = @($counters.CounterSamples | Where-Object Path -Like '*commit limit')
        if (
            $null -eq $os.FreePhysicalMemory -or
            $committed.Count -ne 1 -or
            $limit.Count -ne 1 -or
            $committed[0].Status -ne 0 -or
            $limit[0].Status -ne 0 -or
            $limit[0].CookedValue -le 0
        ) {
            throw 'Memory counters unavailable or invalid.'
        }

        $headroom = ($limit[0].CookedValue - $committed[0].CookedValue) / 1GB
        $decision = Get-TrainingMemoryDecision -AvailableGiB $available -CommitHeadroomGiB $headroom -ElapsedSeconds $clock.Elapsed.TotalSeconds -History $history -Mode $MemoryPolicy -Policy $policy
        $stopReason = $decision.reason
        $monitorError = $null

        # Detailed attribution is normally slower than the 3-second
        # headroom sample so the monitor stays lightweight. Capture immediately
        # on the transition into pressure, then return to the 15-second cadence.
        $currentMemoryWarning = [bool]$decision.warning
        $needDetail = (
            ($clock.Elapsed.TotalSeconds - $lastDetailElapsed -ge 15.0) -or
            ($currentMemoryWarning -and -not $lastMemoryWarning)
        )
        if ($needDetail) {
            $topMemory = Get-TopMemoryProcesses
            $gpuMemory = Get-GpuProcessMemory
            $windowsMemoryBreakdown = Get-WindowsMemoryBreakdown
            $wslMemory = Get-WslMemoryState
            $lastDetailElapsed = $clock.Elapsed.TotalSeconds
        }
        if ($clock.Elapsed.TotalSeconds - $lastCheckpointElapsed -ge 15.0) {
            $latestCheckpoint = Get-LatestCompleteCheckpoint
            $lastCheckpointElapsed = $clock.Elapsed.TotalSeconds
        }

        [ordered]@{
            updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
            elapsed_seconds = $clock.Elapsed.TotalSeconds
            available_gib = [math]::Round($available, 3)
            commit_headroom_gib = [math]::Round($headroom, 3)
            decision = $decision
            top_memory_processes = if ($needDetail) { $topMemory } else { $null }
            gpu_process_memory = if ($needDetail) { $gpuMemory } else { $null }
            windows_memory_breakdown = if ($needDetail) { $windowsMemoryBreakdown } else { $null }
            wsl_memory = if ($needDetail) { $wslMemory } else { $null }
            latest_complete_checkpoint_step = if ($null -ne $latestCheckpoint) { $latestCheckpoint.step } else { $null }
        } | ConvertTo-Json -Compress -Depth 8 | Add-Content -LiteralPath $telemetryPath -Encoding utf8

        Write-GuardStatus -State 'running' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
        $lastMemoryWarning = $currentMemoryWarning
    }
    catch {
        $monitorError = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
        Write-Warning "Training telemetry degraded: $monitorError"
        if ($MemoryPolicy -ne 'MonitorOnly') {
            $stopReason = 'memory_monitor_failed'
        }
        try {
            [ordered]@{
                updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
                elapsed_seconds = $clock.Elapsed.TotalSeconds
                monitor_error = $monitorError
                memory_policy_mode = $MemoryPolicy
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $telemetryPath -Encoding utf8
            Write-GuardStatus -State 'monitor_degraded' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
        }
        catch {
            Write-Warning 'Could not persist degraded monitor status.'
        }
    }

    if ($stopReason) {
        $safetyStop = $true
        try {
            Write-GuardStatus -State 'safety_stop_requested' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
        }
        catch {
            Write-Warning 'Could not write guard status; still stopping training.'
        }
        if (Test-Path -LiteralPath $pidPath) {
            $linuxPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
            if ($linuxPid -match '^[1-9][0-9]*$') {
                $commandLine = wsl.exe -d LocalPilot-Training -- ps -p $linuxPid -o args=
                if (
                    $LASTEXITCODE -eq 0 -and
                    $commandLine -match 'python training/scripts/train_adapter.py' -and
                    $commandLine -match '--train'
                ) {
                    wsl.exe -d LocalPilot-Training -- kill -INT $linuxPid
                }
            }
        }
        break
    }

    Start-Sleep -Seconds 3
    $trainingProcess.Refresh()
}

if ($safetyStop -and -not $trainingProcess.WaitForExit(30000)) {
    if (Test-Path -LiteralPath $pidPath) {
        $linuxPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($linuxPid -match '^[1-9][0-9]*$') {
            $commandLine = wsl.exe -d LocalPilot-Training -- ps -p $linuxPid -o args=
            if (
                $LASTEXITCODE -eq 0 -and
                $commandLine -match 'python training/scripts/train_adapter.py' -and
                $commandLine -match '--train'
            ) {
                wsl.exe -d LocalPilot-Training -- kill -TERM $linuxPid
            }
        }
    }
}

$trainingProcess.WaitForExit()
$latestCheckpoint = Get-LatestCompleteCheckpoint
$finalState = if ($safetyStop) {
    'safety_stopped'
}
elseif ($trainingProcess.ExitCode -eq 0) {
    'completed'
}
else {
    'failed'
}
Write-GuardStatus -State $finalState -AvailableGiB $available -CommitHeadroomGiB $headroom -ExitCode $trainingProcess.ExitCode -LatestCheckpoint $latestCheckpoint
)
            [pscustomobject]@{
                step = [int]$match.Groups[1].Value
                path = $_.FullName
                marker = Join-Path $_.FullName 'localpilot_checkpoint_complete.json'
            }
        } |
        Where-Object { Test-Path -LiteralPath $_.marker -PathType Leaf } |
        Sort-Object step -Descending
    )
    if ($candidates.Count -eq 0) { return $null }
    return $candidates[0]
}

function Get-TopMemoryProcesses {
    $rows = @()
    Get-Process -ErrorAction SilentlyContinue |
        Sort-Object WorkingSet64 -Descending |
        Select-Object -First 15 |
        ForEach-Object {
            $rows += [ordered]@{
                pid = $_.Id
                name = $_.ProcessName
                working_set_gib = [math]::Round($_.WorkingSet64 / 1GB, 3)
                private_gib = [math]::Round($_.PrivateMemorySize64 / 1GB, 3)
                paged_gib = [math]::Round($_.PagedMemorySize64 / 1GB, 3)
                handles = $_.HandleCount
            }
        }
    return @($rows)
}

function Get-GpuProcessMemory {
    try {
        $counters = Get-Counter @(
            '\GPU Process Memory(*)\Dedicated Usage',
            '\GPU Process Memory(*)\Shared Usage'
        ) -ErrorAction Stop
        $byPid = @{}
        foreach ($sample in $counters.CounterSamples) {
            $path = [string]$sample.Path
            if ($sample.Status -ne 0 -or $path -notmatch 'pid_(\d+)') { continue }
            $pid = [int]$Matches[1]
            if (-not $byPid.ContainsKey($pid)) {
                $byPid[$pid] = [ordered]@{
                    pid = $pid
                    name = $null
                    dedicated_gib = 0.0
                    shared_gib = 0.0
                }
            }
            if ($path -like '*dedicated usage') {
                $byPid[$pid].dedicated_gib += $sample.CookedValue / 1GB
            }
            elseif ($path -like '*shared usage') {
                $byPid[$pid].shared_gib += $sample.CookedValue / 1GB
            }
        }
        foreach ($pid in @($byPid.Keys)) {
            try { $byPid[$pid].name = (Get-Process -Id $pid -ErrorAction Stop).ProcessName }
            catch { $byPid[$pid].name = $null }
            $byPid[$pid].dedicated_gib = [math]::Round($byPid[$pid].dedicated_gib, 3)
            $byPid[$pid].shared_gib = [math]::Round($byPid[$pid].shared_gib, 3)
        }
        return @(
            $byPid.Values |
            Sort-Object {
                [double]$_.dedicated_gib + [double]$_.shared_gib
            } -Descending |
            Select-Object -First 15
        )
    }
    catch {
        return @()
    }
}

function Write-GuardStatus {
    param(
        [string]$State,
        [double]$AvailableGiB,
        [double]$CommitHeadroomGiB,
        [int]$ExitCode = -1,
        $LatestCheckpoint = $null
    )
    $warning = if ($null -ne $decision) { [bool]$decision.warning } else { $false }
    $pressureReason = if ($null -ne $decision) { $decision.pressure_reason } else { $null }
    $lowRamSeconds = if ($null -ne $decision) { $decision.low_ram_seconds } else { 0.0 }
    $lowCommitSeconds = if ($null -ne $decision) { $decision.low_commit_seconds } else { 0.0 }

    [ordered]@{
        state = $State
        mode = $Mode
        memory_policy_mode = $MemoryPolicy
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        available_gib = $AvailableGiB
        commit_headroom_gib = $CommitHeadroomGiB
        windows_process_id = $trainingProcess.Id
        exit_code = $ExitCode
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        memory_log = $telemetryPath
        memory_policy = $policy
        memory_warning = $warning
        pressure_reason = $pressureReason
        stop_reason = $stopReason
        monitor_error = $monitorError
        low_ram_seconds = $lowRamSeconds
        low_commit_seconds = $lowCommitSeconds
        latest_complete_checkpoint_step = if ($null -ne $LatestCheckpoint) { $LatestCheckpoint.step } else { $null }
        latest_complete_checkpoint_path = if ($null -ne $LatestCheckpoint) { $LatestCheckpoint.path } else { $null }
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$statusPath.tmp" -Encoding utf8
    [System.IO.File]::Move("$statusPath.tmp", $statusPath, $true)
}

$available = 0.0
$headroom = 0.0
$latestCheckpoint = Get-LatestCompleteCheckpoint
$topMemory = @()
$gpuMemory = @()
$trainingProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

$safetyStop = $false
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$lastDetailElapsed = -999.0
$lastCheckpointElapsed = -999.0

while (-not $trainingProcess.HasExited) {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $available = $os.FreePhysicalMemory / 1MB
        $counters = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
        $committed = @($counters.CounterSamples | Where-Object Path -Like '*committed bytes')
        $limit = @($counters.CounterSamples | Where-Object Path -Like '*commit limit')
        if (
            $null -eq $os.FreePhysicalMemory -or
            $committed.Count -ne 1 -or
            $limit.Count -ne 1 -or
            $committed[0].Status -ne 0 -or
            $limit[0].Status -ne 0 -or
            $limit[0].CookedValue -le 0
        ) {
            throw 'Memory counters unavailable or invalid.'
        }

        $headroom = ($limit[0].CookedValue - $committed[0].CookedValue) / 1GB
        $decision = Get-TrainingMemoryDecision -AvailableGiB $available -CommitHeadroomGiB $headroom -ElapsedSeconds $clock.Elapsed.TotalSeconds -History $history -Mode $MemoryPolicy -Policy $policy
        $stopReason = $decision.reason
        $monitorError = $null

        $needDetail = (
            ($clock.Elapsed.TotalSeconds - $lastDetailElapsed -ge 15.0) -or
            [bool]$decision.warning
        )
        if ($needDetail) {
            $topMemory = Get-TopMemoryProcesses
            $gpuMemory = Get-GpuProcessMemory
            $lastDetailElapsed = $clock.Elapsed.TotalSeconds
        }
        if ($clock.Elapsed.TotalSeconds - $lastCheckpointElapsed -ge 15.0) {
            $latestCheckpoint = Get-LatestCompleteCheckpoint
            $lastCheckpointElapsed = $clock.Elapsed.TotalSeconds
        }

        [ordered]@{
            updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
            elapsed_seconds = $clock.Elapsed.TotalSeconds
            available_gib = [math]::Round($available, 3)
            commit_headroom_gib = [math]::Round($headroom, 3)
            decision = $decision
            top_memory_processes = if ($needDetail) { $topMemory } else { $null }
            gpu_process_memory = if ($needDetail) { $gpuMemory } else { $null }
            latest_complete_checkpoint_step = if ($null -ne $latestCheckpoint) { $latestCheckpoint.step } else { $null }
        } | ConvertTo-Json -Compress -Depth 8 | Add-Content -LiteralPath $telemetryPath -Encoding utf8

        Write-GuardStatus -State 'running' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
    }
    catch {
        $monitorError = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
        Write-Warning "Training telemetry degraded: $monitorError"
        if ($MemoryPolicy -ne 'MonitorOnly') {
            $stopReason = 'memory_monitor_failed'
        }
        try {
            [ordered]@{
                updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
                elapsed_seconds = $clock.Elapsed.TotalSeconds
                monitor_error = $monitorError
                memory_policy_mode = $MemoryPolicy
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $telemetryPath -Encoding utf8
            Write-GuardStatus -State 'monitor_degraded' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
        }
        catch {
            Write-Warning 'Could not persist degraded monitor status.'
        }
    }

    if ($stopReason) {
        $safetyStop = $true
        try {
            Write-GuardStatus -State 'safety_stop_requested' -AvailableGiB $available -CommitHeadroomGiB $headroom -LatestCheckpoint $latestCheckpoint
        }
        catch {
            Write-Warning 'Could not write guard status; still stopping training.'
        }
        if (Test-Path -LiteralPath $pidPath) {
            $linuxPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
            if ($linuxPid -match '^[1-9][0-9]*$') {
                $commandLine = wsl.exe -d LocalPilot-Training -- ps -p $linuxPid -o args=
                if (
                    $LASTEXITCODE -eq 0 -and
                    $commandLine -match 'python training/scripts/train_adapter.py' -and
                    $commandLine -match '--train'
                ) {
                    wsl.exe -d LocalPilot-Training -- kill -INT $linuxPid
                }
            }
        }
        break
    }

    Start-Sleep -Seconds 3
    $trainingProcess.Refresh()
}

if ($safetyStop -and -not $trainingProcess.WaitForExit(30000)) {
    if (Test-Path -LiteralPath $pidPath) {
        $linuxPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($linuxPid -match '^[1-9][0-9]*$') {
            $commandLine = wsl.exe -d LocalPilot-Training -- ps -p $linuxPid -o args=
            if (
                $LASTEXITCODE -eq 0 -and
                $commandLine -match 'python training/scripts/train_adapter.py' -and
                $commandLine -match '--train'
            ) {
                wsl.exe -d LocalPilot-Training -- kill -TERM $linuxPid
            }
        }
    }
}

$trainingProcess.WaitForExit()
$latestCheckpoint = Get-LatestCompleteCheckpoint
$finalState = if ($safetyStop) {
    'safety_stopped'
}
elseif ($trainingProcess.ExitCode -eq 0) {
    'completed'
}
else {
    'failed'
}
Write-GuardStatus -State $finalState -AvailableGiB $available -CommitHeadroomGiB $headroom -ExitCode $trainingProcess.ExitCode -LatestCheckpoint $latestCheckpoint
