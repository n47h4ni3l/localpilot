param(
    [ValidateSet('Fresh', 'Resume', 'Restart')]
    [string]$Mode = 'Fresh'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'training_memory_policy.ps1')
$policy = Get-TrainingMemoryPolicy
$history = @{}
$decision = $null
$stopReason = $null
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$reportDir = Join-Path $repoRoot 'training\reports'
$pidPath = Join-Path $reportDir 'adapter_v1_eager_20g_20260922.pid'
$statusPath = Join-Path $reportDir 'adapter_v1_guard_status.json'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $reportDir "adapter_v1_$stamp.stdout.log"
$stderrPath = Join-Path $reportDir "adapter_v1_$stamp.stderr.log"
$telemetryPath = Join-Path $reportDir "adapter_v1_$stamp.memory.jsonl"
$wslConfig = Join-Path $env:USERPROFILE '.wslconfig'

if (-not (Test-Path -LiteralPath $wslConfig)) {
    throw 'The measured WSL memory configuration is missing.'
}
$wslSettings = Get-Content -LiteralPath $wslConfig -Raw
if ($wslSettings -notmatch '(?m)^\s*memory=20GB\s*$' -or $wslSettings -notmatch '(?m)^\s*swap=16GB\s*$') {
    throw 'This guarded run requires the measured WSL memory=20GB and swap=16GB settings.'
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

$wslArguments = @('-d', 'LocalPilot-Training', '--cd', '/mnt/e/LLM_HOME/src/localpilot', '--', 'bash', 'training/scripts/launch_adapter_v1.sh')
if ($Mode -eq 'Resume') { $wslArguments += '--resume' }
if ($Mode -eq 'Restart') { $wslArguments += '--restart' }

function Write-GuardStatus {
    param([string]$State, [double]$AvailableGiB, [double]$CommitHeadroomGiB, [int]$ExitCode = -1)
    [ordered]@{
        state = $State
        mode = $Mode
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        available_gib = $AvailableGiB
        commit_headroom_gib = $CommitHeadroomGiB
        windows_process_id = $trainingProcess.Id
        exit_code = $ExitCode
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        memory_log = $telemetryPath
        memory_policy = $policy
        memory_warning = [bool]$decision.warning
        stop_reason = $stopReason
        low_ram_seconds = $decision.low_ram_seconds
        low_commit_seconds = $decision.low_commit_seconds
    } | ConvertTo-Json | Set-Content -LiteralPath "$statusPath.tmp" -Encoding utf8
    [System.IO.File]::Move("$statusPath.tmp", $statusPath, $true)
}

$available = 0.0
$headroom = 0.0
$trainingProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$safetyStop = $false
$clock = [System.Diagnostics.Stopwatch]::StartNew()

while (-not $trainingProcess.HasExited) {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $available = $os.FreePhysicalMemory / 1MB
        $counters = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
        $committed = @($counters.CounterSamples | Where-Object Path -Like '*committed bytes')
        $limit = @($counters.CounterSamples | Where-Object Path -Like '*commit limit')
        if ($null -eq $os.FreePhysicalMemory -or $committed.Count -ne 1 -or $limit.Count -ne 1 -or
            $committed[0].Status -ne 0 -or $limit[0].Status -ne 0 -or $limit[0].CookedValue -le 0) {
            throw 'Memory counters unavailable or invalid.'
        }
        $headroom = ($limit[0].CookedValue - $committed[0].CookedValue) / 1GB
        $decision = Get-TrainingMemoryDecision -AvailableGiB $available -CommitHeadroomGiB $headroom -ElapsedSeconds $clock.Elapsed.TotalSeconds -History $history -Policy $policy
        $stopReason = $decision.reason
        [ordered]@{
            updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
            elapsed_seconds = $clock.Elapsed.TotalSeconds
            available_gib = $available
            commit_headroom_gib = $headroom
            decision = $decision
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $telemetryPath -Encoding utf8
        Write-GuardStatus -State 'running' -AvailableGiB $available -CommitHeadroomGiB $headroom
    } catch {
        # Never leave a training process running silently without its guard.
        $stopReason = 'memory_monitor_failed'
        Write-Warning "Memory monitoring failed: $($_.Exception.Message)"
    }
    if ($stopReason) {
        $safetyStop = $true
        try { Write-GuardStatus -State 'safety_stop_requested' -AvailableGiB $available -CommitHeadroomGiB $headroom }
        catch { Write-Warning 'Could not write guard status; still stopping training.' }
        if (Test-Path -LiteralPath $pidPath) {
            $linuxPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
            if ($linuxPid -match '^[1-9][0-9]*$') {
                $commandLine = wsl.exe -d LocalPilot-Training -- ps -p $linuxPid -o args=
                if ($LASTEXITCODE -eq 0 -and $commandLine -match 'python training/scripts/train_adapter.py' -and $commandLine -match '--train') {
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
            if ($LASTEXITCODE -eq 0 -and $commandLine -match 'python training/scripts/train_adapter.py' -and $commandLine -match '--train') {
                wsl.exe -d LocalPilot-Training -- kill -TERM $linuxPid
            }
        }
    }
}

$trainingProcess.WaitForExit()
$finalState = if ($safetyStop) { 'safety_stopped' } elseif ($trainingProcess.ExitCode -eq 0) { 'completed' } else { 'failed' }
Write-GuardStatus -State $finalState -AvailableGiB $available -CommitHeadroomGiB $headroom -ExitCode $trainingProcess.ExitCode
