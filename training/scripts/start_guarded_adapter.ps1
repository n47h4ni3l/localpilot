param(
    [ValidateSet('Fresh', 'Resume', 'Restart')]
    [string]$Mode = 'Fresh'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$reportDir = Join-Path $repoRoot 'training\reports'
$pidPath = Join-Path $reportDir 'adapter_v1_eager_20g_20260922.pid'
$statusPath = Join-Path $reportDir 'adapter_v1_guard_status.json'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $reportDir "adapter_v1_$stamp.stdout.log"
$stderrPath = Join-Path $reportDir "adapter_v1_$stamp.stderr.log"
$wslConfig = Join-Path $env:USERPROFILE '.wslconfig'

if (-not (Test-Path -LiteralPath $wslConfig)) {
    throw 'The measured WSL memory configuration is missing.'
}
$wslSettings = Get-Content -LiteralPath $wslConfig -Raw
if ($wslSettings -notmatch '(?m)^\s*memory=19968MB\s*$' -or $wslSettings -notmatch '(?m)^\s*swap=16GB\s*$') {
    throw 'This guarded run requires WSL memory=19968MB and swap=16GB settings.'
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
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

$trainingProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Write-GuardStatus -State 'running' -AvailableGiB 0 -CommitHeadroomGiB 0
$safetyStop = $false
$samples = 0

while (-not $trainingProcess.HasExited) {
    $os = Get-CimInstance Win32_OperatingSystem
    $available = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $counters = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
    $committed = ($counters.CounterSamples | Where-Object Path -Like '*committed bytes').CookedValue
    $limit = ($counters.CounterSamples | Where-Object Path -Like '*commit limit').CookedValue
    $headroom = [math]::Round(($limit - $committed) / 1GB, 2)

    if ($available -lt 2.5 -or $headroom -lt 2.5) {
        $safetyStop = $true
        Write-GuardStatus -State 'safety_stop_requested' -AvailableGiB $available -CommitHeadroomGiB $headroom
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
    if (($samples % 100) -eq 0) {
        Write-GuardStatus -State 'running' -AvailableGiB $available -CommitHeadroomGiB $headroom
    }
    $samples++
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
