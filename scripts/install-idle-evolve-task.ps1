param(
    [string]$TaskName = "LocalPilot Background Worker",
    [string]$LegacyTaskName = "LocalPilot Idle Evolve",
    [ValidateRange(1, 3600)]
    [int]$PollSeconds = 30,
    [ValidateRange(1, 60)]
    [int]$WatchdogMinutes = 1,
    [ValidateRange(5, 120)]
    [int]$StartupTimeoutSeconds = 30,
    [string]$TrustedBranch = "main"
)

$ErrorActionPreference = "Stop"

if ($TaskName -eq $LegacyTaskName) {
    throw "TaskName and LegacyTaskName must be different so the legacy task remains available until verification succeeds."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonEntryPoint = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonwEntryPoint = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
$configPath = Join-Path $repoRoot "localpilot.toml"
$git = Get-Command git -ErrorAction Stop

$topLevel = (& $git.Source -C $repoRoot rev-parse --show-toplevel 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or (Resolve-Path $topLevel).Path -ne $repoRoot) {
    throw "The LocalPilot directory is not the root of its Git checkout."
}

$branch = (& $git.Source -C $repoRoot branch --show-current | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne $TrustedBranch) {
    throw "Refusing to schedule branch '$branch'; expected trusted branch '$TrustedBranch'."
}

$changes = (& $git.Source -C $repoRoot status --porcelain --untracked-files=all | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $changes) {
    throw "Refusing to schedule a checkout with uncommitted work."
}

if (-not (Test-Path -LiteralPath $pythonEntryPoint -PathType Leaf)) {
    throw "LocalPilot's virtual environment is missing. Run scripts\bootstrap.ps1 first."
}
if (-not (Test-Path -LiteralPath $pythonwEntryPoint -PathType Leaf)) {
    throw "LocalPilot's windowless Python launcher is missing. Run scripts\bootstrap.ps1 first."
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "localpilot.toml is missing. Run scripts\bootstrap.ps1 first."
}

Push-Location $repoRoot
try {
    $dataDir = (& $pythonEntryPoint -c "import sys; from pathlib import Path; from localpilot.config import load_config; print((Path(sys.argv[1]).resolve() / load_config(sys.argv[2]).agent.data_dir).resolve())" $repoRoot $configPath | Out-String).Trim()
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0 -or -not $dataDir) {
    throw "Unable to resolve LocalPilot's configured data directory."
}
$workerPidPath = Join-Path $dataDir "background-worker.pid"
$auditPath = Join-Path $dataDir "audit.jsonl"

$principalName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$actionArguments = "-m localpilot.background_worker --root `"$repoRoot`" --config `"$configPath`" --interval-seconds $PollSeconds"
$action = New-ScheduledTaskAction `
    -Execute $pythonwEntryPoint `
    -Argument $actionArguments `
    -WorkingDirectory $repoRoot
$logonTrigger = New-ScheduledTaskTrigger `
    -AtLogOn `
    -User $principalName
$watchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes($WatchdogMinutes) `
    -RepetitionInterval (New-TimeSpan -Minutes $WatchdogMinutes)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -Hidden
$principal = New-ScheduledTaskPrincipal `
    -UserId $principalName `
    -LogonType Interactive `
    -RunLevel Limited

$task = New-ScheduledTask `
    -Action $action `
    -Trigger @($logonTrigger, $watchdogTrigger) `
    -Settings $settings `
    -Principal $principal `
    -Description "Start one hidden LocalPilot worker at logon. A $WatchdogMinutes-minute trigger is ignored while it runs and relaunches it after a hard crash. The worker polls every $PollSeconds seconds and LocalPilot's existing gates remain authoritative."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

$legacy = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
if ($legacy -and $LegacyTaskName -ne $TaskName -and $legacy.State -eq "Running") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    throw "Legacy task '$LegacyTaskName' is still running. It was left enabled and the replacement was rolled back; retry after that cycle finishes."
}

Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$verifiedProcess = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if (-not (Test-Path -LiteralPath $workerPidPath -PathType Leaf)) {
        continue
    }
    try {
        $workerPid = [int](Get-Content -LiteralPath $workerPidPath -Raw | ConvertFrom-Json).pid
    } catch {
        continue
    }
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $workerPid" -ErrorAction SilentlyContinue
    $nativeProcess = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
    $scheduled = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $cycleStarted = $false
    if (Test-Path -LiteralPath $auditPath -PathType Leaf) {
        $cycleStarted = [bool](Get-Content -LiteralPath $auditPath -Tail 200 | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } | Where-Object { $_.event -eq "background_worker_cycle_start" -and $_.pid -eq $workerPid } | Select-Object -First 1)
    }
    if (
        $candidate.Name -eq "pythonw.exe" -and
        $candidate.CommandLine -like "*localpilot.background_worker*" -and
        $candidate.CommandLine -like "*$repoRoot*" -and
        $nativeProcess -and
        $nativeProcess.MainWindowHandle -eq 0 -and
        $scheduled.State -eq "Running" -and
        $cycleStarted
    ) {
        $verifiedProcess = $candidate
        break
    }
}

if (-not $verifiedProcess) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    throw "The hidden worker did not reach a verified windowless running state and begin an evolve cycle. The replacement was removed and the legacy task was left unchanged."
}

Write-Host "Started hidden worker '$TaskName' (PID $($verifiedProcess.ProcessId)); it polls every $PollSeconds second(s)." -ForegroundColor Green
if ($legacy -and $LegacyTaskName -ne $TaskName) {
    try {
        if ($legacy.Settings.Enabled) {
            Disable-ScheduledTask -TaskName $LegacyTaskName -ErrorAction Stop | Out-Null
        }
        $verifiedLegacy = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction Stop
        if ($verifiedLegacy.Settings.Enabled) {
            throw "Windows still reports the legacy task as enabled."
        }
    } catch {
        throw "The hidden worker is verified and remains running, but '$LegacyTaskName' could not be disabled. Run Disable-ScheduledTask for that exact task from an administrator PowerShell session. Cause: $($_.Exception.Message)"
    }
    Write-Host "Disabled legacy repeating task '$LegacyTaskName' only after the replacement was verified."
}
Write-Host "The worker calls SelfDeveloper.run_once(force=False); all existing safety, authority, idle, resource, persistence, and recovery gates remain authoritative."
