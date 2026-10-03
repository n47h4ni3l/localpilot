param(
    [string]$TaskName = "LocalPilot Background Worker",
    [string]$LegacyTaskName = "LocalPilot Idle Evolve",
    [ValidateRange(1, 3600)]
    [int]$PollSeconds = 30,
    [ValidateRange(1, 60)]
    [int]$WatchdogMinutes = 1,
    [ValidateRange(5, 120)]
    [int]$StartupTimeoutSeconds = 30,
    [string]$TrustedBranch = "main",
    [string]$PythonExecutable = "",
    [string]$ConfigPath = "",
    [switch]$RunAsAdministrator
)

$ErrorActionPreference = "Stop"

function Resolve-WorkerPython {
    param([string]$Executable, [string]$Root)

    $candidate = if ($Executable) { $Executable } else { Join-Path $Root ".venv\Scripts\python.exe" }
    $candidate = [System.IO.Path]::GetFullPath($candidate)
    if ([System.IO.Path]::GetFileName($candidate) -ieq "pythonw.exe") {
        $candidate = Join-Path ([System.IO.Path]::GetDirectoryName($candidate)) "python.exe"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "LocalPilot's Python interpreter is missing: $candidate. Pass -PythonExecutable with the interpreter used to launch LocalPilot."
    }
    $console = (Resolve-Path -LiteralPath $candidate).Path
    $gui = Join-Path ([System.IO.Path]::GetDirectoryName($console)) "pythonw.exe"
    if (-not (Test-Path -LiteralPath $gui -PathType Leaf)) {
        throw "LocalPilot's windowless Python launcher is missing: $gui. Select a Python installation that provides pythonw.exe."
    }
    return [pscustomobject]@{ Console = $console; Gui = $gui }
}

function Resolve-WorkerRunLevel {
    param([bool]$Specified, [bool]$Administrator, [string]$ExistingRunLevel)

    if ($Specified) {
        if ($Administrator) { return "Highest" }
        return "Limited"
    }
    if ($ExistingRunLevel -in @("Highest", "1")) { return "Highest" }
    if (-not $ExistingRunLevel -or $ExistingRunLevel -in @("Limited", "0")) { return "Limited" }
    throw "Cannot preserve the existing worker's unknown run level '$ExistingRunLevel'. Pass -RunAsAdministrator explicitly."
}

function Test-WorkerInterpreter {
    param($Process, [string]$SelectedGui, [string]$HostGui)
    if ($Process.ExecutablePath -ieq $SelectedGui) { return $true }
    if ($Process.ExecutablePath -ine $HostGui) { return $false }
    # Windows venv redirectors retain a launcher while the worker runs in
    # the base Python image. Require the exact selected launcher as parent.
    $launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $($Process.ParentProcessId)" -ErrorAction SilentlyContinue
    return [bool]($launcher -and $launcher.ExecutablePath -ieq $SelectedGui)
}

function Restore-PreviousWorkerTask {
    param([string]$Name, [string]$Xml, [bool]$Enabled, [bool]$Running)

    if (-not $Xml) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
        return
    }
    Register-ScheduledTask -TaskName $Name -Xml $Xml -Force -ErrorAction Stop | Out-Null
    if (-not $Enabled) {
        Disable-ScheduledTask -TaskName $Name -ErrorAction Stop | Out-Null
    } elseif ($Running) {
        Start-ScheduledTask -TaskName $Name -ErrorAction Stop
    }
}

if ($TaskName -eq $LegacyTaskName) {
    throw "TaskName and LegacyTaskName must be different so the legacy task remains available until verification succeeds."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$interpreters = Resolve-WorkerPython -Executable $PythonExecutable -Root $repoRoot
$pythonEntryPoint = $interpreters.Console
$pythonwEntryPoint = $interpreters.Gui
$hostPythonw = (& $pythonEntryPoint -c "from pathlib import Path; import sys; print(Path(getattr(sys, '_base_executable', sys.executable)).with_name('pythonw.exe'))" | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $hostPythonw -PathType Leaf)) {
    throw "Unable to identify the selected Python environment's Windows host executable."
}
$hostPythonw = (Resolve-Path -LiteralPath $hostPythonw).Path
$configPath = if ($ConfigPath) { [System.IO.Path]::GetFullPath($ConfigPath) } else { Join-Path $repoRoot "localpilot.toml" }
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

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "LocalPilot's config file is missing: $configPath. Supply a valid -ConfigPath."
}

Push-Location $repoRoot
try {
    $dataDir = (& $pythonEntryPoint -c "import sys, localpilot.background_worker; from pathlib import Path; from localpilot.config import load_config; print((Path(sys.argv[1]).resolve() / load_config(sys.argv[2]).agent.data_dir).resolve())" $repoRoot $configPath | Out-String).Trim()
    $configCheckExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($configCheckExit -ne 0 -or -not $dataDir) {
    throw "Unable to import the selected LocalPilot worker or resolve its configured data directory."
}
$workerPidPath = Join-Path $dataDir "background-worker.pid"
$auditPath = Join-Path $dataDir "audit.jsonl"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$existingXml = ""
$existingEnabled = $false
$existingRunning = $false
$existingRunLevel = ""
if ($existing) {
    $existingXml = Export-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-String
    $existingEnabled = [bool]$existing.Settings.Enabled
    $existingRunning = ($existing.State -eq "Running")
    $existingRunLevel = [string]$existing.Principal.RunLevel
}
$runLevel = Resolve-WorkerRunLevel -Specified $PSBoundParameters.ContainsKey('RunAsAdministrator') -Administrator ([bool]$RunAsAdministrator) -ExistingRunLevel $existingRunLevel
if ($runLevel -eq "Highest") {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $currentPrincipal = [System.Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $currentPrincipal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this installer from an administrator PowerShell session to register or preserve the elevated LocalPilot worker."
    }
}

$legacy = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
if ($legacy -and $legacy.State -eq "Running") {
    throw "Legacy task '$LegacyTaskName' is still running. Existing tasks were left unchanged; retry after that cycle finishes."
}

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
    -RunLevel $runLevel

$task = New-ScheduledTask `
    -Action $action `
    -Trigger @($logonTrigger, $watchdogTrigger) `
    -Settings $settings `
    -Principal $principal `
    -Description "Start one hidden LocalPilot worker at logon. A $WatchdogMinutes-minute trigger is ignored while it runs and relaunches it after a hard crash. The worker polls every $PollSeconds seconds and LocalPilot's existing gates remain authoritative."

try {
    if ($existingRunning) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    }
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force -ErrorAction Stop | Out-Null
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
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
            $candidate -and
            $candidate.Name -eq "pythonw.exe" -and
            (Test-WorkerInterpreter -Process $candidate -SelectedGui $pythonwEntryPoint -HostGui $hostPythonw) -and
            $candidate.CommandLine -like "*localpilot.background_worker*" -and
            $candidate.CommandLine -like "*$repoRoot*" -and
            $candidate.CommandLine -like "*$configPath*" -and
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
        throw "The selected hidden worker did not reach a verified windowless running state and begin an evolve cycle."
    }
} catch {
    $replacementError = $_
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    try {
        Restore-PreviousWorkerTask -Name $TaskName -Xml $existingXml -Enabled $existingEnabled -Running $existingRunning
    } catch {
        throw "Worker replacement failed, and the previous task could not be restored: $($_.Exception.Message). Original failure: $($replacementError.Exception.Message)"
    }
    $recovery = if ($existingXml) { "The previous task was restored" } else { "The failed replacement was removed" }
    throw "Worker replacement failed. $recovery and the legacy task was left unchanged. Cause: $($replacementError.Exception.Message)"
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
