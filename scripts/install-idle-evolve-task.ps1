param(
    [string]$TaskName = "LocalPilot Idle Evolve",
    [ValidateRange(1, 60)]
    [int]$PollMinutes = 5,
    [string]$TrustedBranch = "main"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonEntryPoint = Join-Path $repoRoot ".venv\Scripts\localpilot.exe"
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
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "localpilot.toml is missing. Run scripts\bootstrap.ps1 first."
}

$actionArguments = "--config `"$configPath`" evolve"
$action = New-ScheduledTaskAction `
    -Execute $pythonEntryPoint `
    -Argument $actionArguments `
    -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $PollMinutes)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Poll LocalPilot while this user is signed in; LocalPilot's own idle/resource gate decides whether an evolve cycle may run."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Write-Host "Scheduled '$TaskName' every $PollMinutes minute(s)." -ForegroundColor Green
Write-Host "The task never uses --force; LocalPilot's idle and resource gates remain authoritative."
