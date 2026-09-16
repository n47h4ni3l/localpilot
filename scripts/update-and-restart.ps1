param(
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [string]$TaskName = "LocalPilot Background Worker",
    [switch]$SkipHardwareSmokeCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$localpilot = Join-Path $repoRoot ".venv\Scripts\localpilot.exe"

function Assert-LastExitCode {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

function Get-RuntimeIdentifier {
    $architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
    switch ($architecture) {
        "arm64" { return "win-arm64" }
        "x86" { return "win-x86" }
        default { return "win-x64" }
    }
}

Write-Host "`n=== LocalPilot clean update/restart ===" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "LocalPilot's virtual environment is missing. Run .\scripts\bootstrap.ps1 once, then rerun this script."
}

# These are generated .NET outputs only. Remove them before the cleanliness
# check so older checkouts created before the ignore rules can recover cleanly.
$knownBuildArtifacts = @(
    (Join-Path $repoRoot "tools\SystemSense.HardwareProvider\bin"),
    (Join-Path $repoRoot "tools\SystemSense.HardwareProvider\obj")
)
foreach ($artifact in $knownBuildArtifacts) {
    if (Test-Path -LiteralPath $artifact) {
        Write-Host "Removing generated build artifact: $artifact"
        Remove-Item -LiteralPath $artifact -Recurse -Force
    }
}

Push-Location $repoRoot
$task = $null
$taskWasEnabled = $false
$taskWasRunning = $false
$taskTemporarilyDisabled = $false
$updateSucceeded = $false

try {
    $dirty = @(git status --porcelain --untracked-files=all)
    Assert-LastExitCode "Could not inspect the Git working tree."
    if ($dirty.Count -gt 0) {
        $details = ($dirty -join [Environment]::NewLine)
        throw "Working tree is not clean. Commit or stash your changes first:`n$details"
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        $taskWasEnabled = ($task.State -ne "Disabled")
        $taskWasRunning = ($task.State -eq "Running")

        if ($taskWasEnabled) {
            Write-Host "Stopping scheduled background worker..."
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Disable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
            $taskTemporarilyDisabled = $true
        }
    }

    Write-Host "Stopping all LocalPilot processes..."
    & $python -c "from pathlib import Path; from localpilot.desktop_updater import _stop_localpilot_processes; _stop_localpilot_processes(Path.cwd())"
    Assert-LastExitCode "LocalPilot processes could not be stopped cleanly."
    Start-Sleep -Seconds 2

    Write-Host "Updating $Branch from $Remote..."
    git switch $Branch
    Assert-LastExitCode "Could not switch to branch '$Branch'."
    git fetch $Remote $Branch
    Assert-LastExitCode "Could not fetch '$Remote/$Branch'."
    git pull --ff-only $Remote $Branch
    Assert-LastExitCode "Could not fast-forward '$Branch' from '$Remote/$Branch'."

    Write-Host "Refreshing LocalPilot environment..."
    & (Join-Path $repoRoot "scripts\bootstrap.ps1")
    Assert-LastExitCode "LocalPilot bootstrap failed."

    $runtimeIdentifier = Get-RuntimeIdentifier
    Write-Host "Rebuilding SystemSense hardware provider ($runtimeIdentifier)..."
    & (Join-Path $repoRoot "scripts\build-systemsense-hardware.ps1") -RuntimeIdentifier $runtimeIdentifier
    Assert-LastExitCode "SystemSense hardware provider build failed."

    if (-not $SkipHardwareSmokeCheck) {
        $provider = Join-Path $repoRoot "localpilot\_hardware\$runtimeIdentifier\LocalPilot.SystemSense.HardwareProvider.exe"
        if (-not (Test-Path -LiteralPath $provider -PathType Leaf)) {
            throw "SystemSense hardware provider executable was not produced: $provider"
        }

        Write-Host "Checking SystemSense hardware sensors..."
        $snapshotLines = @(& $provider --snapshot)
        Assert-LastExitCode "SystemSense hardware provider snapshot failed."
        $snapshotText = ($snapshotLines -join "`n").Trim()
        if (-not $snapshotText) {
            throw "SystemSense hardware provider returned an empty snapshot."
        }
        try {
            $snapshot = $snapshotText | ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "SystemSense hardware provider returned invalid JSON: $($_.Exception.Message)"
        }
        if (-not $snapshot.ok) {
            throw "SystemSense hardware provider reported a failed snapshot."
        }

        $sensors = @($snapshot.sensors)
        $temperatureSensors = @(
            $sensors | Where-Object {
                $_.SensorType -eq "Temperature" -and $null -ne $_.Value
            }
        )
        Write-Host ("SystemSense sensor check: {0} sensors, {1} live temperature sensors." -f $sensors.Count, $temperatureSensors.Count) -ForegroundColor Green
        if ($temperatureSensors.Count -eq 0) {
            Write-Warning "The provider returned no live temperature sensors. LocalPilot will start, but the temperature card may remain unavailable."
        }
    }

    if (-not (Test-Path -LiteralPath $localpilot -PathType Leaf)) {
        throw "LocalPilot launcher is missing after bootstrap: $localpilot"
    }

    Write-Host "Starting fresh LocalPilot desktop..."
    Start-Process -FilePath $localpilot -ArgumentList @("desktop") -WorkingDirectory $repoRoot
    $updateSucceeded = $true

    Write-Host "`nLocalPilot updated, rebuilt, and restarted." -ForegroundColor Green
}
finally {
    if ($null -ne $task -and $taskWasEnabled -and $taskTemporarilyDisabled) {
        try {
            Write-Host "Restoring scheduled background worker..."
            Enable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
            if ($updateSucceeded -or $taskWasRunning) {
                Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
            }
        } catch {
            Write-Warning "Could not fully restore scheduled task '$TaskName': $($_.Exception.Message)"
        }
    }
    Pop-Location
}
