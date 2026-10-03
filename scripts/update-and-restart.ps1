param(
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [string]$TaskName = "LocalPilot Background Worker",
    [switch]$SkipHardwareSmokeCheck,
    [switch]$SkipFetch,
    [string]$ExpectedOldSha = "",
    [string]$ExpectedTargetSha = "",
    [string]$ConfigPath = "",
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-UpdatePython {
    param([string]$Executable, [string]$Root)

    $candidate = if ($Executable) { $Executable } else { Join-Path $Root ".venv\Scripts\python.exe" }
    $candidate = [System.IO.Path]::GetFullPath($candidate)
    if ([System.IO.Path]::GetFileName($candidate) -ieq "pythonw.exe") {
        $candidate = Join-Path ([System.IO.Path]::GetDirectoryName($candidate)) "python.exe"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "LocalPilot's Python interpreter is missing: $candidate. Pass -PythonExecutable with the interpreter used to launch LocalPilot."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-UpdateGuiPython {
    param([string]$Executable)

    $pythonw = Join-Path ([System.IO.Path]::GetDirectoryName($Executable)) "pythonw.exe"
    if (Test-Path -LiteralPath $pythonw -PathType Leaf) {
        return $pythonw
    }
    return $Executable
}

function Assert-UpdateAdministrator {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run the LocalPilot updater from an administrator PowerShell session so elevated desktop and sensor processes can be stopped and restarted reliably."
    }
}

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

function Get-GitCommit {
    param([string]$Revision)
    $value = (& git rev-parse --verify "$Revision^{commit}" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $value) {
        throw "Could not resolve Git revision '$Revision'."
    }
    return ($value | Select-Object -First 1).Trim()
}

function Assert-CleanWorkingTree {
    $dirty = @(git status --porcelain --untracked-files=all)
    Assert-LastExitCode "Could not inspect the Git working tree."
    if ($dirty.Count -gt 0) {
        $details = ($dirty -join [Environment]::NewLine)
        throw "Working tree is not clean. Commit or stash your changes first:`n$details"
    }
}

function Start-LocalPilotDesktop {
    $arguments = @("-m", "localpilot.cli")
    if ($resolvedConfig) {
        # --config is a root CLI option and must precede the desktop subcommand.
        $arguments += "--config"
        $arguments += ('"{0}"' -f $resolvedConfig)
    }
    $arguments += "desktop"

    Start-Process -FilePath $guiPython -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden
}

if ($Remote -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "Unsafe Git remote name '$Remote'."
}
if ($Branch -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$') {
    throw "Unsafe Git branch name '$Branch'."
}
if ($ExpectedOldSha -and $ExpectedOldSha -notmatch '^[0-9a-fA-F]{40}$') {
    throw "ExpectedOldSha must be a full 40-character Git commit SHA."
}
if ($ExpectedTargetSha -and $ExpectedTargetSha -notmatch '^[0-9a-fA-F]{40}$') {
    throw "ExpectedTargetSha must be a full 40-character Git commit SHA."
}
if ($SkipFetch -and -not $ExpectedTargetSha) {
    throw "SkipFetch requires ExpectedTargetSha so the already-fetched update target is explicit."
}

Write-Host "`n=== LocalPilot clean update/restart ===" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

$python = Resolve-UpdatePython -Executable $PythonExecutable -Root $repoRoot
$guiPython = Get-UpdateGuiPython -Executable $python
$resolvedConfig = ""
if ($ConfigPath) {
    $resolvedConfig = [System.IO.Path]::GetFullPath($ConfigPath)
    if (-not (Test-Path -LiteralPath $resolvedConfig -PathType Leaf)) {
        throw "Configured LocalPilot config file does not exist: $resolvedConfig"
    }
}
Write-Host "Python: $python"
Assert-UpdateAdministrator

# These are generated .NET outputs only. Remove them before the cleanliness
# check so older checkouts created before the ignore rules can recover cleanly.
$knownBuildArtifacts = @(
    (Join-Path $repoRoot "tools\SystemSense.HardwareProvider\bin"),
    (Join-Path $repoRoot "tools\SystemSense.HardwareProvider\obj")
)
foreach ($artifact in $knownBuildArtifacts) {
    if (Test-Path -LiteralPath $artifact) {
        $resolvedArtifact = (Resolve-Path -LiteralPath $artifact).Path
        $repoPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (-not $resolvedArtifact.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove generated build artifact outside the checkout: $resolvedArtifact"
        }
        $artifactItem = Get-Item -LiteralPath $resolvedArtifact -Force
        if ($artifactItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to remove a redirected generated build directory: $resolvedArtifact"
        }
        Write-Host "Removing generated build artifact: $resolvedArtifact"
        Remove-Item -LiteralPath $resolvedArtifact -Recurse -Force
    }
}

Push-Location $repoRoot
$task = $null
$taskWasEnabled = $false
$taskWasRunning = $false
$taskTemporarilyDisabled = $false
$processesStopped = $false
$updateSucceeded = $false
$oldSha = ""
$targetSha = ""

try {
    # Everything through the fetch/target validation is deliberately done while
    # LocalPilot is still running. A transient DNS/GitHub failure therefore
    # leaves the current desktop and workers untouched.
    Assert-CleanWorkingTree

    $currentBranch = (& git branch --show-current).Trim()
    Assert-LastExitCode "Could not inspect the current Git branch."
    if ($currentBranch -ne $Branch) {
        throw "Update requires branch '$Branch'; current checkout is '$currentBranch'. Switch branches before running the updater."
    }

    $oldSha = Get-GitCommit "HEAD"
    if ($ExpectedOldSha -and $oldSha -ne $ExpectedOldSha.ToLowerInvariant()) {
        throw "Local HEAD changed before update handoff. Expected $ExpectedOldSha, found $oldSha."
    }

    if (-not $SkipFetch) {
        Write-Host "Checking $Remote/$Branch for updates (LocalPilot remains running)..."
        $remoteRef = "refs/remotes/$Remote/$Branch"
        & git fetch --no-tags --prune $Remote "+refs/heads/$($Branch):$remoteRef"
        Assert-LastExitCode "Could not fetch '$Remote/$Branch'. LocalPilot was left running unchanged."
    } else {
        Write-Host "Using previously fetched update target (no network access required after handoff)..."
    }

    if ($ExpectedTargetSha) {
        $targetSha = Get-GitCommit $ExpectedTargetSha
        if (-not $SkipFetch) {
            $fetchedTarget = Get-GitCommit "refs/remotes/$Remote/$Branch"
            if ($fetchedTarget -ne $targetSha) {
                throw "Fetched '$Remote/$Branch' no longer matches the expected update target."
            }
        }
    } else {
        $targetSha = Get-GitCommit "refs/remotes/$Remote/$Branch"
    }

    & git merge-base --is-ancestor $oldSha $targetSha
    if ($LASTEXITCODE -ne 0) {
        throw "Update refused because local '$Branch' is ahead of or diverged from target $($targetSha.Substring(0, 7))."
    }

    Write-Host ("Update preflight complete: {0} -> {1}. LocalPilot is still running." -f $oldSha.Substring(0, 7), $targetSha.Substring(0, 7)) -ForegroundColor Green

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
    $processesStopped = $true
    Start-Sleep -Seconds 2

    # The preflight was performed while LocalPilot was live. Reconfirm that the
    # checkout did not change between preflight and process shutdown.
    Assert-CleanWorkingTree
    $headAfterStop = Get-GitCommit "HEAD"
    if ($headAfterStop -ne $oldSha) {
        throw "Local HEAD changed during update shutdown; refusing to apply the fetched target."
    }

    if ($oldSha -ne $targetSha) {
        Write-Host "Fast-forwarding $Branch to already-fetched target $($targetSha.Substring(0, 7))..."
        & git merge --ff-only --no-edit $targetSha
        Assert-LastExitCode "Could not fast-forward '$Branch' to the validated target."
        $mergedSha = Get-GitCommit "HEAD"
        if ($mergedSha -ne $targetSha) {
            throw "Fast-forward verification failed: HEAD is not the validated target."
        }
    } else {
        Write-Host "$Branch is already current; continuing with environment refresh and clean restart."
    }

    Write-Host "Refreshing LocalPilot environment..."
    # Refresh the interpreter that launched LocalPilot. Bootstrap creates a new
    # repo .venv and must not silently switch an existing installed runtime.
    & $python -m pip install -e .
    Assert-LastExitCode "LocalPilot dependency installation failed in the selected Python environment."
    & $python -m pip check
    Assert-LastExitCode "LocalPilot dependency verification failed in the selected Python environment."
    & $python -c "import localpilot, localpilot.native_avatar_companion; from PIL import Image, ImageTk"
    Assert-LastExitCode "Updated LocalPilot desktop dependencies could not be imported."

    $runtimeIdentifier = Get-RuntimeIdentifier
    Write-Host "Rebuilding SystemSense hardware provider ($runtimeIdentifier)..."
    # The build script uses terminating errors. LASTEXITCODE can retain an
    # unrelated native command's result after a successful PowerShell script.
    & (Join-Path $repoRoot "scripts\build-systemsense-hardware.ps1") -RuntimeIdentifier $runtimeIdentifier

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

    Write-Host "Starting fresh LocalPilot desktop..."
    Start-LocalPilotDesktop
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

    if ($processesStopped -and -not $updateSucceeded) {
        try {
            Write-Warning "Update/rebuild failed after LocalPilot was stopped. Attempting to relaunch the current checkout."
            Start-LocalPilotDesktop
        } catch {
            Write-Warning "Could not relaunch LocalPilot after the failed update: $($_.Exception.Message)"
        }
    }

    Pop-Location
}
