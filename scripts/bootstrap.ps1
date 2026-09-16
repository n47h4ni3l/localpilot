$ErrorActionPreference = "Stop"

Write-Host "LocalPilot v0.1 bootstrap" -ForegroundColor Cyan

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $refreshed = @($machinePath, $userPath, $env:Path) | Where-Object { $_ }
    $env:Path = ($refreshed -join ";")
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageId,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    if ($env:OS -ne "Windows_NT") {
        throw "$DisplayName is required but automatic installation is only supported on Windows. Install it manually, then rerun bootstrap."
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$DisplayName is required, but WinGet is not available. Install Microsoft App Installer/WinGet, then rerun bootstrap."
    }

    Write-Host "$DisplayName is missing. Installing it with WinGet..." -ForegroundColor Yellow
    & $winget.Source install `
        --id $PackageId `
        --exact `
        --source winget `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "WinGet could not install $DisplayName ($PackageId). Exit code: $LASTEXITCODE."
    }

    Refresh-ProcessPath
}

function Test-Python311 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$PrefixArgs = @()
    )

    try {
        $output = & $Executable @PrefixArgs -c "import platform,sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info >= (3,11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            return [string]($output | Select-Object -Last 1)
        }
    } catch {
        # Treat launch failures (including the Windows Store python alias) as
        # an unusable interpreter and continue probing other candidates.
    }

    return $null
}

function Find-CompatiblePython {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $version = Test-Python311 -Executable $pyLauncher.Source -PrefixArgs @("-3")
        if ($version) {
            return [pscustomobject]@{
                Executable = $pyLauncher.Source
                PrefixArgs = @("-3")
                Version = $version
            }
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $version = Test-Python311 -Executable $pythonCommand.Source
        if ($version) {
            return [pscustomobject]@{
                Executable = $pythonCommand.Source
                PrefixArgs = @()
                Version = $version
            }
        }
    }

    $explicitCandidates = @()
    if ($env:LOCALAPPDATA) {
        $explicitCandidates += Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Programs\Python\Python*\python.exe") -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }
    if ($env:ProgramFiles) {
        $explicitCandidates += Get-ChildItem -Path (Join-Path $env:ProgramFiles "Python*\python.exe") -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }

    foreach ($candidate in ($explicitCandidates | Sort-Object -Descending -Unique)) {
        $version = Test-Python311 -Executable $candidate
        if ($version) {
            return [pscustomobject]@{
                Executable = $candidate
                PrefixArgs = @()
                Version = $version
            }
        }
    }

    return $null
}

function Find-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

function Find-DotNet8Sdk {
    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    $dotnetPath = if ($dotnet) { $dotnet.Source } else { $null }

    if (-not $dotnetPath -and $env:ProgramFiles) {
        $candidate = Join-Path $env:ProgramFiles "dotnet\dotnet.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $dotnetPath = $candidate
        }
    }

    if (-not $dotnetPath) {
        return $null
    }

    try {
        $sdkLines = @(& $dotnetPath --list-sdks 2>$null)
        if ($LASTEXITCODE -eq 0) {
            foreach ($sdkLine in $sdkLines) {
                if ($sdkLine -match '^\s*(\d+)\.' -and [int]$Matches[1] -ge 8) {
                    return $dotnetPath
                }
            }
        }
    } catch {
        return $null
    }

    return $null
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"
$pythonVersion = $null

if (Test-Path -LiteralPath $python -PathType Leaf) {
    $pythonVersion = Test-Python311 -Executable $python
    if (-not $pythonVersion) {
        Write-Host "LocalPilot's existing .venv is not usable with Python 3.11+. Rebuilding it..." -ForegroundColor Yellow
        Remove-Item -LiteralPath (Join-Path $PWD ".venv") -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $bootstrapPython = Find-CompatiblePython
    if (-not $bootstrapPython) {
        Install-WingetPackage -PackageId "Python.Python.3.12" -DisplayName "Python 3.12"
        $bootstrapPython = Find-CompatiblePython
    }

    if (-not $bootstrapPython) {
        throw "Python 3.11+ was installed or requested but could not be discovered in this PowerShell session. Open a new terminal and rerun bootstrap."
    }

    Write-Host "Creating isolated Python environment..."
    & $bootstrapPython.Executable @($bootstrapPython.PrefixArgs) -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create LocalPilot's virtual environment."
    }

    $pythonVersion = Test-Python311 -Executable $python
    if (-not $pythonVersion) {
        throw "LocalPilot's virtual environment was created but its Python interpreter could not be validated."
    }
}

$ollama = Find-Ollama
if (-not $ollama) {
    Install-WingetPackage -PackageId "Ollama.Ollama" -DisplayName "Ollama"
    $ollama = Find-Ollama
}
if (-not $ollama) {
    throw "Ollama was installed or requested but could not be discovered in this PowerShell session. Open a new terminal and rerun bootstrap."
}

Write-Host "Python: $pythonVersion"
Write-Host "Ollama: $((& $ollama --version) -join ' ')"

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in LocalPilot's virtual environment."
}

Write-Host "Installing and synchronizing LocalPilot runtime dependencies..."
& $python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "LocalPilot dependency installation failed. The installation is incomplete."
}

# Treat declared dependencies as installation requirements, not optional
# renderer fallbacks. This catches stale or partially-updated environments
# before LocalPilot is started through windowless pythonw.exe.
& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "LocalPilot dependency verification failed. Repair the virtual environment before starting LocalPilot."
}

$pillowVersion = & $python -c "from PIL import Image, ImageTk, __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Pillow could not be imported after installation. LocalPilot's illustrated avatar requires Pillow."
}
Write-Host "Pillow: $pillowVersion"

# Source checkouts build the same self-contained sensor helper that packaged
# releases carry inside LocalPilot. A complete source bootstrap therefore
# installs the required .NET SDK when the helper needs to be built.
if ($env:OS -eq "Windows_NT") {
    $rid = switch ($env:PROCESSOR_ARCHITECTURE) {
        "ARM64" { "win-arm64" }
        "x86" { "win-x86" }
        default { "win-x64" }
    }
    $hardwareProvider = Join-Path $PWD "localpilot\_hardware\$rid\LocalPilot.SystemSense.HardwareProvider.exe"
    if (-not (Test-Path -LiteralPath $hardwareProvider -PathType Leaf)) {
        $dotnet = Find-DotNet8Sdk
        if (-not $dotnet) {
            Install-WingetPackage -PackageId "Microsoft.DotNet.SDK.8" -DisplayName ".NET SDK 8"
            $dotnet = Find-DotNet8Sdk
        }

        if (-not $dotnet) {
            throw ".NET SDK 8+ was installed or requested but could not be discovered. Open a new terminal and rerun bootstrap."
        }

        & (Join-Path $PWD "scripts\build-systemsense-hardware.ps1") -RuntimeIdentifier $rid -DotNetPath $dotnet
        if ($LASTEXITCODE -ne 0) {
            throw "SystemSense hardware provider build failed."
        }
    } else {
        Write-Host "SystemSense hardware provider: bundled ($rid)"
    }
}

if (-not (Test-Path "localpilot.toml")) {
    Copy-Item "config.example.toml" "localpilot.toml"
    Write-Host "Created localpilot.toml from the example config."
}

$model = "gpt-oss:20b"
$models = (& $ollama list | Out-String)
if ($models -notmatch [regex]::Escape($model)) {
    Write-Host ""
    Write-Host "$model is not installed. Pulling it is a large download and may take some time." -ForegroundColor Yellow
    $answer = Read-Host "Download $model now? [y/N]"
    if ($answer -match '^[Yy]$') {
        & $ollama pull $model
    } else {
        Write-Host "Skipped model download. You can run 'ollama pull $model' later." -ForegroundColor Yellow
    }
} else {
    Write-Host "$model is already installed."
}

Write-Host ""
Write-Host "Bootstrap complete." -ForegroundColor Green
Write-Host "Run:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "localpilot doctor"
Write-Host "localpilot"
Write-Host ""
Write-Host "After pulling a newer LocalPilot revision, rerun .\scripts\bootstrap.ps1 to synchronize any new dependencies."
