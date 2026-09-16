$ErrorActionPreference = "Stop"

Write-Host "LocalPilot v0.1 bootstrap" -ForegroundColor Cyan

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

$python = Join-Path $PWD ".venv\Scripts\python.exe"
$pythonVersion = $null

if (Test-Path -LiteralPath $python -PathType Leaf) {
    $pythonVersion = Test-Python311 -Executable $python
    if (-not $pythonVersion) {
        throw "LocalPilot's existing .venv does not contain a usable Python 3.11+ interpreter. Recreate the virtual environment, then rerun this script."
    }
} else {
    $bootstrapPython = $null
    $bootstrapPythonArgs = @()

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $candidateVersion = Test-Python311 -Executable $pyLauncher.Source -PrefixArgs @("-3")
        if ($candidateVersion) {
            $bootstrapPython = $pyLauncher.Source
            $bootstrapPythonArgs = @("-3")
        }
    }

    if (-not $bootstrapPython) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $candidateVersion = Test-Python311 -Executable $pythonCommand.Source
            if ($candidateVersion) {
                $bootstrapPython = $pythonCommand.Source
            }
        }
    }

    if (-not $bootstrapPython) {
        throw "Python 3.11 or newer was not found. Install Python 3.11+ (or make the Python launcher available), then rerun this script."
    }

    Write-Host "Creating isolated Python environment..."
    & $bootstrapPython @bootstrapPythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create LocalPilot's virtual environment."
    }

    $pythonVersion = Test-Python311 -Executable $python
    if (-not $pythonVersion) {
        throw "LocalPilot's virtual environment was created but its Python interpreter could not be validated."
    }
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was not found on PATH. Install Ollama for Windows, then rerun this script."
}

Write-Host "Python: $pythonVersion"
Write-Host "Ollama: $((ollama --version) -join ' ')"

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
# releases carry inside LocalPilot. The helper is optional only when a usable
# .NET 8+ SDK is unavailable during development; SystemSense retains its
# native/WMI fallbacks.
if ($env:OS -eq "Windows_NT") {
    $rid = switch ($env:PROCESSOR_ARCHITECTURE) {
        "ARM64" { "win-arm64" }
        "x86" { "win-x86" }
        default { "win-x64" }
    }
    $hardwareProvider = Join-Path $PWD "localpilot\_hardware\$rid\LocalPilot.SystemSense.HardwareProvider.exe"
    if (-not (Test-Path -LiteralPath $hardwareProvider -PathType Leaf)) {
        $dotnetCommand = Get-Command dotnet -ErrorAction SilentlyContinue
        $hasDotNet8Sdk = $false

        if ($dotnetCommand) {
            try {
                $sdkLines = @(& $dotnetCommand.Source --list-sdks 2>$null)
                if ($LASTEXITCODE -eq 0) {
                    foreach ($sdkLine in $sdkLines) {
                        if ($sdkLine -match '^\s*(\d+)\.' -and [int]$Matches[1] -ge 8) {
                            $hasDotNet8Sdk = $true
                            break
                        }
                    }
                }
            } catch {
                $hasDotNet8Sdk = $false
            }
        }

        if ($hasDotNet8Sdk) {
            & (Join-Path $PWD "scripts\build-systemsense-hardware.ps1") -RuntimeIdentifier $rid
            if ($LASTEXITCODE -ne 0) {
                throw "SystemSense hardware provider build failed."
            }
        } else {
            Write-Host "SystemSense hardware provider was not built because a .NET SDK 8 or newer is not installed. Native/WMI telemetry will remain available; packaged releases include the helper automatically." -ForegroundColor Yellow
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
$models = (ollama list | Out-String)
if ($models -notmatch [regex]::Escape($model)) {
    Write-Host ""
    Write-Host "$model is not installed. Pulling it is a large download and may take some time." -ForegroundColor Yellow
    $answer = Read-Host "Download $model now? [y/N]"
    if ($answer -match '^[Yy]$') {
        ollama pull $model
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
