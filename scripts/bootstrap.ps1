$ErrorActionPreference = "Stop"

Write-Host "LocalPilot v0.1 bootstrap" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.11 or newer, then rerun this script."
}

$versionOk = python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if ($LASTEXITCODE -ne 0) {
    $actual = python -c "import platform; print(platform.python_version())"
    throw "Python $actual is too old. LocalPilot requires Python 3.11+."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was not found on PATH. Install Ollama for Windows, then rerun this script."
}

$pyver = python -c "import platform; print(platform.python_version())"
Write-Host "Python: $pyver"
Write-Host "Ollama: $((ollama --version) -join ' ')"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating isolated Python environment..."
    python -m venv .venv
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"
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
# releases carry inside LocalPilot. The helper is optional only when the SDK is
# unavailable during development; SystemSense retains its native/WMI fallbacks.
if ($env:OS -eq "Windows_NT") {
    $rid = switch ($env:PROCESSOR_ARCHITECTURE) {
        "ARM64" { "win-arm64" }
        "x86" { "win-x86" }
        default { "win-x64" }
    }
    $hardwareProvider = Join-Path $PWD "localpilot\_hardware\$rid\LocalPilot.SystemSense.HardwareProvider.exe"
    if (-not (Test-Path -LiteralPath $hardwareProvider -PathType Leaf)) {
        if (Get-Command dotnet -ErrorAction SilentlyContinue) {
            & (Join-Path $PWD "scripts\build-systemsense-hardware.ps1") -RuntimeIdentifier $rid
            if ($LASTEXITCODE -ne 0) {
                throw "SystemSense hardware provider build failed."
            }
        } else {
            Write-Host "SystemSense hardware provider was not built because the .NET SDK is not installed. Native/WMI telemetry will remain available; packaged releases include the helper automatically." -ForegroundColor Yellow
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
