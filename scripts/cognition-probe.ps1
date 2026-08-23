param(
    [string]$Config
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "LocalPilot virtual environment was not found. Run .\scripts\bootstrap.ps1 first."
}

$arguments = @("-m", "localpilot.cognition_probe")
if ($Config) {
    $arguments += @("--config", $Config)
}

& $python @arguments
exit $LASTEXITCODE
