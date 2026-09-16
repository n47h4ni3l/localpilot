param(
    [ValidateSet("win-x64", "win-x86", "win-arm64")]
    [string]$RuntimeIdentifier = "win-x64"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repoRoot "tools\SystemSense.HardwareProvider\SystemSense.HardwareProvider.csproj"
$output = Join-Path $repoRoot "localpilot\_hardware\$RuntimeIdentifier"

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw ".NET SDK 8 or newer is required to build the SystemSense hardware provider. Packaged LocalPilot releases ship the published helper and do not require the SDK at runtime."
}

$major = [int]((& dotnet --version).Split('.')[0])
if ($major -lt 8) {
    throw ".NET SDK 8 or newer is required to build the SystemSense hardware provider."
}

if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Recurse -Force
}
New-Item -ItemType Directory -Path $output -Force | Out-Null

Write-Host "Publishing bundled SystemSense hardware provider ($RuntimeIdentifier)..." -ForegroundColor Cyan
& dotnet publish $project `
    --configuration Release `
    --framework net8.0-windows `
    --runtime $RuntimeIdentifier `
    --self-contained true `
    --output $output `
    -p:PublishSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -p:PublishTrimmed=false `
    -p:DebugType=None `
    -p:DebugSymbols=false
if ($LASTEXITCODE -ne 0) {
    throw "SystemSense hardware provider publish failed."
}

$exe = Join-Path $output "LocalPilot.SystemSense.HardwareProvider.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Publish completed without the expected hardware provider executable."
}

$notice = Join-Path $repoRoot "tools\SystemSense.HardwareProvider\THIRD_PARTY_NOTICES.md"
if (Test-Path -LiteralPath $notice -PathType Leaf) {
    Copy-Item -LiteralPath $notice -Destination (Join-Path $output "THIRD_PARTY_NOTICES.md") -Force
}

Write-Host "SystemSense hardware provider ready: $exe" -ForegroundColor Green
