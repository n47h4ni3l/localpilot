param(
    [string]$OutputDirectory = "",
    [ValidateSet('win-x64', 'win-arm64')][string]$RuntimeIdentifier = 'win-x64'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repoRoot 'dist' }
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null
Push-Location $repoRoot
try {
    $dirty = (& git status --porcelain --untracked-files=all | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Commit the release sources before packaging.' }
    $sha = (& git rev-parse HEAD | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $sha -notmatch '^[0-9a-f]{40}$') { throw 'Unable to identify release source.' }
    $package = Get-Content (Join-Path $repoRoot 'pyproject.toml') -Raw
    if ($package -notmatch '(?m)^version = "([^"]+)"') { throw 'Package version is missing.' }
    $version = $Matches[1]
    $buildId = [guid]::NewGuid().ToString('N')
    $helperDir = Join-Path $repoRoot "localpilot\_hardware\installer-build\$buildId\$RuntimeIdentifier"
    $helper = Join-Path $helperDir 'LocalPilot.SystemSense.HardwareProvider.exe'
    & (Join-Path $PSScriptRoot 'build-systemsense-hardware.ps1') -RuntimeIdentifier $RuntimeIdentifier -OutputDirectory $helperDir
    $scratch = Join-Path $repoRoot ('localpilot-data\installer-build\' + $buildId)
    $payload = Join-Path $scratch 'payload'
    $wrapper = Join-Path $scratch 'wrapper'
    New-Item -ItemType Directory -Path $payload, $wrapper -Force | Out-Null
    $sourceZip = Join-Path $scratch 'source.zip'
    & git archive --format=zip --output=$sourceZip $sha
    if ($LASTEXITCODE -ne 0) { throw 'Could not archive the committed release source.' }
    Expand-Archive -LiteralPath $sourceZip -DestinationPath $payload
    $bundled = Join-Path $payload "localpilot\_hardware\$RuntimeIdentifier"
    New-Item -ItemType Directory -Path $bundled -Force | Out-Null
    Copy-Item -LiteralPath $helper -Destination $bundled
    Copy-Item -LiteralPath (Join-Path $helperDir 'THIRD_PARTY_NOTICES.md') -Destination $bundled
    @{
        version = $version
        source_sha = $sha
        runtime_id = $RuntimeIdentifier
        hardware_provider_sha256 = (Get-FileHash -LiteralPath $helper -Algorithm SHA256).Hash.ToLowerInvariant()
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $payload '.localpilot-release.json') -Encoding utf8
    if (-not (Test-Path -LiteralPath (Join-Path $payload 'scripts/install-localpilot.ps1'))) {
        throw 'The one-click installer entrypoint is missing from the committed release.'
    }
    $stem = "LocalPilot-Setup-$version-$RuntimeIdentifier"
    $zip = Join-Path $output "$stem.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $zip) { throw "Output already exists: $zip" }
    [IO.Compression.ZipFile]::CreateFromDirectory($payload, $zip, [IO.Compression.CompressionLevel]::Optimal, $false)
    Copy-Item -LiteralPath $zip -Destination (Join-Path $wrapper 'payload.zip')
    @'
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Extract-Setup.ps1"
set "setupResult=%errorlevel%"
if not "%setupResult%"=="0" pause
exit /b %setupResult%
'@ | Set-Content -LiteralPath (Join-Path $wrapper 'Start-Setup.cmd') -Encoding ascii
    @'
$ErrorActionPreference = 'Stop'
$destination = Join-Path $env:TEMP ('LocalPilotSetup-' + [guid]::NewGuid().ToString('N'))
Expand-Archive -LiteralPath (Join-Path $PSScriptRoot 'payload.zip') -DestinationPath $destination
$global:LASTEXITCODE = 0
& (Join-Path $destination 'scripts/install-localpilot.ps1')
if (-not $? -or $LASTEXITCODE -ne 0) { throw "LocalPilot setup failed. Extracted evidence is preserved at $destination." }
$resolved = (Resolve-Path -LiteralPath $destination).Path
$tempPrefix = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
if (-not $resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($resolved) -notmatch '^LocalPilotSetup-[0-9a-f]{32}$') {
    throw 'Refusing to clean an unexpected extraction directory.'
}
Remove-Item -LiteralPath $resolved -Recurse -Force
'@ | Set-Content -LiteralPath (Join-Path $wrapper 'Extract-Setup.ps1') -Encoding utf8
    $exe = Join-Path $output "$stem.exe"
    if (Test-Path -LiteralPath $exe) { throw "Output already exists: $exe" }
    $embeddedZip = Join-Path $scratch 'embedded-wrapper.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($wrapper, $embeddedZip, [IO.Compression.CompressionLevel]::Optimal, $false)
    $bootstrapOutput = Join-Path $scratch 'bootstrap-publish'
    & dotnet publish (Join-Path $repoRoot 'tools\WindowsInstaller\WindowsInstaller.csproj') `
        --configuration Release --framework net8.0-windows --runtime $RuntimeIdentifier `
        --self-contained true --output $bootstrapOutput `
        "-p:PayloadArchive=$embeddedZip" "-p:Version=$version" `
        -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:EnableCompressionInSingleFile=true -p:PublishTrimmed=false `
        -p:DebugType=None -p:DebugSymbols=false
    if ($LASTEXITCODE -ne 0) { throw 'Windows installer bootstrap compilation failed; the source ZIP is preserved.' }
    $compiled = Join-Path $bootstrapOutput 'LocalPilot.Setup.exe'
    if (-not (Test-Path -LiteralPath $compiled -PathType Leaf)) { throw 'Windows installer executable was not produced.' }
    Copy-Item -LiteralPath $compiled -Destination $exe
    $header = [IO.File]::ReadAllBytes($exe)
    if ($header.Length -lt 2 -or $header[0] -ne 77 -or $header[1] -ne 90) { throw 'Installer has an invalid executable header.' }
    $roundtrip = Join-Path $scratch 'extracted-check'
    New-Item -ItemType Directory -Path $roundtrip -Force | Out-Null
    $check = Start-Process -FilePath $exe -ArgumentList @('"/extract:' + $roundtrip + '"') -WindowStyle Hidden -PassThru -Wait
    if ($check.ExitCode -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $roundtrip 'payload.zip'))) {
        throw 'The executable could not extract its embedded installation payload.'
    }
    $embeddedHash = (Get-FileHash -LiteralPath (Join-Path $roundtrip 'payload.zip') -Algorithm SHA256).Hash
    if ($embeddedHash -ne (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash) {
        throw 'Extracted installation payload differs from the release ZIP.'
    }
    foreach ($name in @('Start-Setup.cmd', 'Extract-Setup.ps1')) {
        if ((Get-FileHash -LiteralPath (Join-Path $roundtrip $name)).Hash -ne
            (Get-FileHash -LiteralPath (Join-Path $wrapper $name)).Hash) {
            throw "Extracted installation wrapper differs: $name"
        }
    }
    @($exe, $zip) | ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $([IO.Path]::GetFileName($_))"
    } | Set-Content -LiteralPath (Join-Path $output "$stem.sha256") -Encoding ascii
    Write-Host "Windows release package ready: $exe" -ForegroundColor Green
} finally { Pop-Location }
