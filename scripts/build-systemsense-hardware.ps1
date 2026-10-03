param(
    [ValidateSet("win-x64", "win-x86", "win-arm64")]
    [string]$RuntimeIdentifier = "win-x64",
    [string]$DotNetPath,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repoRoot "tools\SystemSense.HardwareProvider\SystemSense.HardwareProvider.csproj"
$hardwareRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "localpilot\_hardware"))
$output = if ($OutputDirectory) { [IO.Path]::GetFullPath($OutputDirectory) } else { Join-Path $hardwareRoot $RuntimeIdentifier }
$outputParent = Split-Path $output
$outputName = Split-Path $output -Leaf
$buildId = [Guid]::NewGuid().ToString('N')
$staging = Join-Path $outputParent "$outputName.build-$buildId"
$backup = Join-Path $outputParent "$outputName.previous-$buildId"

function Assert-HardwareDirectory {
    param([string]$Path)
    $resolved = [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $prefix = $hardwareRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Hardware build directory escaped its expected root: $resolved"
    }
}
Assert-HardwareDirectory $output
Assert-HardwareDirectory $staging
Assert-HardwareDirectory $backup

if ($DotNetPath) {
    if (-not (Test-Path -LiteralPath $DotNetPath -PathType Leaf)) {
        throw "The supplied dotnet executable does not exist: $DotNetPath"
    }
    $dotnetExecutable = $DotNetPath
} else {
    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $dotnet) {
        throw ".NET SDK 8 or newer is required to build the SystemSense hardware provider."
    }
    $dotnetExecutable = $dotnet.Source
}

try {
    $sdkLines = @(& $dotnetExecutable --list-sdks 2>$null)
} catch {
    throw ".NET was found, but the installed SDKs could not be queried. Install .NET SDK 8 or newer to build the SystemSense hardware provider."
}

if ($LASTEXITCODE -ne 0) {
    throw ".NET was found, but the installed SDKs could not be queried. Install .NET SDK 8 or newer to build the SystemSense hardware provider."
}

$hasCompatibleSdk = $false
foreach ($sdkLine in $sdkLines) {
    if ($sdkLine -match '^\s*(\d+)\.' -and [int]$Matches[1] -ge 8) {
        $hasCompatibleSdk = $true
        break
    }
}

if (-not $hasCompatibleSdk) {
    throw ".NET SDK 8 or newer is required to build the SystemSense hardware provider. The .NET host may be installed, but no compatible SDK is available."
}

try {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    Write-Host "Publishing bundled SystemSense hardware provider ($RuntimeIdentifier)..." -ForegroundColor Cyan
    & $dotnetExecutable publish $project `
        --configuration Release `
        --framework net8.0-windows `
        --runtime $RuntimeIdentifier `
        --self-contained true `
        --output $staging `
        -p:PublishSingleFile=true `
        -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:PublishTrimmed=false `
        -p:DebugType=None `
        -p:DebugSymbols=false
    if ($LASTEXITCODE -ne 0) {
        throw "SystemSense hardware provider publish failed. Existing provider was preserved."
    }

    $stagedExe = Join-Path $staging "LocalPilot.SystemSense.HardwareProvider.exe"
    if (-not (Test-Path -LiteralPath $stagedExe -PathType Leaf)) {
        throw "Publish completed without the expected hardware provider executable. Existing provider was preserved."
    }
    $notice = Join-Path $repoRoot "tools\SystemSense.HardwareProvider\THIRD_PARTY_NOTICES.md"
    if (Test-Path -LiteralPath $notice -PathType Leaf) {
        Copy-Item -LiteralPath $notice -Destination (Join-Path $staging "THIRD_PARTY_NOTICES.md") -Force
    }

    # Keep the installed helper intact until a complete replacement is ready.
    # If replacement fails, restore it; a failed publish never removes it.
    if (Test-Path -LiteralPath $output) {
        Move-Item -LiteralPath $output -Destination $backup
    }
    try {
        Move-Item -LiteralPath $staging -Destination $output
    } catch {
        if (Test-Path -LiteralPath $backup) {
            Move-Item -LiteralPath $backup -Destination $output
        }
        throw
    }
    if (Test-Path -LiteralPath $backup) {
        Assert-HardwareDirectory $backup
        Remove-Item -LiteralPath $backup -Recurse -Force
    }
    Write-Host "SystemSense hardware provider ready: $(Join-Path $output 'LocalPilot.SystemSense.HardwareProvider.exe')" -ForegroundColor Green
} finally {
    if (Test-Path -LiteralPath $staging) {
        Assert-HardwareDirectory $staging
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
