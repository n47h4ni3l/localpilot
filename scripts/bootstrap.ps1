param(
    [switch]$Unattended,
    [switch]$RuntimeOnly,
    [string]$PythonPath = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {

Write-Host "LocalPilot installation" -ForegroundColor Cyan

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
        --silent `
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
        foreach ($selector in @('-3.12', '-3')) {
            $version = Test-Python311 -Executable $pyLauncher.Source -PrefixArgs @($selector)
            if ($version) {
                return [pscustomobject]@{
                    Executable = $pyLauncher.Source
                    PrefixArgs = @($selector)
                    Version = $version
                }
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

function Read-OllamaModels {
    # Windows PowerShell turns redirected native stderr into ErrorRecords.
    # A stopped Ollama service is expected here and must remain recoverable.
    $ErrorActionPreference = 'Continue'
    $lines = @(& $ollama list 2>$null)
    return [pscustomobject]@{ Lines = $lines; ExitCode = $LASTEXITCODE }
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

$python = if ($PythonPath) { (Resolve-Path -LiteralPath $PythonPath).Path } else { Join-Path $repoRoot ".venv\Scripts\python.exe" }
$pythonVersion = $null

if (Test-Path -LiteralPath $python -PathType Leaf) {
    $pythonVersion = Test-Python311 -Executable $python
    if (-not $pythonVersion) {
        if ($PythonPath) { throw "The supplied Python interpreter requires Python 3.11+." }
        Write-Host "LocalPilot's existing .venv is not usable with Python 3.11+. Rebuilding it..." -ForegroundColor Yellow
        $environmentPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv"))
        $preservedEnvironment = [IO.Path]::GetFullPath((Join-Path $repoRoot (".venv.unusable-" + [Guid]::NewGuid().ToString('N'))))
        if ((Split-Path $environmentPath) -ne $repoRoot -or (Split-Path $preservedEnvironment) -ne $repoRoot) {
            throw "Virtual environment recovery escaped this checkout."
        }
        Move-Item -LiteralPath $environmentPath -Destination $preservedEnvironment
        Write-Host "Previous environment preserved at $preservedEnvironment"
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    if ($PythonPath) { throw "The supplied Python interpreter is missing." }
    $partialEnvironment = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv"))
    if (Test-Path -LiteralPath $partialEnvironment) {
        $preservedEnvironment = [IO.Path]::GetFullPath((Join-Path $repoRoot (".venv.unusable-" + [Guid]::NewGuid().ToString('N'))))
        if ((Split-Path $partialEnvironment) -ne $repoRoot -or (Split-Path $preservedEnvironment) -ne $repoRoot) {
            throw "Virtual environment recovery escaped this checkout."
        }
        Move-Item -LiteralPath $partialEnvironment -Destination $preservedEnvironment
        Write-Host "Incomplete environment preserved at $preservedEnvironment"
    }
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
if ($RuntimeOnly) {
    & $python -m pip install -e "."
} else {
    & $python -m pip install -e ".[dev]"
}
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
    $rid = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
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

$model = (& $python -c "import sys,tomllib; from pathlib import Path; print(tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('model',{}).get('name','gpt-oss:20b'))" (Join-Path $repoRoot 'localpilot.toml') | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $model) { throw "Could not read the configured LocalPilot model." }
$modelProbe = Read-OllamaModels
$modelLines = $modelProbe.Lines
if ($modelProbe.ExitCode -ne 0) {
    Write-Host "Starting the local Ollama server..."
    Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
    $ollamaReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        $modelProbe = Read-OllamaModels
        $modelLines = $modelProbe.Lines
        if ($modelProbe.ExitCode -eq 0) { $ollamaReady = $true; break }
    }
    if (-not $ollamaReady) { throw "Ollama did not become ready. Check the Ollama installation, then retry." }
}
$installedModels = @($modelLines | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] })
if ($installedModels -notcontains $model) {
    Write-Host ""
    Write-Host "$model is not installed. Pulling it is a large download and may take some time." -ForegroundColor Yellow
    if ($Unattended -and $model -like 'nestra:*') {
        throw "Configured local model '$model' is missing. Restore that model before starting LocalPilot; its configuration was preserved."
    }
    $answer = if ($Unattended) { 'y' } else { Read-Host "Download $model now? [y/N]" }
    if ($answer -match '^[Yy]$') {
        & $ollama pull $model
        if ($LASTEXITCODE -ne 0) { throw "Ollama could not download '$model'. Retry installation to resume the download." }
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
} finally { Pop-Location }
