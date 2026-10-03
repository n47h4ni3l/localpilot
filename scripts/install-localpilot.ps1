param(
    [string]$InstallDirectory = "",
    [switch]$SkipLaunch,
    [string]$ShortcutPath = ""
)

$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:restartRequired = $false

function Refresh-InstallerPath {
    $env:Path = (@(
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User'),
        $env:Path
    ) | Where-Object { $_ }) -join ';'
}

function Install-RequiredPackage {
    param([string]$PackageId, [string]$DisplayName)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "WinGet is missing. Install Microsoft App Installer from the Microsoft Store, then rerun Install LocalPilot.cmd."
    }
    Write-Host "Installing $DisplayName..."
    & $winget.Source install --id $PackageId --exact --source winget --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -in @(3010, 1641)) {
        $script:restartRequired = $true
        Write-Host "$DisplayName requires a Windows restart. Restart before using LocalPilot's hardware sensors."
    } elseif ($LASTEXITCODE -ne 0) {
        throw "Could not install $DisplayName. WinGet exit code: $LASTEXITCODE."
    }
    Refresh-InstallerPath
}

function Test-WebViewRuntime {
    $clientId = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    foreach ($key in @(
        "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$clientId",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$clientId",
        "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$clientId"
    )) {
        $version = (Get-ItemProperty -LiteralPath $key -Name pv -ErrorAction SilentlyContinue).pv
        if ($version -and $version -ne '0.0.0.0') { return $true }
    }
    return $false
}

function Ensure-ClaudeExecutable {
    param([string]$Executable)
    $command = Get-Command -Name $Executable -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    if (Test-Path -LiteralPath $Executable -PathType Leaf) { return (Resolve-Path -LiteralPath $Executable).Path }
    if ($Executable -notin @('claude', 'claude.exe')) {
        throw "The configured Claude Code executable is missing: $Executable. Restore it or update selfdev.implementation_executable; setup preserved your selection."
    }
    Install-RequiredPackage 'Anthropic.ClaudeCode' 'Claude Code' | Out-Host
    $command = Get-Command -Name $Executable -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) { throw 'Claude Code was installed but could not be discovered. Restart setup to refresh its environment.' }
    return $command.Source
}

function Test-ImplementationReadiness {
    param([string]$Python, [string]$ConfigPath)
    $settingsCode = @'
import json, sys
from localpilot.config import load_config
cfg = load_config(sys.argv[1]).selfdev
print(json.dumps({'enabled': cfg.enabled, 'backend': cfg.implementation_backend, 'executable': cfg.implementation_executable}))
'@
    $settings = & $Python -c $settingsCode $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the configured implementation backend.' }
    $settings = ($settings | Out-String) | ConvertFrom-Json
    if (-not $settings.enabled -or $settings.backend -ne 'claude_code') { return }
    Ensure-ClaudeExecutable -Executable $settings.executable | Out-Null
    Write-Host 'Checking the implementation backend and its local model context...'
    $preflightCode = @'
import sys
from localpilot.config import load_config
from localpilot.implementation_backend import ClaudeCodeBackend
cfg = load_config(sys.argv[1]).selfdev
result = ClaudeCodeBackend(
    executable=cfg.implementation_executable, model=cfg.implementation_model,
    context_tokens=cfg.implementation_context_tokens,
    timeout_seconds=cfg.implementation_timeout_seconds,
    base_url=cfg.implementation_base_url,
).preflight()
print(f'Claude Code: {result.version}; model: {result.model}; context: {result.context_tokens}')
for message in result.messages:
    print(message)
raise SystemExit(0 if result.healthy else 1)
'@
    & $Python -c $preflightCode $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw 'The configured implementation backend is not ready. See the checks above and installation log, then retry setup.'
    }
}

function Assert-ReleasePayload {
    param([string]$Root, [string]$ExpectedRuntime)
    $manifestPath = Join-Path $Root '.localpilot-release.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "This source bundle has no release identity. Download the Windows installer from LocalPilot's GitHub Releases or clone the Git repository."
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ([string]$manifest.source_sha -notmatch '^[0-9a-fA-F]{40}$' -or
        [string]$manifest.version -notmatch '^\d+\.\d+\.\d+(?:[.+-][A-Za-z0-9.-]+)?$') {
        throw 'The release bundle has an invalid source or version identity.'
    }
    if ([string]$manifest.runtime_id -ne $ExpectedRuntime) {
        throw "This release is for '$($manifest.runtime_id)'; this PC requires '$ExpectedRuntime'. Download the matching Windows release."
    }
    if ([string]$manifest.hardware_provider_sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'The release bundle has no valid hardware provider fingerprint.'
    }
    $provider = Join-Path $Root "localpilot\_hardware\$ExpectedRuntime\LocalPilot.SystemSense.HardwareProvider.exe"
    if (-not (Test-Path -LiteralPath $provider -PathType Leaf)) {
        throw 'The bundled hardware provider is missing or corrupt. Download the complete release again; no files were installed.'
    }
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($provider)
    try { $providerHash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') }
    finally { $stream.Dispose(); $hasher.Dispose() }
    if ($providerHash -ne $manifest.hardware_provider_sha256) {
        throw 'The bundled hardware provider is missing or corrupt. Download the complete release again; no files were installed.'
    }
    return $manifest
}

function Assert-InstalledReleaseVersion {
    param([string]$Root, [string]$SourceSha)
    $pendingPath = Join-Path $Root '.git\localpilot-install-pending'
    if (Test-Path -LiteralPath $pendingPath) {
        if ((Get-Content -LiteralPath $pendingPath -Raw).Trim() -ne $SourceSha) {
            throw 'This unfinished install belongs to a different release. Resume its original installer first.'
        }
        return
    }
    $head = & git -C $Root rev-parse --verify HEAD
    if ($LASTEXITCODE -ne 0 -or ($head | Out-String).Trim() -ne $SourceSha) {
        throw "The installed checkout has a different version. Use LocalPilot's desktop update flow first, or choose a new installation directory. Its code, hardware provider, configuration and data were preserved."
    }
}

function Copy-ReleaseSources {
    param([string]$Source, [string]$Destination)
    $sourcePath = [IO.Path]::GetFullPath($Source).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $destinationPath = [IO.Path]::GetFullPath($Destination)
    if ($destinationPath.StartsWith($sourcePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Choose an installation directory outside the extracted installer folder.'
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        # User configuration, data, Python environment and Git history belong
        # to the installed checkout and are never replaced by bundle contents.
        if ($item.Name -in @('.git', '.venv', 'localpilot.toml', 'localpilot-data')) { continue }
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse -Force
    }
}

function Initialize-ReleaseCheckout {
    param(
        [string]$Root,
        [string]$RepositoryUrl = 'https://github.com/n47h4ni3l/localpilot.git'
    )
    $gitDirectory = Join-Path $Root '.git'
    $pendingPath = Join-Path $gitDirectory 'localpilot-install-pending'
    if ((Test-Path -LiteralPath $gitDirectory) -and -not (Test-Path -LiteralPath $pendingPath)) { return }
    $manifestPath = Join-Path $Root '.localpilot-release.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "This source bundle has no release identity. Download the Windows installer from LocalPilot's GitHub Releases or clone the Git repository."
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $sourceSha = [string]$manifest.source_sha
    if ($sourceSha -notmatch '^[0-9a-fA-F]{40}$') {
        throw "The release bundle has an invalid source commit identity."
    }
    if (-not (Test-Path -LiteralPath $gitDirectory)) {
        & git -C $Root init -b main
        if ($LASTEXITCODE -ne 0) { throw "Could not initialize LocalPilot's Git checkout." }
        Set-Content -LiteralPath $pendingPath -Value $sourceSha -NoNewline
    } elseif ((Get-Content -LiteralPath $pendingPath -Raw).Trim() -ne $sourceSha) {
        throw "An incomplete install belongs to a different release. Retry with its original installer to preserve the existing checkout."
    }
    $remoteNames = @(& git -C $Root remote)
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect LocalPilot's Git remotes." }
    if ($remoteNames -notcontains 'origin') {
        & git -C $Root remote add origin $RepositoryUrl
        if ($LASTEXITCODE -ne 0) { throw "Could not configure LocalPilot's GitHub remote." }
    } else {
        $existingRemote = & git -C $Root remote get-url origin
        if ($LASTEXITCODE -ne 0 -or ($existingRemote | Out-String).Trim() -ne $RepositoryUrl) {
            throw "The incomplete install's Git remote differs from the expected repository."
        }
    }
    & git -C $Root fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/main' $sourceSha
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch the release's source commit. Check the network and retry installation." }
    # Seed Git's index/history from the packaged commit without replacing any
    # files. In particular, this never resets owner configuration or data.
    & git -C $Root reset --mixed $sourceSha
    if ($LASTEXITCODE -ne 0) { throw "Could not register the release's source files with Git." }
    & git -C $Root branch --set-upstream-to=origin/main main
    if ($LASTEXITCODE -ne 0) { throw "Could not connect LocalPilot's update branch." }
    Remove-Item -LiteralPath $pendingPath
}

if ($env:OS -ne 'Windows_NT') { throw 'The LocalPilot desktop installer requires Windows.' }
$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($architecture -notin @('X64', 'Arm64')) { throw 'LocalPilot requires 64-bit Windows.' }
$sourceHasGit = Test-Path -LiteralPath (Join-Path $sourceRoot '.git')
$expectedRuntime = 'win-' + $architecture.ToLowerInvariant()
$release = if (-not $sourceHasGit) { Assert-ReleasePayload -Root $sourceRoot -ExpectedRuntime $expectedRuntime } else { $null }
$logDirectory = Join-Path $env:LOCALAPPDATA 'LocalPilot'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $arguments = @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath))
    if ($InstallDirectory) { $arguments += @('-InstallDirectory', ('"{0}"' -f $InstallDirectory)) }
    if ($SkipLaunch) { $arguments += '-SkipLaunch' }
    if ($ShortcutPath) { $arguments += @('-ShortcutPath', ('"{0}"' -f $ShortcutPath)) }
    try {
        $elevated = Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $arguments -Wait -PassThru -WindowStyle Normal
    } catch {
        $message = "Windows did not start administrator setup: $($_.Exception.Message). Run Install LocalPilot.cmd again and approve the administrator prompt."
        Add-Content -LiteralPath (Join-Path $logDirectory 'installation.log') -Value $message
        Write-Host $message -ForegroundColor Yellow
        exit 1
    }
    exit $elevated.ExitCode
}

$transcriptStarted = $false
try {
    Start-Transcript -Path (Join-Path $logDirectory 'installation.log') -Append | Out-Null
    $transcriptStarted = $true
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Install-RequiredPackage 'Git.Git' 'Git' }
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Install-RequiredPackage 'GitHub.cli' 'GitHub CLI' }
    if (-not (Test-WebViewRuntime)) { Install-RequiredPackage 'Microsoft.EdgeWebView2Runtime' 'Microsoft WebView2 Runtime' }
    if (-not (Get-Service -Name PawnIO -ErrorAction SilentlyContinue)) { Install-RequiredPackage 'namazso.PawnIO' 'PawnIO hardware sensor driver' }

    $root = if ($InstallDirectory) {
        [IO.Path]::GetFullPath($InstallDirectory)
    } elseif ($sourceHasGit) {
        $sourceRoot
    } else {
        Join-Path $logDirectory 'app'
    }
    if ($root -ne $sourceRoot) {
        if (-not (Test-Path -LiteralPath (Join-Path $root '.git'))) {
            Copy-ReleaseSources -Source $sourceRoot -Destination $root
        } else {
            if ($release) { Assert-InstalledReleaseVersion -Root $root -SourceSha $release.source_sha }
            Write-Host "Preserving the existing LocalPilot checkout at $root."
        }
    }
    if ($release) { Assert-ReleasePayload -Root $root -ExpectedRuntime $expectedRuntime | Out-Null }
    Initialize-ReleaseCheckout -Root $root
    Push-Location $root
    try {
        & (Join-Path $root 'scripts\bootstrap.ps1') -Unattended -RuntimeOnly
        $python = Join-Path $root '.venv\Scripts\python.exe'
        Test-ImplementationReadiness -Python $python -ConfigPath (Join-Path $root 'localpilot.toml')
        $shortcutOptions = @{ PythonPath = $python; RunAsAdministrator = $true }
        if ($ShortcutPath) { $shortcutOptions.ShortcutPath = $ShortcutPath }
        & (Join-Path $root 'scripts\install-desktop-shortcut.ps1') @shortcutOptions
        & $python -m localpilot.cli --config (Join-Path $root 'localpilot.toml') doctor
        if ($LASTEXITCODE -ne 0) { throw 'LocalPilot readiness checks failed. See the installation log, then retry.' }
        $savedErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & gh auth status 1>$null 2>$null
            $githubAuthenticated = $LASTEXITCODE -eq 0
        } finally { $ErrorActionPreference = $savedErrorPreference }
        if (-not $githubAuthenticated) {
            Write-Host 'Desktop chat is ready. To enable GitHub repair PRs, sign in once with: gh auth login --web'
        }
        if (-not $SkipLaunch -and -not $script:restartRequired) {
            $shortcut = if ($ShortcutPath) { [IO.Path]::GetFullPath($ShortcutPath) } else { Join-Path ([Environment]::GetFolderPath('Desktop')) 'LocalPilot.lnk' }
            Start-Process -FilePath $shortcut -WorkingDirectory $root
        }
        Write-Host "LocalPilot installed at $root. The LocalPilot desktop icon is ready." -ForegroundColor Green
        if ($script:restartRequired) { Write-Host 'Restart Windows, then open LocalPilot using its desktop icon.' }
    } finally { Pop-Location }
} catch {
    Write-Error "LocalPilot installation failed: $($_.Exception.Message). Details: $(Join-Path $logDirectory 'installation.log')" -ErrorAction Continue
    exit 1
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
exit 0
