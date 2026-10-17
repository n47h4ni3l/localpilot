param(
    [string]$InstallDirectory = '',
    [string]$BundleRoot = '',
    [string]$OwnerSid = '',
    [switch]$VerifyOnly,
    [switch]$SkipLaunch,
    [switch]$SkipHardwareInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-NestraPlainPath {
    param([string]$Path)
    $absolute = [IO.Path]::GetFullPath($Path)
    $item = $absolute
    while ($item) {
        if (Test-Path -LiteralPath $item) {
            $entry = Get-Item -LiteralPath $item -Force
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Linked paths and junctions are not supported: $item"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($item)
        if ($parent -eq $item) { break }
        $item = $parent
    }
    return $absolute
}

function Resolve-NestraBundleFile {
    param([string]$Root, [string]$Relative)
    if (-not $Relative -or $Relative -match '[\\:<>"]|[\x00-\x1f]' -or
        $Relative.StartsWith('/') -or $Relative.EndsWith('/')) {
        throw "Invalid bundle inventory path: $Relative"
    }
    foreach ($part in $Relative.Split('/')) {
        if (-not $part -or $part -in @('.', '..') -or $part -match '[. ]$' -or
            $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)' -or
            $part -match '[?*|]') { throw "Invalid bundle inventory path: $Relative" }
    }
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $resolved = Assert-NestraPlainPath -Path (Join-Path $Root $Relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Bundle inventory path escapes its root: $Relative"
    }
    return $resolved
}

function Get-NestraBundleInventory {
    param([string]$Root)
    $rootPath = Assert-NestraPlainPath -Path $Root
    $manifestPath = Join-Path $rootPath 'bundle-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'bundle-manifest.json is missing. Extract the complete Nestra ZIP first.' }
    if ((Get-Item -LiteralPath $manifestPath).Length -gt 32MB) { throw 'The bundle manifest is too large.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $format = if ($manifest.PSObject.Properties['format_version']) { $manifest.format_version } elseif ($manifest.PSObject.Properties['schema_version']) { $manifest.schema_version } else { 0 }
    if ($format -ne 1 -or [string]$manifest.source_sha -notmatch '^[0-9a-f]{40}$' -or
        [string]$manifest.version -notmatch '^\d+\.\d+\.\d+[A-Za-z0-9.+-]*$') { throw 'The bundle release identity is invalid.' }
    $files = @($manifest.files)
    if (-not $files.Count -or $files.Count -gt 200000) { throw 'The bundle file inventory is empty or too large.' }
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    [long]$total = 0
    foreach ($file in $files) {
        $relative = [string]$file.path
        if ($relative -eq 'bundle-manifest.json' -or -not $seen.Add($relative)) { throw "Duplicate or invalid inventory entry: $relative" }
        $size = 0L
        if (-not [long]::TryParse([string]$file.size, [ref]$size) -or $size -lt 0 -or $size -gt 256GB -or
            [string]$file.sha256 -notmatch '^[0-9a-f]{64}$') { throw "Invalid bundle size or SHA256: $relative" }
        $path = Resolve-NestraBundleFile -Root $rootPath -Relative $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Bundle file is missing: $relative" }
        if ((Get-Item -LiteralPath $path).Length -ne $size) { throw "Bundle file size differs: $relative" }
        $hasher = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($path)
        try { $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ($digest -ne [string]$file.sha256) { throw "Bundle file SHA256 differs: $relative" }
        if ($total -gt ([long]::MaxValue - $size)) { throw 'Bundle size overflow.' }
        $total += $size
    }
    $prefixLength = $rootPath.TrimEnd('\', '/').Length + 1
    $directories = New-Object 'System.Collections.Generic.Queue[string]'
    $directories.Enqueue($rootPath)
    while ($directories.Count) {
        foreach ($entry in Get-ChildItem -LiteralPath $directories.Dequeue() -Force) {
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Linked bundle entry is not supported: $($entry.FullName)" }
            if ($entry.PSIsContainer) { $directories.Enqueue($entry.FullName) }
            else {
                $relative = $entry.FullName.Substring($prefixLength).Replace('\', '/')
                if ($relative -ne 'bundle-manifest.json' -and -not $seen.Contains($relative)) { throw "Bundle contains an unlisted file: $relative" }
            }
        }
    }
    foreach ($required in @(
        'source.bundle',
        'payload/dependencies/runtime-deps/python/python/python.exe',
        'payload/dependencies/runtime-deps/python/python/pythonw.exe',
        'payload/dependencies/runtime-deps/python/python/Lib/tkinter/__init__.py',
        'payload/dependencies/runtime-deps/ollama/ollama.exe',
        'payload/dependencies/runtime-deps/gh/bin/gh.exe',
        'payload/dependencies/installers/webview2/MicrosoftEdgeWebView2RuntimeInstallerX64.exe',
        'payload/dependencies/installers/pawnio/PawnIO_setup.exe',
        'payload/dependencies/installers/claude/claude.exe',
        'payload/dependencies/requirements-offline.txt',
        'payload/model/bundle-model.json', 'payload/model/manifest.json',
        'payload/hardware/win-x64/LocalPilot.SystemSense.HardwareProvider.exe',
        'payload/hardware/win-x64/THIRD_PARTY_NOTICES.md'
    )) {
        if (-not $seen.Contains($required)) { throw "Required bundle payload is missing: $required" }
    }
    $gitFiles = @($files | Where-Object { $_.path -match '^payload/dependencies/installers/portable-git/PortableGit-[^/]+\.7z\.exe$' })
    if ($gitFiles.Count -ne 1) { throw 'Exactly one pinned PortableGit installer is required.' }
    if (-not @($files | Where-Object { $_.path -match '^payload/dependencies/wheelhouse/[^/]+\.whl$' }).Count) { throw 'The offline wheelhouse is missing.' }
    return [pscustomobject]@{ Manifest = $manifest; TotalBytes = $total; PortableGit = [string]$gitFiles[0].path; Root = $rootPath }
}

function Get-NestraInstallHostState {
    param([string]$Destination)
    if ($env:OS -ne 'Windows_NT') { throw 'Nestra ALPHA2 requires Windows x64.' }
    $operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem
    if (-not ('NestraBundle.NativeFeatures' -as [type])) {
        Add-Type -TypeDefinition 'namespace NestraBundle { public static class NativeFeatures { [System.Runtime.InteropServices.DllImport("kernel32.dll")] [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)] public static extern bool IsProcessorFeaturePresent(uint feature); } }'
    }
    $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($Destination))
    return [pscustomobject]@{
        Architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        Is64BitProcess = [Environment]::Is64BitProcess
        WindowsMajor = ([version]$operatingSystem.Version).Major
        WindowsBuild = [int]$operatingSystem.BuildNumber
        Avx2 = [NestraBundle.NativeFeatures]::IsProcessorFeaturePresent(40)
        RamBytes = [long]$computer.TotalPhysicalMemory
        FreeBytes = [long]$drive.AvailableFreeSpace
        FixedDrive = $drive.DriveType -eq [IO.DriveType]::Fixed
    }
}

function Assert-NestraInstallPreflight {
    param($Inventory, [string]$Destination)
    $destinationPath = Assert-NestraPlainPath -Path $Destination
    if (Test-Path -LiteralPath $destinationPath) {
        throw "An installation or partial installation already exists at '$destinationPath'. It was preserved. Choose a new -InstallDirectory, or keep using the existing installation; upgrades require a separate reviewed migration."
    }
    $bundlePrefix = $Inventory.Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if ($destinationPath.StartsWith($bundlePrefix, [StringComparison]::OrdinalIgnoreCase) -or $destinationPath -eq $Inventory.Root) { throw 'Install outside the extracted bundle directory.' }
    $state = Get-NestraInstallHostState -Destination $destinationPath
    if ($state.Architecture -ne 'X64' -or -not $state.Is64BitProcess -or $state.WindowsMajor -lt 10 -or $state.WindowsBuild -lt 19041) { throw 'Nestra ALPHA2 requires 64-bit Windows 10 build 19041 or newer on an x64 processor.' }
    if (-not $state.Avx2) { throw 'This Nestra bundle requires an AVX2-capable x64 CPU. No alternate model was selected.' }
    if ($state.RamBytes -lt 24GB) { throw 'This Nestra model requires at least 24 GiB of physical RAM. Paging and security settings were not changed.' }
    [long]$required = $Inventory.TotalBytes * 2 + 5GB
    if (-not $state.FixedDrive -or $state.FreeBytes -lt $required) { throw "Install on a local fixed drive with at least $([math]::Ceiling($required / 1GB)) GiB free. Required: bundle inventory bytes times two plus 5 GiB." }
    return $destinationPath
}

function Assert-NestraSetupOwner {
    param([string]$ExpectedSid)
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($ExpectedSid -notmatch '^S-1-5-21-(\d+-){3}\d+$' -or $identity.User.Value -ne $ExpectedSid) { throw 'Setup must run as the same Windows account that opened Install Nestra.cmd. Different-account administrator credentials are not supported.' }
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run Install Nestra.cmd and approve its administrator prompt using this same Windows account.' }
    return $identity.User.Value
}

function ConvertTo-NestraArgument {
    param([string]$Value)
    # Windows CommandLineToArgvW quoting, including terminal backslashes.
    if ($Value -and $Value -notmatch '[\s"]') { return $Value }
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Invoke-NestraSetupProcess {
    param([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory, [string]$Stage, [int[]]$SuccessCodes = @(0))
    $script:receipt.stage = $Stage
    Save-NestraSetupReceipt
    Write-Host (($Stage.Replace('-', ' ')) + '...')
    $stdout = Join-Path $script:logDirectory ($Stage + '.stdout.log')
    $stderr = Join-Path $script:logDirectory ($Stage + '.stderr.log')
    $quoted = @($Arguments | ForEach-Object { ConvertTo-NestraArgument -Value $_ })
    $process = Start-Process -FilePath $Executable -ArgumentList ($quoted -join ' ') -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $process.Handle
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode -or $exitCode -notin $SuccessCodes) { throw "Stage '$Stage' failed with exit code '$exitCode'. Its output is preserved at $script:logDirectory" }
    if ($exitCode -in @(3010, 1641)) { $script:receipt.restart_required = $true }
    return $exitCode
}

function Save-NestraSetupReceipt {
    $script:receipt.updated_at = [DateTime]::UtcNow.ToString('o')
    $temporary = $script:receiptPath + '.tmp'
    $script:receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $script:receiptPath -Force
}

function Test-NestraWebViewRuntime {
    $client = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    foreach ($key in @("HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$client", "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$client", "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$client")) {
        $entry = Get-ItemProperty -LiteralPath $key -Name pv -ErrorAction SilentlyContinue
        $version = if ($entry) { $entry.pv } else { $null }
        if ($version -and $version -ne '0.0.0.0') { return $true }
    }
    return $false
}

function Get-NestraBundleTaskNames {
    param([string]$Root, [string]$Sid)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes(([IO.Path]::GetFullPath($Root).ToLowerInvariant() + '|' + $Sid)))).Replace('-', '').ToLowerInvariant().Substring(0, 12) }
    finally { $hasher.Dispose() }
    return [ordered]@{ Broker = "Nestra ALPHA2 $digest Broker"; Worker = "Nestra ALPHA2 $digest Worker"; Desktop = "Nestra ALPHA2 $digest Desktop" }
}

function Assert-NestraLauncherAvailability {
    param([string]$Root, [string]$Sid, [string]$DesktopDirectory = '')
    if (-not $DesktopDirectory) { $DesktopDirectory = [Environment]::GetFolderPath('Desktop') }
    $shortcutPath = Join-Path $DesktopDirectory 'Nestra ALPHA2.lnk'
    if (Test-Path -LiteralPath $shortcutPath) { throw "An existing desktop shortcut was preserved: $shortcutPath. Review the existing installation before installing another copy." }
    $names = Get-NestraBundleTaskNames -Root $Root -Sid $Sid
    $existing = @(Get-ScheduledTask -ErrorAction Stop)
    foreach ($role in @('Broker', 'Worker', 'Desktop')) {
        if (@($existing | Where-Object { $_.TaskName -eq $names[$role] -and $_.TaskPath -eq '\' }).Count) {
            throw "A task named '$($names[$role])' already exists and was preserved. Use the existing installation or review migration first."
        }
    }
}

function Register-NestraBundleTasks {
    param([string]$Root, [string]$Sid)
    $launcher = Join-Path $Root 'app/scripts/launch-nestra.ps1'
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw 'The bundled source is missing its runtime launcher.' }
    # Recheck immediately before registration as well as during preflight.
    Assert-NestraLauncherAvailability -Root $Root -Sid $Sid
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
    $names = Get-NestraBundleTaskNames -Root $Root -Sid $Sid
    $script:receipt.planned_tasks = $names
    $script:receipt.registered_tasks = @()
    Save-NestraSetupReceipt
    foreach ($role in @('Broker', 'Worker', 'Desktop')) {
        $level = if ($role -eq 'Desktop') { 'Limited' } else { 'Highest' }
        $arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File ' + (ConvertTo-NestraArgument $launcher) + ' -Role ' + $role + ' -InstallDirectory ' + (ConvertTo-NestraArgument $Root)
        $action = New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe') -Argument $arguments -WorkingDirectory (Join-Path $Root 'app')
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $Sid
        $principal = New-ScheduledTaskPrincipal -UserId $Sid -LogonType Interactive -RunLevel $level
        Register-ScheduledTask -TaskName $names[$role] -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Nestra ALPHA2; root=$Root; owner=$Sid; role=$role" | Out-Null
        $script:receipt.registered_tasks += $names[$role]
        Save-NestraSetupReceipt
    }
    [ordered]@{ format_version = 1; owner_sid = $Sid; install_directory = $Root; tasks = $names } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Root 'tasks.json') -Encoding utf8
    $desktopDirectory = [Environment]::GetFolderPath('Desktop')
    $shortcutPath = Join-Path $desktopDirectory 'Nestra ALPHA2.lnk'
    if (Test-Path -LiteralPath $shortcutPath) { throw "An existing desktop shortcut was preserved: $shortcutPath" }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    # This task's Limited principal guarantees the GUI is standard privilege.
    $shortcut.TargetPath = Join-Path $env:SystemRoot 'System32/schtasks.exe'
    $shortcut.Arguments = '/Run /TN ' + (ConvertTo-NestraArgument $names.Desktop)
    $shortcut.WorkingDirectory = Join-Path $Root 'app'
    $shortcut.WindowStyle = 7
    $shortcut.Description = 'Open Nestra ALPHA2'
    $shortcut.Save()
    return [pscustomobject]@{ Shortcut = $shortcutPath; Names = $names }
}

function Install-NestraBundle {
    param($Inventory, [string]$Destination, [string]$Sid, [bool]$NoHardware, [bool]$NoLaunch)
    $script:receipt.stage = 'copy-runtime'; Save-NestraSetupReceipt
    Write-Host 'Copying private runtime files...'
    New-Item -ItemType Directory -Path $Destination | Out-Null
    $runtime = Join-Path $Destination 'runtime'
    New-Item -ItemType Directory -Path $runtime | Out-Null
    $dependencies = Join-Path $Inventory.Root 'payload/dependencies'
    foreach ($name in @('python', 'ollama', 'gh')) {
        Copy-Item -LiteralPath (Join-Path $dependencies "runtime-deps/$name") -Destination (Join-Path $runtime $name) -Recurse
    }
    $gitRoot = Join-Path $runtime 'git'
    Invoke-NestraSetupProcess -Executable (Join-Path $Inventory.Root $Inventory.PortableGit) -Arguments @('-y', "-o$gitRoot") -WorkingDirectory $Destination -Stage 'extract-git' | Out-Null
    $git = Join-Path $gitRoot 'cmd/git.exe'
    if (-not (Test-Path -LiteralPath $git -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $gitRoot 'bin/bash.exe') -PathType Leaf)) { throw 'PortableGit did not provide git.exe and Bash. The installation is incomplete and was preserved.' }
    $env:PATH = @((Join-Path $gitRoot 'cmd'), (Join-Path $gitRoot 'bin'), (Join-Path $runtime 'gh/bin'), $env:PATH) -join ';'
    $app = Join-Path $Destination 'app'
    Invoke-NestraSetupProcess -Executable $git -Arguments @('clone', '--branch', 'main', '--no-hardlinks', (Join-Path $Inventory.Root 'source.bundle'), $app) -WorkingDirectory $Destination -Stage 'clone-source' | Out-Null
    Invoke-NestraSetupProcess -Executable $git -Arguments @('-C', $app, 'rev-parse', 'HEAD') -WorkingDirectory $Destination -Stage 'verify-source' | Out-Null
    $installedSha = (Get-Content -LiteralPath (Join-Path $script:logDirectory 'verify-source.stdout.log') -Raw).Trim()
    if ($installedSha -cne [string]$Inventory.Manifest.source_sha) { throw 'The source bundle main branch differs from its release identity. No model was started.' }
    Invoke-NestraSetupProcess -Executable $git -Arguments @('-C', $app, 'remote', 'set-url', 'origin', 'https://github.com/n47h4ni3l/localpilot.git') -WorkingDirectory $Destination -Stage 'configure-origin' | Out-Null
    $hardwareDestination = Join-Path $app 'localpilot/_hardware/win-x64'
    New-Item -ItemType Directory -Path $hardwareDestination -Force | Out-Null
    Copy-Item -Path (Join-Path $Inventory.Root 'payload/hardware/win-x64/*') -Destination $hardwareDestination -Recurse
    $claudeDirectory = Join-Path $runtime 'claude'
    New-Item -ItemType Directory -Path $claudeDirectory | Out-Null
    Copy-Item -LiteralPath (Join-Path $dependencies 'installers/claude/claude.exe') -Destination $claudeDirectory
    $basePython = Join-Path $runtime 'python/python/python.exe'
    Invoke-NestraSetupProcess -Executable $basePython -Arguments @('-m', 'venv', (Join-Path $app '.venv')) -WorkingDirectory $app -Stage 'create-environment' | Out-Null
    $python = Join-Path $app '.venv/Scripts/python.exe'
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-m', 'pip', '--isolated', 'install', '--no-index', '--only-binary=:all:', '--find-links', (Join-Path $dependencies 'wheelhouse'), '--require-hashes', '-r', (Join-Path $dependencies 'requirements-offline.txt')) -WorkingDirectory $app -Stage 'install-offline-dependencies' | Out-Null
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps', '--no-build-isolation', '-e', $app) -WorkingDirectory $app -Stage 'install-application' | Out-Null
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-m', 'pip', '--isolated', 'check') -WorkingDirectory $app -Stage 'verify-dependencies' | Out-Null
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-c', 'import tkinter; from PIL import ImageTk; import webview, win32api, pytest; import localpilot.cli') -WorkingDirectory $app -Stage 'verify-imports' | Out-Null
    $models = Join-Path $Destination 'models'
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-m', 'localpilot.nestra_bundle_model', 'install', '--bundle', (Join-Path $Inventory.Root 'payload/model'), '--destination', $models) -WorkingDirectory $app -Stage 'install-model' | Out-Null
    Invoke-NestraSetupProcess -Executable $python -Arguments @('-m', 'localpilot.nestra_bundle_runtime', 'configure', '--root', $app, '--runtime-root', $runtime, '--models-root', $models, '--port', '11435') -WorkingDirectory $app -Stage 'configure-runtime' | Out-Null
    if (-not (Test-NestraWebViewRuntime)) {
        Invoke-NestraSetupProcess -Executable (Join-Path $dependencies 'installers/webview2/MicrosoftEdgeWebView2RuntimeInstallerX64.exe') -Arguments @('/silent', '/install') -WorkingDirectory $Destination -Stage 'install-webview2' -SuccessCodes @(0, 3010, 1641) | Out-Null
        if (-not (Test-NestraWebViewRuntime)) { throw 'WebView2 installation did not verify. Restart Windows if requested, then inspect the preserved receipt.' }
    }
    if (-not $NoHardware -and -not (Get-Service -Name PawnIO -ErrorAction SilentlyContinue)) {
        # PawnIO 2.2.0's published WinGet manifest specifies -install -silent.
        Invoke-NestraSetupProcess -Executable (Join-Path $dependencies 'installers/pawnio/PawnIO_setup.exe') -Arguments @('-install', '-silent') -WorkingDirectory $Destination -Stage 'install-pawnio' -SuccessCodes @(0, 3010, 1641) | Out-Null
        if (-not (Get-Service -Name PawnIO -ErrorAction SilentlyContinue)) { throw 'PawnIO installation did not register its service. No driver or security fallback was applied.' }
    }
    $script:receipt.stage = 'register-launchers'; Save-NestraSetupReceipt
    Write-Host 'Registering this account''s Nestra launchers...'
    $launchers = Register-NestraBundleTasks -Root $Destination -Sid $Sid
    $script:receipt.hardware_install_skipped = $NoHardware
    $script:receipt.stage = 'installed'; $script:receipt.status = 'installed'; Save-NestraSetupReceipt
    if (-not $NoLaunch -and -not $script:receipt.restart_required) {
        $script:receipt.stage = 'launching'; Save-NestraSetupReceipt
        Write-Host 'Starting Nestra...'
        Start-ScheduledTask -TaskName $launchers.Names.Broker
        Start-ScheduledTask -TaskName $launchers.Names.Worker
        # Explorer brokers the initial shortcut; the task itself is Limited.
        $explorer = New-Object -ComObject Shell.Application
        $explorer.ShellExecute($launchers.Shortcut, '', [IO.Path]::GetDirectoryName($launchers.Shortcut), 'open', 1)
    }
    $script:receipt.stage = 'complete'; Save-NestraSetupReceipt
}

# No bundled executable, installer, task, receipt or destination is touched by
# VerifyOnly. Its work is inventory hashing and local, read-only host checks.
if (-not $BundleRoot) { $BundleRoot = Split-Path -Parent $PSScriptRoot }
if (-not $InstallDirectory) { $InstallDirectory = Join-Path $env:LOCALAPPDATA 'Programs/Nestra' }
$script:receipt = $null
$script:receiptPath = $null
try {
    $verifiedSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if (-not $VerifyOnly) {
        $verifiedSid = Assert-NestraSetupOwner -ExpectedSid $OwnerSid
        $script:logDirectory = Join-Path $env:LOCALAPPDATA ('Nestra/InstallerLogs/' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $script:logDirectory -Force | Out-Null
        $script:receiptPath = Join-Path $script:logDirectory 'receipt.json'
        $script:receipt = [ordered]@{ format_version = 1; status = 'running'; stage = 'verify-bundle'; install_directory = [IO.Path]::GetFullPath($InstallDirectory); owner_sid = $verifiedSid; started_at = [DateTime]::UtcNow.ToString('o'); updated_at = ''; restart_required = $false; error = $null }
        Save-NestraSetupReceipt
    }
    Write-Host 'Verifying the complete bundle and this PC...'
    $inventory = Get-NestraBundleInventory -Root $BundleRoot
    $destination = Assert-NestraInstallPreflight -Inventory $inventory -Destination $InstallDirectory
    Assert-NestraLauncherAvailability -Root $destination -Sid $verifiedSid
    if ($VerifyOnly) {
        [pscustomobject]@{ status = 'verified'; source_sha = $inventory.Manifest.source_sha; version = $inventory.Manifest.version; inventory_bytes = $inventory.TotalBytes; install_directory = $destination } | ConvertTo-Json -Compress
        exit 0
    }
    $script:receipt.source_sha = [string]$inventory.Manifest.source_sha
    $script:receipt.version = [string]$inventory.Manifest.version
    Save-NestraSetupReceipt
    Install-NestraBundle -Inventory $inventory -Destination $destination -Sid $verifiedSid -NoHardware ([bool]$SkipHardwareInstall) -NoLaunch ([bool]$SkipLaunch)
    Write-Host "Nestra ALPHA2 installation files are ready at $destination"
    Write-Host "Installation receipt: $script:receiptPath"
    if ($script:receipt.restart_required) { Write-Host 'Restart Windows before opening Nestra to complete prerequisite installation.' }
} catch {
    if ($null -ne $script:receipt) {
        $script:receipt.status = 'failed'; $script:receipt.error = $_.Exception.Message
        Save-NestraSetupReceipt
        Write-Host "Installation stopped. Existing and partial files were preserved. Receipt: $script:receiptPath"
    }
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
