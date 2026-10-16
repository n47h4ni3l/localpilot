param([string]$InstallDirectory = '')
$ErrorActionPreference = 'Stop'
function Quote-NestraSetupArgument {
    param([string]$Value)
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
try {
    $bundle = Split-Path -Parent $PSScriptRoot
    $installer = Join-Path $PSScriptRoot 'install-nestra-bundle.ps1'
    if (-not (Test-Path -LiteralPath (Join-Path $bundle 'bundle-manifest.json')) -or -not (Test-Path -LiteralPath $installer)) {
        throw 'Extract the entire Nestra ZIP before opening Install Nestra.cmd.'
    }
    $ownerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $arguments = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File ' + (Quote-NestraSetupArgument $installer) + ' -BundleRoot ' + (Quote-NestraSetupArgument $bundle) + ' -OwnerSid ' + (Quote-NestraSetupArgument $ownerSid)
    if ($InstallDirectory) { $arguments += ' -InstallDirectory ' + (Quote-NestraSetupArgument $InstallDirectory) }
    Write-Host 'Nestra setup will verify the bundle, install its private runtimes and model, and register its launch tasks.'
    Write-Host 'Approve the Windows prompt using this same Windows account. Existing installations are preserved.'
    $process = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe') -Verb RunAs -ArgumentList $arguments -Wait -PassThru
    exit $process.ExitCode
} catch {
    Write-Host $_.Exception.Message
    exit 1
}
