param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [switch]$RunAsAdministrator
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = (Resolve-Path -LiteralPath $PythonPath).Path
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw "The selected Python installation has no pythonw.exe."
}
Push-Location $repoRoot
try {
    & $python -c "import localpilot.cli, webview; from PIL import Image; from pathlib import Path; p=Path('localpilot-data/desktop'); p.mkdir(parents=True, exist_ok=True); Image.open('localpilot/webview/avatar/state-0.png').save(p/'localpilot.ico', sizes=[(16,16),(32,32),(48,48),(256,256)])"
    if ($LASTEXITCODE -ne 0) { throw "Desktop dependencies or icon creation failed." }
    $shell = New-Object -ComObject WScript.Shell
    $shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'LocalPilot.lnk'
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = '-m localpilot.cli --config "' + (Join-Path $repoRoot 'localpilot.toml') + '" desktop'
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.IconLocation = (Join-Path $repoRoot 'localpilot-data/desktop/localpilot.ico') + ',0'
    $shortcut.Description = 'Open the LocalPilot desktop companion'
    $shortcut.Save()
    if ($RunAsAdministrator) {
        # Shell Link header LinkFlags: SLDF_RUNAS_USER requests UAC elevation.
        $bytes = [IO.File]::ReadAllBytes($shortcutPath)
        if ($bytes.Length -lt 76 -or [BitConverter]::ToUInt32($bytes, 0) -ne 76) {
            throw "Unexpected Windows shortcut header."
        }
        $bytes[0x14] = $bytes[0x14] -bor 0x20
        [IO.File]::WriteAllBytes($shortcutPath, $bytes)
    }
    Write-Host "Desktop shortcut ready: $shortcutPath"
} finally { Pop-Location }
