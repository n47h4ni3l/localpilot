from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest


def _script() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "update-and-restart.ps1").read_text(encoding="utf-8")


def test_update_script_cleans_only_known_systemsense_build_artifacts_before_git_guard() -> None:
    script = _script()

    bin_marker = 'tools\\SystemSense.HardwareProvider\\bin'
    obj_marker = 'tools\\SystemSense.HardwareProvider\\obj'
    # The Git status command now lives inside a helper defined near the top of
    # the script, so source-order against that command would be meaningless.
    # Verify the actual execution order instead: known generated artifacts are
    # removed before the first call to the cleanliness guard.
    guard_call_marker = "    Assert-CleanWorkingTree"

    assert bin_marker in script
    assert obj_marker in script
    assert 'Remove-Item -LiteralPath $resolvedArtifact -Recurse -Force' in script
    assert 'Refusing to remove generated build artifact outside the checkout' in script
    assert script.index(bin_marker) < script.index(guard_call_marker)
    assert script.index(obj_marker) < script.index(guard_call_marker)
    assert 'git reset --hard' not in script
    assert 'git clean -fd' not in script


def test_update_script_prefetches_before_shutdown_then_uses_pinned_target() -> None:
    script = _script()

    fetch_marker = '& git fetch --no-tags --prune $Remote'
    stop_marker = 'from localpilot.desktop_updater import _stop_localpilot_processes'
    merge_marker = '& git merge --ff-only --no-edit $targetSha'

    assert fetch_marker in script
    assert stop_marker in script
    assert merge_marker in script
    assert script.index(fetch_marker) < script.index(stop_marker)
    assert script.index(stop_marker) < script.index(merge_marker)
    assert 'git pull --ff-only' not in script
    assert "LocalPilot was left running unchanged" in script
    assert "Update preflight complete" in script
    assert "[switch]$SkipFetch" in script
    assert "[string]$ExpectedOldSha" in script
    assert "[string]$ExpectedTargetSha" in script
    assert script.index("\nAssert-UpdateAdministrator\n") < script.index(stop_marker)


def test_update_script_stops_rebuilds_restores_and_restarts_localpilot() -> None:
    script = _script()

    expected_steps = [
        'Disable-ScheduledTask -TaskName $TaskName',
        'from localpilot.desktop_updater import _stop_localpilot_processes',
        '& $python -m pip install -e .',
        '& $python -m pip check',
        'scripts\\build-systemsense-hardware.ps1',
        'LocalPilot.SystemSense.HardwareProvider.exe',
        'Enable-ScheduledTask -TaskName $TaskName',
        'Start-ScheduledTask -TaskName $TaskName',
        'Start-Process -FilePath $guiPython',
    ]
    for marker in expected_steps:
        assert marker in script

    assert 'finally {' in script
    assert '$taskWasEnabled' in script
    assert '$taskWasRunning' in script
    assert '$processesStopped' in script
    assert '$updateSucceeded' in script
    assert 'Attempting to relaunch the current checkout' in script
    assert 'live temperature sensors' in script


def test_systemsense_dotnet_outputs_are_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    assert "tools/SystemSense.HardwareProvider/bin/" in gitignore
    assert "tools/SystemSense.HardwareProvider/obj/" in gitignore


def _quote_ps(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_update_functions(body: str) -> dict:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if os.name != "nt" or not powershell:
        pytest.skip("Windows PowerShell is needed for updater integration checks")
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "update-and-restart.ps1"
    command = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({_quote_ps(script)}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ throw ($errors | Out-String) }}
$functions = $ast.FindAll({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }}, $false)
foreach ($function in $functions) {{ . ([scriptblock]::Create($function.Extent.Text)) }}
{body}
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    return json.loads(result.stdout)


def test_update_interpreter_keeps_installed_runtime_and_pythonw_siblings(tmp_path):
    root = tmp_path / "repo without venv"
    root.mkdir()
    installed = tmp_path / "installed Python"
    installed.mkdir()
    python = installed / "python.exe"
    pythonw = installed / "pythonw.exe"
    python.touch()
    pythonw.touch()

    result = _run_update_functions(f"""
$selected = Resolve-UpdatePython -Executable {_quote_ps(pythonw)} -Root {_quote_ps(root)}
$gui = Get-UpdateGuiPython -Executable $selected
$missingRejected = $false
try {{ Resolve-UpdatePython -Executable '' -Root {_quote_ps(root)} | Out-Null }} catch {{ $missingRejected = $true }}
@{{ python = $selected; gui = $gui; missingRejected = $missingRejected }} | ConvertTo-Json -Compress
""")

    assert Path(result["python"]) == python.resolve()
    assert Path(result["gui"]) == pythonw.resolve()
    assert result["missingRejected"] is True
    assert not (root / ".venv").exists()


def test_update_relaunches_canonical_desktop_with_config_and_selected_interpreter(tmp_path):
    root = tmp_path / "production repo"
    config = root / "my config.toml"
    pythonw = tmp_path / "installed Python" / "pythonw.exe"
    result = _run_update_functions(f"""
$repoRoot = {_quote_ps(root)}
$resolvedConfig = {_quote_ps(config)}
$guiPython = {_quote_ps(pythonw)}
function Start-Process {{
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle)
    @{{ executable = $FilePath; arguments = $ArgumentList; root = $WorkingDirectory; windowStyle = $WindowStyle }} | ConvertTo-Json -Compress
}}
Start-LocalPilotDesktop
""")

    assert Path(result["executable"]) == pythonw
    assert result["arguments"] == ["-m", "localpilot.cli", "--config", f'"{config}"', "desktop"]
    assert Path(result["root"]) == root
    assert result["windowStyle"] == "Hidden"


def test_update_preflight_checks_current_windows_elevation():
    if os.name != "nt":
        pytest.skip("Windows elevation token check")
    import ctypes

    administrator = bool(ctypes.windll.shell32.IsUserAnAdmin())
    result = _run_update_functions("""
try { Assert-UpdateAdministrator; @{ allowed = $true; message = '' } | ConvertTo-Json -Compress }
catch { @{ allowed = $false; message = $_.Exception.Message } | ConvertTo-Json -Compress }
""")
    assert result["allowed"] is administrator
    if not administrator:
        assert "administrator PowerShell session" in result["message"]


@pytest.mark.parametrize('scenario', ['existing', 'missing', 'install_failure', 'restart'])
def test_update_obtains_sdk_before_shutdown_and_keeps_existing_desktop_on_failure(tmp_path, scenario):
    sdk = tmp_path / 'dotnet.ps1'
    sdk.write_text(
        "if ($global:LOCALPILOT_TEST_SDK_READY) { '8.0.409 [test]' } else { '7.0.410 [test]' }\n"
        '$global:LASTEXITCODE=0\n', encoding='utf-8',
    )
    winget = tmp_path / 'winget.ps1'
    code = 123 if scenario == 'install_failure' else 3010 if scenario == 'restart' else 0
    winget.write_text(
        "'Downloading .NET SDK... (test progress)'\n"
        "$global:LOCALPILOT_TEST_SDK_INSTALLED = $true\n"
        "$global:LOCALPILOT_TEST_SDK_READY = $true\n"
        f'$global:LASTEXITCODE={code}\n', encoding='utf-8',
    )
    ready = '$true' if scenario == 'existing' else '$false'
    result = _run_update_functions(f"""
$global:LOCALPILOT_TEST_SDK_READY = {ready}
$global:LOCALPILOT_TEST_SDK_INSTALLED = $false
function Out-Host {{ param([Parameter(ValueFromPipeline)]$InputObject) process {{ }} }}
function Get-Command {{
    param($Name, $ErrorAction)
    if ($Name -eq 'dotnet') {{ return [pscustomobject]@{{ Source={_quote_ps(sdk)} }} }}
    if ($Name -eq 'winget') {{ return [pscustomobject]@{{ Source={_quote_ps(winget)} }} }}
}}
$selected = $null
$errorText = ''
try {{ $selected = Ensure-UpdateDotNetSdk 6>$null }} catch {{ $errorText = $_.Exception.Message }}
@{{ selected=$selected; installed=$global:LOCALPILOT_TEST_SDK_INSTALLED; error=$errorText }} | ConvertTo-Json -Compress
""")
    assert result['installed'] is (scenario != 'existing')
    if scenario in ('existing', 'missing'):
        assert Path(result['selected']) == sdk
        assert not result['error']
    else:
        assert result['selected'] is None
        assert 'LocalPilot' in result['error']
    script = _script()
    assert script.index('$dotnetForUpdate = Ensure-UpdateDotNetSdk') < script.index(
        'from localpilot.desktop_updater import _stop_localpilot_processes'
    )
    assert 'Stop-ScheduledTask -TaskName $TaskName' not in script


@pytest.mark.parametrize('arguments, expected', [([], None), ([''], None), (['custom config.toml'], 'custom config.toml')])
def test_update_shutdown_handles_windows_powershell_dropped_empty_argument(monkeypatch, arguments, expected):
    snippet = re.search(r'& \$python -c "([^"]+_stop_localpilot_processes\(Path\.cwd\(\)[^"]+)" \$resolvedConfig', _script()).group(1)
    calls = []
    fake = types.ModuleType('localpilot.desktop_updater')
    fake._stop_localpilot_processes = lambda root, **options: calls.append((root, options))
    monkeypatch.setitem(sys.modules, 'localpilot.desktop_updater', fake)
    monkeypatch.setattr(sys, 'argv', ['-c', *arguments])
    exec(snippet, {})
    assert calls == [(Path.cwd(), {'config_path': expected})]
