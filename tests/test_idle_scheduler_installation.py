from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _quote_ps(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_functions(body: str) -> dict:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if os.name != "nt" or not powershell:
        pytest.skip("Windows PowerShell is needed for scheduler installation checks")
    script = Path(__file__).resolve().parents[1] / "scripts" / "install-idle-evolve-task.ps1"
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


def test_worker_installer_selects_existing_runtime_without_creating_venv(tmp_path):
    root = tmp_path / "production checkout"
    root.mkdir()
    installed = tmp_path / "installed Python"
    installed.mkdir()
    python = installed / "python.exe"
    pythonw = installed / "pythonw.exe"
    python.touch()
    pythonw.touch()
    result = _run_functions(f"""
$selected = Resolve-WorkerPython -Executable {_quote_ps(pythonw)} -Root {_quote_ps(root)}
$missingRejected = $false
try {{ Resolve-WorkerPython -Executable '' -Root {_quote_ps(root)} | Out-Null }} catch {{ $missingRejected = $true }}
@{{ console = $selected.Console; gui = $selected.Gui; missingRejected = $missingRejected }} | ConvertTo-Json -Compress
""")
    assert Path(result["console"]) == python.resolve()
    assert Path(result["gui"]) == pythonw.resolve()
    assert result["missingRejected"] is True
    assert not (root / ".venv").exists()


def test_worker_installer_preserves_existing_elevation_unless_explicitly_changed():
    result = _run_functions("""
@{
    preservedHighest = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel Highest
    preservedLimited = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel Limited
    newLimited = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel ''
    explicitHighest = Resolve-WorkerRunLevel -Specified $true -Administrator $true -ExistingRunLevel Limited
    explicitLimited = Resolve-WorkerRunLevel -Specified $true -Administrator $false -ExistingRunLevel Highest
} | ConvertTo-Json -Compress
""")
    assert result == {
        "preservedHighest": "Highest",
        "preservedLimited": "Limited",
        "newLimited": "Limited",
        "explicitHighest": "Highest",
        "explicitLimited": "Limited",
    }


@pytest.mark.parametrize("enabled,running", [(True, True), (True, False), (False, False)])
def test_failed_worker_replacement_restores_original_definition_and_state(enabled, running):
    result = _run_functions(f"""
$script:events = [System.Collections.Generic.List[string]]::new()
function Register-ScheduledTask {{ param($TaskName, $Xml, [switch]$Force, $ErrorAction); $script:events.Add("register:$TaskName`:$Xml") }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction); $script:events.Add("disable:$TaskName") }}
function Start-ScheduledTask {{ param($TaskName, $ErrorAction); $script:events.Add("start:$TaskName") }}
function Unregister-ScheduledTask {{ param($TaskName, $Confirm, $ErrorAction); throw 'An existing task must be restored rather than deleted' }}
Restore-PreviousWorkerTask -Name 'LocalPilot Background Worker' -Xml '<Task>old-definition</Task>' -Enabled ${str(enabled).lower()} -Running ${str(running).lower()}
@{{ events = @($script:events) }} | ConvertTo-Json -Compress
""")
    expected = ["register:LocalPilot Background Worker:<Task>old-definition</Task>"]
    if not enabled:
        expected.append("disable:LocalPilot Background Worker")
    elif running:
        expected.append("start:LocalPilot Background Worker")
    assert result["events"] == expected
