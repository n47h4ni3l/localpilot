from __future__ import annotations

from pathlib import Path


def _script() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "update-and-restart.ps1").read_text(encoding="utf-8")


def test_update_script_cleans_only_known_systemsense_build_artifacts_before_git_guard() -> None:
    script = _script()

    bin_marker = 'tools\\SystemSense.HardwareProvider\\bin'
    obj_marker = 'tools\\SystemSense.HardwareProvider\\obj'
    status_marker = 'git status --porcelain --untracked-files=all'

    assert bin_marker in script
    assert obj_marker in script
    assert 'Remove-Item -LiteralPath $artifact -Recurse -Force' in script
    assert script.index(bin_marker) < script.index(status_marker)
    assert script.index(obj_marker) < script.index(status_marker)
    assert 'git reset --hard' not in script
    assert 'git clean -fd' not in script


def test_update_script_stops_updates_rebuilds_and_restarts_localpilot() -> None:
    script = _script()

    expected_steps = [
        'Disable-ScheduledTask -TaskName $TaskName',
        'from localpilot.desktop_updater import _stop_localpilot_processes',
        'git pull --ff-only $Remote $Branch',
        'scripts\\bootstrap.ps1',
        'scripts\\build-systemsense-hardware.ps1',
        'LocalPilot.SystemSense.HardwareProvider.exe',
        'Enable-ScheduledTask -TaskName $TaskName',
        'Start-ScheduledTask -TaskName $TaskName',
        'Start-Process -FilePath $localpilot',
    ]
    for marker in expected_steps:
        assert marker in script

    assert 'finally {' in script
    assert '$taskWasEnabled' in script
    assert '$taskWasRunning' in script
    assert '$updateSucceeded' in script
    assert 'live temperature sensors' in script


def test_systemsense_dotnet_outputs_are_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    assert "tools/SystemSense.HardwareProvider/bin/" in gitignore
    assert "tools/SystemSense.HardwareProvider/obj/" in gitignore
