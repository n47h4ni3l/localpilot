from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from localpilot import desktop_startup as startup


def _link(root: Path, *, config: Path | None = None) -> dict[str, str]:
    arguments = ["-m", "localpilot.native_avatar_companion", "--root", str(root.resolve())]
    if config is not None:
        arguments.extend(["--config", str(config.resolve())])
    return {"root": str(root.resolve()), "target": str(root / ".venv/Scripts/pythonw.exe"),
            "arguments": subprocess.list2cmdline(arguments)}


def _configs(tmp_path: Path, *, same: bool = True) -> tuple[Path, Path]:
    root, old = tmp_path / "current", tmp_path / "previous"
    root.mkdir()
    old.mkdir()
    (old / "localpilot.toml").write_text('[agent]\ndata_dir="localpilot-data"\n', encoding="utf-8")
    data = old / "localpilot-data" if same else root / "localpilot-data"
    (root / "localpilot.toml").write_text('[agent]\ndata_dir=' + json.dumps(str(data)) + '\n', encoding="utf-8")
    return root, old


def test_operation_uses_canonical_cli_selected_interpreter_and_environment_only(tmp_path, monkeypatch):
    root = tmp_path / "O'Brien $([danger]) project"
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return SimpleNamespace(stdout='{"enabled":true}')

    monkeypatch.setattr(startup.subprocess, "run", fake_run)
    assert startup._operation("enable", root, "owner's config.toml", python_executable=python) == {"enabled": True}
    assert str(root) not in captured["argv"][-1]
    assert "$([danger])" not in captured["argv"][-1]
    environment = captured["kwargs"]["env"]
    assert environment["LOCALPILOT_STARTUP_EXECUTABLE"] == str(pythonw.resolve())
    assert environment["LOCALPILOT_STARTUP_ARGUMENTS"] == subprocess.list2cmdline(
        ["-m", "localpilot.cli", "--config", str((root / "owner's config.toml").resolve()), "desktop"])
    assert environment["LOCALPILOT_STARTUP_ROOT"] == str(root.resolve())
    assert "-Verb" not in captured["argv"]
    assert "Start-ScheduledTask" not in captured["argv"][-1]
    assert "Stop-ScheduledTask" not in captured["argv"][-1]


@pytest.mark.parametrize("elevated", [False, None])
def test_enabling_requires_existing_admin_without_even_querying_tasks(tmp_path, monkeypatch, elevated):
    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(startup, "current_process_elevated", lambda: elevated)
    monkeypatch.setattr(startup, "_operation", lambda *a, **k: pytest.fail("No ambient UAC or task change is permitted"))
    result = startup.set_startup(tmp_path, None, True)
    assert result["ok"] is False
    assert "administrator" in result["reason"]


def test_non_windows_is_inert(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(startup, "_operation", lambda *a, **k: pytest.fail("Windows tasks are unavailable"))
    assert startup.startup_status(tmp_path, None) == {"ok": True, "enabled": False}
    assert startup.set_startup(tmp_path, None, True) == {"ok": False, "reason": "not-windows"}


@pytest.mark.parametrize("owned", [True, False])
def test_status_only_counts_legacy_shortcut_for_this_root(tmp_path, monkeypatch, owned):
    root = tmp_path / "one"
    link = _link(root if owned else tmp_path / "other")
    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt", path=os.path, environ=os.environ))
    monkeypatch.setattr(startup, "_operation", lambda *a, **k: {"enabled": False, "legacy": link})
    assert startup.startup_status(root, None) == {"ok": True, "enabled": owned}


def test_disabled_preference_is_not_enabled_by_installer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt", path=os.path, environ=os.environ))
    monkeypatch.setattr(startup, "_operation", lambda action, *a, **k: calls.append(action) or {"enabled": False, "legacy": None})
    monkeypatch.setattr(startup, "current_process_elevated", lambda: pytest.fail("Disabled install requires no elevation"))
    assert startup.migrate_startup(tmp_path, None) == {"ok": True, "enabled": False, "migrated": False}
    assert calls == ["status"]


def test_shared_data_legacy_enabled_preference_migrates_with_selected_python(tmp_path, monkeypatch):
    root, old = _configs(tmp_path)
    link = _link(old, config=old / "localpilot.toml")
    calls = []

    def operation(action, *args, **kwargs):
        calls.append((action, args, kwargs))
        return {"enabled": action == "enable", "legacy": link}

    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt", path=os.path, environ=os.environ))
    monkeypatch.setattr(startup, "current_process_elevated", lambda: True)
    monkeypatch.setattr(startup, "_operation", operation)
    selected = tmp_path / "python.exe"
    assert startup.migrate_startup(root, None, legacy_root=old, python_executable=selected) == {
        "ok": True, "enabled": True, "migrated": True}
    assert [call[0] for call in calls] == ["status", "enable"]
    assert calls[-1][2] == {"python_executable": selected, "legacy": link}


def test_different_data_legacy_migration_is_rejected_before_registration(tmp_path, monkeypatch):
    root, old = _configs(tmp_path, same=False)
    calls = []
    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt", path=os.path, environ=os.environ))
    monkeypatch.setattr(startup, "_operation", lambda action, *a, **k: calls.append(action) or {"enabled": False, "legacy": _link(old)})
    with pytest.raises(ValueError, match="different LocalPilot data"):
        startup.migrate_startup(root, None, legacy_root=old)
    assert calls == ["status"]


def test_validate_real_configs_sharing_data_returns_exact_legacy_config(tmp_path):
    root, old = _configs(tmp_path)
    assert startup.validate_legacy_install(root, "localpilot.toml", old) == (old / "localpilot.toml").resolve()


def test_validate_different_or_missing_configs_refuses_alias(tmp_path):
    root, old = _configs(tmp_path, same=False)
    with pytest.raises(ValueError, match="different LocalPilot data"):
        startup.validate_legacy_install(root, None, old)
    (old / "localpilot.toml").unlink()
    with pytest.raises(ValueError, match="existing configuration files"):
        startup.validate_legacy_install(root, None, old)


@pytest.mark.parametrize("change", ["root", "target", "arguments"])
def test_legacy_shortcut_ownership_refuses_other_launches(tmp_path, change):
    root = tmp_path.resolve()
    link = _link(root, config=root / "localpilot.toml")
    link[change] = {"root": str(root / "other"), "target": str(root / "evil.exe"),
                    "arguments": link["arguments"] + " --arbitrary"}[change]
    assert not startup._owned_legacy(link, root, root / "localpilot.toml")


# Exercise the production PowerShell control flow with every Scheduler/COM
# mutation replaced by local functions. No real task is read, created or run.
_MOCK_SCHEDULER = r"""
function Get-IdentitySid { return 'S-1-5-21-111-222-333-1001' }
function Resolve-IdentitySid([string]$identity) {
    if ($identity -in @($sid, 'test-user', 'TEST-PC\test-user')) { return $sid }
    return $null
}
function Test-Administrator { return $true }
$script:task = $null
$script:previous = $null
function Record([string]$value) { Add-Content -LiteralPath $env:TEST_RECORD -Value $value }
function Make-Task([bool]$enabled) {
    return [pscustomobject]@{
        TaskName=$taskName; Description=$description; State='Ready'
        Actions=@([pscustomobject]@{Execute=$executable;Arguments=$arguments;WorkingDirectory=$root})
        Principal=[pscustomobject]@{UserId=$sid;LogonType='Interactive';RunLevel='Highest'}
        Settings=[pscustomobject]@{Enabled=$enabled;MultipleInstances='IgnoreNew'}
        Triggers=@([pscustomobject]@{UserId=$sid;CimClass=[pscustomobject]@{CimClassName='MSFT_TaskLogonTrigger'}})
    }
}
function Get-ScheduledTask {
    param($TaskPath, $ErrorAction)
    if ($TaskPath -cne '\') { throw 'Wrong task namespace' }
    if (-not $script:initialized) {
        $script:initialized=$true
        if ($env:TEST_EXISTING -eq 'disabled') { $script:task=Make-Task $false; $script:previous=$script:task }
        if ($env:TEST_EXISTING -eq 'enabled') { $script:task=Make-Task $true; $script:previous=$script:task }
        if ($env:TEST_EXISTING -eq 'foreign') {
            $script:task=Make-Task $true; $script:task.Description='Different program'
        }
    }
    return $script:task
}
function Get-LegacyShortcut {
    if ($env:TEST_CHANGED -eq 'true') { return @{target='different.exe';root=$root;arguments=$arguments} }
    return $legacyExpected
}
function New-ScheduledTaskAction { param($Execute,$Argument,$WorkingDirectory) Record 'action'; return [pscustomobject]@{Execute=$Execute;Arguments=$Argument;WorkingDirectory=$WorkingDirectory} }
function New-ScheduledTaskTrigger { param([switch]$AtLogOn,$User) Record 'trigger'; return [pscustomobject]@{UserId=$User;CimClass=[pscustomobject]@{CimClassName='MSFT_TaskLogonTrigger'}} }
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel) Record ('principal:'+$LogonType+':'+$RunLevel); return [pscustomobject]@{UserId=$UserId;LogonType=$LogonType;RunLevel=$RunLevel} }
function New-ScheduledTaskSettingsSet { param($MultipleInstances,$ExecutionTimeLimit,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries) Record ('settings:'+$MultipleInstances+':'+$ExecutionTimeLimit.TotalSeconds); return [pscustomobject]@{Enabled=$true;MultipleInstances=$MultipleInstances} }
function Register-ScheduledTask {
    param($TaskName,$TaskPath,$Action,$Trigger,$Principal,$Settings,$Description,[switch]$Force,$ErrorAction,$Xml)
    if ($Xml) { Record 'restore'; $script:task=$script:previous; return }
    Record 'register'
    $script:task=[pscustomobject]@{TaskName=$TaskName;Description=$Description;Actions=@($Action);Triggers=@($Trigger);Principal=$Principal;Settings=$Settings;State='Ready'}
    if ($env:TEST_INVALID -eq 'level') { $script:task.Principal.RunLevel='Limited' }
    if ($env:TEST_INVALID -eq 'trigger') { $script:task.Triggers[0].UserId='another-user' }
    if ($env:TEST_INVALID -eq 'execute') { $script:task.Actions[0].Execute='different.exe' }
    if ($env:TEST_INVALID -eq 'enabled') { $script:task.Settings.Enabled=$false }
    if ($env:TEST_INVALID -eq 'account-name') {
        $script:task.Principal.UserId='test-user'
        $script:task.Triggers[0].UserId='TEST-PC\test-user'
    }
    return $script:task
}
function Export-ScheduledTask { param($TaskName,$TaskPath,$ErrorAction) Record 'export'; return '<previous-task/>' }
function Disable-ScheduledTask { param($TaskName,$TaskPath,$ErrorAction) Record 'disable'; $script:task.Settings.Enabled=$false }
function Unregister-ScheduledTask { param($TaskName,$TaskPath,$Confirm,$ErrorAction) Record 'unregister'; $script:task=$null }
function Remove-Item { param($LiteralPath,$ErrorAction) Record 'remove-link' }
"""


def _run_mock_scheduler(tmp_path: Path, *, operation="enable", existing="", invalid="", changed=False):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell needed for mocked scheduler control flow")
    executable = tmp_path / "pythonw.exe"
    executable.write_bytes(b"")
    record = tmp_path / "events.txt"
    script = tmp_path / "mock-startup.ps1"
    script.write_text(startup._TASK_FUNCTIONS + _MOCK_SCHEDULER + startup._TASK_ACTION, encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "LOCALPILOT_STARTUP_OPERATION": operation,
        "LOCALPILOT_STARTUP_ROOT": str(tmp_path),
        "LOCALPILOT_STARTUP_ROOT_HASH": "1234567890abcdef",
        "LOCALPILOT_STARTUP_EXECUTABLE": str(executable),
        "LOCALPILOT_STARTUP_ARGUMENTS": subprocess.list2cmdline(["-m", "localpilot.cli", "--config", str(tmp_path / "localpilot.toml"), "desktop"]),
        "LOCALPILOT_STARTUP_SHORTCUT": str(tmp_path / "LocalPilot.lnk"),
        "LOCALPILOT_STARTUP_LEGACY": json.dumps(_link(tmp_path)),
        "TEST_RECORD": str(record), "TEST_EXISTING": existing, "TEST_INVALID": invalid,
        "TEST_CHANGED": "true" if changed else "false",
    })
    # Command accepts only a fixed, quoted file path; every scheduler operation
    # in that file is a mock. No execution-policy override or elevation occurs.
    command = "& '" + str(script).replace("'", "''") + "'"
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=environment, timeout=30, creationflags=startup.hidden_process_creation_flags())
    events = record.read_text(encoding="utf-8-sig").splitlines() if record.exists() else []
    return result, events


def test_highest_task_is_verified_before_legacy_link_removed(tmp_path):
    result, events = _run_mock_scheduler(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"enabled": True}
    assert "principal:Interactive:Highest" in events
    assert "settings:IgnoreNew:0" in events
    assert events.index("register") < events.index("remove-link")
    assert "restore" not in events and "unregister" not in events


def test_scheduler_account_names_are_verified_by_sid_before_migration(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, invalid="account-name")
    assert result.returncode == 0, result.stderr
    assert events.index("register") < events.index("remove-link")
    assert "restore" not in events and "unregister" not in events


@pytest.mark.skipif(os.name != "nt", reason="Windows account identity")
def test_real_windows_account_names_resolve_to_current_sid():
    script = startup._TASK_FUNCTIONS + r'''
$sid = Get-IdentitySid
$account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
if ((Resolve-IdentitySid $sid) -ne $sid -or (Resolve-IdentitySid $account) -ne $sid -or
    (Resolve-IdentitySid ($account.Split('\')[-1])) -ne $sid -or
    (Resolve-IdentitySid 'localpilot-nonexistent-account-9a2d') -ne $null) { exit 1 }
'''
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, timeout=30,
                            creationflags=startup.hidden_process_creation_flags())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("invalid", ["level", "trigger", "execute", "enabled"])
def test_failed_new_task_verification_removes_only_new_task_and_preserves_link(tmp_path, invalid):
    result, events = _run_mock_scheduler(tmp_path, invalid=invalid)
    assert result.returncode != 0
    assert "unregister" in events
    assert "remove-link" not in events


def test_failed_replacement_restores_previous_disabled_task(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, existing="disabled", invalid="level")
    assert result.returncode != 0
    assert events[-2:] == ["restore", "disable"]
    assert "unregister" not in events and "remove-link" not in events


def test_changed_legacy_link_rolls_back_replacement_and_preserves_link(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, existing="enabled", changed=True)
    assert result.returncode != 0
    assert "restore" in events and "remove-link" not in events


def test_foreign_task_is_preserved_without_mutation(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, existing="foreign")
    assert result.returncode != 0
    assert events == []


def test_disabled_task_status_does_not_mutate_anything(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, operation="status", existing="disabled")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["enabled"] is False
    assert events == []


def test_disable_only_disables_owned_task_and_removes_verified_owned_link(tmp_path):
    result, events = _run_mock_scheduler(tmp_path, operation="disable", existing="enabled")
    assert result.returncode == 0, result.stderr
    assert events == ["disable", "remove-link"]
    assert json.loads(result.stdout) == {"enabled": False}
