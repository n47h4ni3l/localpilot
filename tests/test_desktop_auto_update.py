from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from localpilot import desktop_auto_update, desktop_updater
from localpilot.desktop_state import DesktopUIState


def _result(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_update_check_due_uses_fixed_internal_30_minute_cadence(tmp_path):
    state = DesktopUIState(tmp_path)
    now = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    assert desktop_auto_update.update_check_due(state, now=now)
    state.update(update_last_checked=(now - timedelta(minutes=29)).isoformat())
    assert not desktop_auto_update.update_check_due(state, now=now)
    state.update(update_last_checked=(now - timedelta(minutes=31)).isoformat())
    assert desktop_auto_update.update_check_due(state, now=now)


def test_safe_boundary_blocks_live_foreground_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_auto_update, "active_foreground_turns", lambda data: ({"request_id": "r"},))
    monkeypatch.setattr(desktop_auto_update, "evolution_active", lambda data: False)
    assert desktop_auto_update.update_safe_boundary(tmp_path) is False


def test_safe_boundary_blocks_live_evolution(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_auto_update, "active_foreground_turns", lambda data: ())
    monkeypatch.setattr(desktop_auto_update, "evolution_active", lambda data: True)
    assert desktop_auto_update.update_safe_boundary(tmp_path) is False


def test_prepare_handoff_records_exact_old_and_target_sha(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = tmp_path / "data"
    state = DesktopUIState(data)
    state.update(
        automatic_updates=True,
        update_last_checked=datetime.now(timezone.utc).isoformat(),
        update_available=True,
        update_target_version="bbbbbbb",
    )
    old_sha = "a" * 40
    target_sha = "b" * 40

    monkeypatch.setattr(desktop_auto_update, "update_safe_boundary", lambda data_dir: True)
    monkeypatch.setattr(
        desktop_auto_update,
        "current_update_status",
        lambda root_path, ui_state: SimpleNamespace(
            check_error=None,
            update_available=True,
        ),
    )

    def fake_run(call_root: Path, args: list[str], *, timeout: int = 30):
        assert call_root == root.resolve()
        if args[:3] == ["git", "branch", "--show-current"]:
            return _result(stdout="main")
        if args[:3] == ["git", "status", "--porcelain"]:
            return _result(stdout="")
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return _result(stdout=old_sha if args[3] == "HEAD^{commit}" else target_sha)
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return _result()
        raise AssertionError(args)

    monkeypatch.setattr(desktop_auto_update, "_run", fake_run)
    handoff = desktop_auto_update.prepare_update_handoff(
        root,
        state,
        config_path=None,
        remote="origin",
        main_branch="main",
    )
    assert handoff is not None
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    assert payload["old_sha"] == old_sha
    assert payload["target_sha"] == target_sha
    assert payload["main_branch"] == "main"
    assert payload["remote"] == "origin"


def test_updater_process_matching_is_scoped_to_repo_and_localpilot(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    class FakeProcess:
        pid = 12345
        def cmdline(self):
            return ["python.exe", "-m", "localpilot.broker", "--root", str(root)]
        def cwd(self):
            return str(root)

    monkeypatch.setattr(desktop_updater.os, "getpid", lambda: 99999)
    assert desktop_updater._belongs_to_localpilot(FakeProcess(), root.resolve()) is True

    class OtherProcess(FakeProcess):
        def cmdline(self):
            return ["python.exe", "some_other_app.py", str(root)]

    assert desktop_updater._belongs_to_localpilot(OtherProcess(), root.resolve()) is False


def test_fast_forward_uses_exact_target_and_verifies_head(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    target = "b" * 40
    calls = []

    def fake_run(call_root: Path, args: list[str], *, timeout: int = 120):
        calls.append(tuple(args))
        if args[:3] == ["git", "merge", "--ff-only"]:
            return _result()
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return _result(stdout=target)
        raise AssertionError(args)

    monkeypatch.setattr(desktop_updater, "_run", fake_run)
    desktop_updater._fast_forward(root, target)
    assert ("git", "merge", "--ff-only", "--no-edit", target) in calls
