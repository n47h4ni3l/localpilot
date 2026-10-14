from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
from types import SimpleNamespace
import threading

import pytest

from localpilot import desktop_auto_update as updates, desktop_update_status as status
from localpilot import desktop_updater, native_avatar_companion as companion
from localpilot.desktop_state import DesktopUIState


TARGET = "b" * 40


def offered(state, now, target=TARGET):
    state.update(automatic_updates=False, update_last_checked=now.isoformat(),
                 update_available=True, update_target_sha=target, update_check_error=None)


def test_off_check_fetches_only_when_due_and_notifies_once_across_restart(tmp_path, monkeypatch):
    state = DesktopUIState(tmp_path)
    now = datetime.now(timezone.utc)
    calls = []
    def check(root, ui_state, **kwargs):
        calls.append(kwargs)
        offered(ui_state, now)
    monkeypatch.setattr(updates, "check_for_updates", check)
    assert updates.check_update_notification(tmp_path, state, remote="origin", main_branch="main", now=now) == TARGET
    assert updates.check_update_notification(tmp_path, DesktopUIState(tmp_path), remote="origin", main_branch="main", now=now) is None
    assert len(calls) == 1
    assert state.read()["automatic_updates"] is False
    assert not (tmp_path / updates.HANDOFF_FILENAME).exists()


def test_new_full_target_is_rate_limited_even_when_short_prefix_matches(tmp_path):
    state = DesktopUIState(tmp_path)
    now = datetime.now(timezone.utc)
    offered(state, now)
    assert updates.check_update_notification(tmp_path, state, remote="origin", main_branch="main", now=now) == TARGET
    changed = "b" * 7 + "c" * 33
    offered(state, now + timedelta(minutes=1), changed)
    assert updates.check_update_notification(tmp_path, state, remote="origin", main_branch="main", now=now + timedelta(minutes=1)) is None
    offered(state, now + timedelta(minutes=31), changed)
    assert updates.check_update_notification(tmp_path, state, remote="origin", main_branch="main", now=now + timedelta(minutes=31)) == changed


@pytest.mark.parametrize("changes", [
    {"automatic_updates": True}, {"update_check_error": "dirty main"},
    {"update_available": False}, {"update_target_sha": "bad"},
    {"update_dismissed_target_sha": TARGET},
])
def test_untrusted_or_unwanted_notice_is_suppressed(tmp_path, changes):
    state = DesktopUIState(tmp_path)
    now = datetime.now(timezone.utc)
    offered(state, now)
    state.update(**changes)
    assert updates.check_update_notification(tmp_path, state, remote="origin", main_branch="main", now=now) is None


def test_owner_choices_preserve_automatic_updates_preference(tmp_path):
    state = DesktopUIState(tmp_path)
    offered(state, datetime.now(timezone.utc))
    assert status.request_manual_update(state, "a" * 40)["ok"] is False
    assert status.request_manual_update(state, TARGET)["ok"] is True
    assert state.read()["update_apply_requested_sha"] == TARGET
    assert state.read()["automatic_updates"] is False
    assert status.dismiss_update_notice(state, TARGET)["ok"] is True
    assert state.read()["update_dismissed_target_sha"] == TARGET


def test_manual_handoff_keeps_exact_owner_selected_commit(tmp_path, monkeypatch):
    state = DesktopUIState(tmp_path / "data")
    offered(state, datetime.now(timezone.utc))
    status.request_manual_update(state, TARGET)
    monkeypatch.setattr(updates, "update_safe_boundary", lambda path: True)
    monkeypatch.setattr(updates, "current_update_status", lambda *args: SimpleNamespace(check_error=None, update_available=True))
    monkeypatch.setattr(updates, "check_for_updates", lambda *args, **kwargs: SimpleNamespace(check_error=None, update_available=True))
    commands = []
    def git(root, args, **kwargs):
        commands.append(args)
        if args[:3] == ["git", "branch", "--show-current"]: text = "main"
        elif args[:3] == ["git", "status", "--porcelain"]: text = ""
        elif args[:3] == ["git", "rev-parse", "--verify"]: text = "a" * 40 if args[-1] == "HEAD^{commit}" else TARGET
        elif args[:3] == ["git", "merge-base", "--is-ancestor"]: text = ""
        else: pytest.fail(str(args))
        return SimpleNamespace(returncode=0, stdout=text, stderr="")
    monkeypatch.setattr(updates, "_run", git)
    handoff = updates.prepare_update_handoff(tmp_path, state, config_path=None, remote="origin", main_branch="main", manual_target_sha=TARGET)
    value = json.loads(handoff.read_text())
    assert value["target_sha"] == TARGET and value["manual"] is True
    assert state.read()["automatic_updates"] is False
    assert not any(args[1] in {"merge", "reset", "pull"} for args in commands)


def test_manual_handoff_refuses_changed_target_and_busy_evolution(tmp_path, monkeypatch):
    state = DesktopUIState(tmp_path / "data")
    offered(state, datetime.now(timezone.utc))
    status.request_manual_update(state, TARGET)
    monkeypatch.setattr(updates, "update_safe_boundary", lambda path: False)
    monkeypatch.setattr(updates, "check_for_updates", lambda *args, **kwargs: pytest.fail("Busy cycle must defer before fetching"))
    kwargs = dict(config_path=None, remote="origin", main_branch="main", manual_target_sha=TARGET)
    assert updates.prepare_update_handoff(tmp_path, state, **kwargs) is None
    assert state.read()["update_apply_requested_sha"] == TARGET
    monkeypatch.setattr(updates, "update_safe_boundary", lambda path: True)
    monkeypatch.setattr(updates, "current_update_status", lambda *args: SimpleNamespace(check_error=None, update_available=True))
    def changed(*args, **kwargs):
        state.update(update_target_sha="c" * 40)
        return SimpleNamespace(check_error=None, update_available=True)
    monkeypatch.setattr(updates, "check_for_updates", changed)
    assert updates.prepare_update_handoff(tmp_path, state, **kwargs) is None
    assert state.read()["update_apply_requested_sha"] is None
    assert not (state.path.parent / updates.HANDOFF_FILENAME).exists()


@pytest.mark.parametrize("authorized", [False, True])
def test_external_updater_requires_matching_manual_owner_request(tmp_path, monkeypatch, authorized):
    state = DesktopUIState(tmp_path / "data")
    state.update(automatic_updates=False, update_apply_requested_sha=TARGET if authorized else "c" * 40)
    handoff = tmp_path / "handoff.json"
    payload = {"version": 1, "root": str(tmp_path), "data_dir": str(state.path.parent), "old_sha": "a" * 40,
               "target_sha": TARGET, "main_branch": "main", "manual": True}
    handoff.write_text(json.dumps(payload))
    events = []
    monkeypatch.setattr(desktop_updater, "_wait_for_parent_exit", lambda *args: None)
    monkeypatch.setattr(desktop_updater, "_verify_clean_trusted_main", lambda *args: events.append("verify"))
    monkeypatch.setattr(desktop_updater, "_run_shared_update_script", lambda *args, **kwargs: (events.append("apply") or SimpleNamespace(returncode=0, stdout="", stderr="")))
    monkeypatch.setattr(desktop_updater, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=TARGET if authorized else "a" * 40, stderr=""))
    monkeypatch.setattr(desktop_updater, "_launch_companion", lambda *args: None)
    monkeypatch.setattr(desktop_updater, "_localpilot_process_running", lambda *args: True)
    assert desktop_updater.apply_handoff(tmp_path, handoff, 1) is authorized
    assert ("apply" in events) is authorized
    assert state.read()["automatic_updates"] is False
    assert state.read()["update_apply_requested_sha"] is None


def test_wave_is_brief_idle_only_and_never_overrides_active_runtime(tmp_path, monkeypatch):
    state = DesktopUIState(tmp_path)
    offered(state, datetime.now(timezone.utc))
    app = companion.NativeAvatarCompanion.__new__(companion.NativeAvatarCompanion)
    app._chat_process = None
    app._events = queue.Queue()
    app._broker_runtime_state = "working"
    app._update_notice_target = TARGET
    app._update_notice_until = 0.0
    app.state_store = state
    app._stop = threading.Event()
    app._stop.set()
    app._draw = lambda: None
    ticks = iter([100.0, 103.0])
    monkeypatch.setattr(companion.time, "monotonic", lambda: next(ticks))
    app._drain_events()
    assert app.runtime_state == "working"  # no interruption, pending wave retained
    app._broker_runtime_state = "idle"
    app._drain_events()
    assert app.runtime_state == "suggestion"
    app._drain_events()
    assert app.runtime_state == "idle" and app._update_notice_target is None
