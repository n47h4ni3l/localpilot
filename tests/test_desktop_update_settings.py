from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from localpilot import desktop_update_status, webview_app
from localpilot.desktop_state import DesktopUIState


def _result(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_desktop_update_preference_defaults_off_and_persists(tmp_path):
    state = DesktopUIState(tmp_path)
    values = state.read()
    assert values["automatic_updates"] is False
    assert values["update_last_checked"] is None

    state.update(automatic_updates=True)
    reloaded = DesktopUIState(tmp_path).read()
    assert reloaded["automatic_updates"] is True


def test_update_check_records_remote_advance_without_modifying_checkout(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    state = DesktopUIState(tmp_path / "data")
    calls: list[tuple[str, ...]] = []
    head = "a" * 40
    target = "b" * 40

    def fake_run(call_root: Path, args: list[str], *, timeout: int = 30):
        assert call_root == root
        calls.append(tuple(args))
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return _result(stdout=str(root))
        if args[:3] == ["git", "branch", "--show-current"]:
            return _result(stdout="main")
        if args[:3] == ["git", "status", "--porcelain"]:
            return _result(stdout="")
        if args[:2] == ["git", "fetch"]:
            return _result()
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return _result(stdout=head if args[3] == "HEAD^{commit}" else target)
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return _result()
        if args[:3] == ["git", "rev-parse", "--short=7"]:
            return _result(stdout="aaaaaaa")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(desktop_update_status, "_run", fake_run)
    result = desktop_update_status.check_for_updates(root, state)

    assert result.current_version == "aaaaaaa"
    assert result.update_available is True
    assert result.target_version == "bbbbbbb"
    assert result.last_checked
    assert result.check_error is None
    assert not any(call[:2] in {("git", "merge"), ("git", "pull"), ("git", "reset")} for call in calls)


def test_update_check_fails_closed_on_dirty_main(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    state = DesktopUIState(tmp_path / "data")

    def fake_run(call_root: Path, args: list[str], *, timeout: int = 30):
        assert call_root == root
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return _result(stdout=str(root))
        if args[:3] == ["git", "branch", "--show-current"]:
            return _result(stdout="main")
        if args[:3] == ["git", "status", "--porcelain"]:
            return _result(stdout=" M localpilot/example.py")
        if args[:3] == ["git", "rev-parse", "--short=7"]:
            return _result(stdout="abc1234")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(desktop_update_status, "_run", fake_run)
    result = desktop_update_status.check_for_updates(root, state)
    assert result.update_available is None
    assert result.check_error and "uncommitted work" in result.check_error
    assert result.last_checked


def test_window_bridge_persists_automatic_update_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        webview_app,
        "current_update_status",
        lambda root, state: desktop_update_status.UpdateStatus(
            bool(state.read()["automatic_updates"]),
            "abc1234",
            None,
            None,
            None,
            None,
        ),
    )

    class FakeWindow:
        width = webview_app.EXPANDED_SIZE[0]
        height = webview_app.EXPANDED_SIZE[1]
        x = 0
        y = 0

    bridge = webview_app.WindowBridge(FakeWindow(), tmp_path, None)
    result = bridge.set_automatic_updates(True)
    assert result["ok"] is True
    assert result["enabled"] is True
    assert bridge._state.read()["automatic_updates"] is True


def test_settings_surface_stays_compact_and_has_only_requested_update_controls():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="toggle-auto-updates"' in index
    assert 'id="update-last-checked"' in index
    assert 'id="update-current-version"' in index
    assert "Check every" not in index
    assert "Update source" not in index
    assert "Check for updates now" not in index
    assert 'src="settings-updates.js"' in index
    assert 'href="settings-updates.css"' in index
