from __future__ import annotations

from localpilot import native_avatar, webview_app


class FakeWindow:
    def __init__(self) -> None:
        self.destroyed = False
        self.x = 100
        self.y = 200
        self.width = webview_app.EXPANDED_SIZE[0]
        self.height = webview_app.EXPANDED_SIZE[1]
        self.scripts: list[str] = []

    def destroy(self) -> None:
        self.destroyed = True

    def evaluate_js(self, script: str) -> None:
        self.scripts.append(script)


class FakeTkRoot:
    def __init__(self) -> None:
        self.geometry_calls: list[str] = []

    def winfo_id(self) -> int:
        return 123

    def update_idletasks(self) -> None:
        return None

    def geometry(self, value: str) -> None:
        self.geometry_calls.append(value)


def test_chat_position_is_clamped_inside_avatar_monitor_work_area():
    work_area = (-1920, 0, 0, 1040)
    x, y = native_avatar._chat_position_from_avatar(-1900, 10, work_area)
    width, height = native_avatar.EXPANDED_SIZE
    assert x == -1920
    assert y == 0
    assert x + width <= work_area[2]
    assert y + height <= work_area[3]


def test_chat_position_keeps_normal_anchor_when_it_already_fits():
    work_area = (0, 0, 1920, 1040)
    x, y = native_avatar._chat_position_from_avatar(1700, 820, work_area)
    width, height = native_avatar.EXPANDED_SIZE
    assert x + width == 1700 + native_avatar.AVATAR_SIZE
    assert y + height == 820 + native_avatar.AVATAR_SIZE


def test_webview_collapse_prefers_persisted_avatar_home_position(tmp_path, monkeypatch):
    window = FakeWindow()
    captured = {}

    def fake_launch(root, config_path, *, x=None, y=None):
        captured.update(x=x, y=y)
        return True

    monkeypatch.setattr(webview_app, "_launch_native_avatar", fake_launch)
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    bridge._state.update(avatar_x=-720, avatar_y=330)

    assert bridge.collapse() == {"ok": True}
    assert captured == {"x": -720, "y": 330}
    assert window.destroyed is True


def test_webview_explicit_exit_does_not_request_avatar_respawn(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)

    assert bridge.exit_requested is False
    assert bridge.exit_companion() == {"ok": True}
    assert bridge.exit_requested is True
    assert window.destroyed is True


def test_native_close_marks_real_exit_until_avatar_was_already_spawned(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    bridge.mark_native_close()
    assert bridge.exit_requested is True

    second = webview_app.WindowBridge(FakeWindow(), tmp_path / "second", None)
    second._avatar_spawned = True
    second.mark_native_close()
    assert second.exit_requested is False


def test_expanded_window_chrome_installs_drag_regions_and_close_control():
    window = FakeWindow()
    webview_app._install_expanded_window_chrome(window)
    script = window.scripts[-1]
    assert "pywebview-drag-region" in script
    assert "close-app-btn" in script
    assert "exit_companion" in script


def test_non_windows_tk_position_fallback_preserves_signed_coordinates(monkeypatch):
    root = FakeTkRoot()
    monkeypatch.setattr(native_avatar.os, "name", "posix")
    native_avatar._set_window_position(root, -640, 120)
    assert root.geometry_calls == [
        f"{native_avatar.AVATAR_SIZE}x{native_avatar.AVATAR_SIZE}-640+120"
    ]
