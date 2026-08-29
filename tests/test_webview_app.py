"""Tests for the expanded WebView chat host.

The actual Windows GUI backend remains a manual smoke-test boundary. These
checks exercise the window handoff, placement, startup registration, and local
frontend security without requiring a display server.
"""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

from localpilot import webview_app


class FakeWindow:
    def __init__(self) -> None:
        self.resized: list[tuple[int, int, object]] = []
        self.on_top_value: bool | None = None
        self.destroyed = False
        self.x = 100
        self.y = 200
        self.width = webview_app.EXPANDED_SIZE[0]
        self.height = webview_app.EXPANDED_SIZE[1]

    def resize(self, width: int, height: int, fix_point=None) -> None:
        self.resized.append((width, height, fix_point))
        self.width = width
        self.height = height

    def destroy(self) -> None:
        self.destroyed = True

    @property
    def on_top(self) -> bool | None:
        return self.on_top_value

    @on_top.setter
    def on_top(self, value: bool) -> None:
        self.on_top_value = value


class FakeScreen:
    def __init__(self, *, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def test_expand_keeps_webview_at_expanded_size(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.expand() == {"ok": True}
    assert window.resized == [
        (*webview_app.EXPANDED_SIZE, webview_app._ANCHOR_BOTTOM_RIGHT)
    ]


def test_collapse_spawns_native_avatar_then_destroys_webview(tmp_path, monkeypatch):
    window = FakeWindow()
    captured = {}

    def fake_launch(root, config_path, *, x=None, y=None):
        captured.update(root=root, config_path=config_path, x=x, y=y)
        return True

    monkeypatch.setattr(webview_app, "_launch_native_avatar", fake_launch)
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.collapse() == {"ok": True}
    assert window.destroyed is True
    assert captured["x"] == window.x + window.width - webview_app.NATIVE_AVATAR_SIZE
    assert captured["y"] == window.y + window.height - webview_app.NATIVE_AVATAR_SIZE


def test_collapse_keeps_chat_open_if_native_avatar_cannot_start(tmp_path, monkeypatch):
    window = FakeWindow()
    monkeypatch.setattr(webview_app, "_launch_native_avatar", lambda *args, **kwargs: False)
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.collapse() == {"ok": False, "reason": "native-avatar-launch-failed"}
    assert window.destroyed is False


def test_set_always_on_top_sets_window_and_persists_state(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.set_always_on_top(False) == {"ok": True}
    assert window.on_top_value is False
    assert bridge._state.read()["always_on_top"] is False


def test_start_with_windows_refuses_on_non_windows_without_touching_pathlib(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    if webview_app.os.name != "nt":
        assert bridge.set_start_with_windows(True) == {"ok": False, "reason": "not-windows"}


def test_get_start_with_windows_reports_disabled_when_no_shortcut_exists(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    result = bridge.get_start_with_windows()
    assert result["ok"] is True
    assert isinstance(result["enabled"], bool)


def test_open_config_file_without_a_config_path_reports_a_clear_reason(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.open_config_file() == {"ok": False, "reason": "no-config-path"}


def test_bridge_payload_carries_no_more_than_frontend_needs():
    class FakeClient:
        base_url = "http://127.0.0.1:8765"
        token = "secret-token"

    assert webview_app._bridge_payload(FakeClient(), "/some/config.toml") == {
        "baseUrl": "http://127.0.0.1:8765",
        "token": "secret-token",
        "hasConfigPath": True,
    }


def test_initial_position_degrades_gracefully_without_display_backend():
    x, y = webview_app._initial_position(*webview_app.EXPANDED_SIZE)
    assert (x is None and y is None) or (isinstance(x, int) and isinstance(y, int))


def test_position_on_screen_preserves_negative_virtual_desktop_origin():
    screen = FakeScreen(x=-1920, y=160, width=1920, height=1080)
    x, y = webview_app._position_on_screen(screen, *webview_app.EXPANDED_SIZE)
    assert x == -444
    assert y == 576
    assert x < 0


def test_position_on_screen_includes_positive_nonzero_origin():
    screen = FakeScreen(x=2560, y=-200, width=1920, height=1080)
    x, y = webview_app._position_on_screen(screen, *webview_app.EXPANDED_SIZE)
    assert x == 4036
    assert y == 216


def test_desktop_python_executable_prefers_pythonw_on_windows(tmp_path):
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_text("", encoding="utf-8")
    pythonw.write_text("", encoding="utf-8")
    assert webview_app._desktop_python_executable(python, platform_name="nt") == pythonw.resolve()
    assert webview_app._desktop_python_executable(python, platform_name="posix") == python.resolve()


def test_console_script_detaches_only_for_normal_windows_desktop_entrypoint():
    assert webview_app._should_detach_gui("C:/venv/Scripts/localpilot.exe", platform_name="nt")
    assert not webview_app._should_detach_gui("C:/venv/Scripts/python.exe", platform_name="nt")


def test_normal_detached_launcher_starts_native_avatar(tmp_path, monkeypatch):
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(webview_app, "_desktop_python_executable", lambda: pythonw)

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(webview_app.subprocess, "Popen", fake_popen)
    root = tmp_path / "repo"
    root.mkdir()
    assert webview_app._launch_detached(root, None) is True
    assert captured["argv"] == [
        str(pythonw),
        "-m",
        "localpilot.native_avatar",
        "--root",
        str(root),
    ]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL


def test_build_parser_accepts_explicit_virtual_desktop_coordinates():
    parser = webview_app.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["--root", "/tmp/example", "--x", "-700", "--y", "220"])
    assert args.root == "/tmp/example"
    assert args.x == -700
    assert args.y == 220


def test_startup_shortcut_path_is_under_startup_folder():
    path = webview_app._startup_shortcut_path()
    assert path.name == "LocalPilot.lnk"
    assert "Startup" in path.parts


def test_startup_shortcut_starts_native_avatar_and_uses_environment_values(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

    monkeypatch.setattr(webview_app.subprocess, "run", fake_run)
    root = tmp_path / "O'Brien $([danger]) project"
    target = tmp_path / "Startup" / "LocalPilot.lnk"
    config = root / "owner's localpilot.toml"
    webview_app._write_startup_shortcut(target, root, str(config))

    script = captured["argv"][-1]
    assert str(root) not in script
    assert "$([danger])" not in script
    environment = captured["kwargs"]["env"]
    assert environment["LOCALPILOT_ARGUMENTS"] == subprocess.list2cmdline(
        [
            "-m",
            "localpilot.native_avatar",
            "--root",
            str(root.resolve()),
            "--config",
            str(config.resolve()),
        ]
    )


def test_frontend_is_fully_local_and_uses_strict_csp():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in index
    assert "http-equiv=\"Content-Security-Policy\"" in index
    assert "default-src 'none'" in index
    assert "script-src 'self'" in index
    assert "style-src 'self'" in index
    assert "'unsafe-inline'" not in index
    assert "'unsafe-eval'" not in index
    assert ".innerHTML" not in javascript
    assert "renderSafeMarkdown" in javascript
    assert "document.createTextNode" in javascript
    assert 'document.createElement(token.startsWith("**") ? "strong" : "code")' in javascript


def test_systemsense_glance_panel_uses_authenticated_summary_surface_only():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")
    stylesheet = (webview_app.WEBVIEW_DIR / "app.css").read_text(encoding="utf-8")

    assert 'id="system-toggle"' in index
    assert 'aria-controls="system-panel"' in index
    assert 'id="system-panel"' in index
    assert 'aria-hidden="true"' in index
    assert "Read-only · local telemetry" in index
    assert 'api("GET", "/v1/systemsense/summary")' in javascript
    assert 'api("POST", "/v1/systemsense' not in javascript
    assert "collect_if_missing" not in javascript
    assert ".panel.is-system-open .system-panel" in stylesheet
    assert ".panel.is-system-open .composer-wrap" in stylesheet


def test_webview_can_create_and_select_a_new_conversation():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="history-new"' in index
    assert 'api("POST", "/v1/sessions", {})' in javascript
    assert "await switchSession(created.session.id)" in javascript


def test_webview_is_expanded_chat_only_not_compact_avatar():
    source = Path(webview_app.__file__).read_text(encoding="utf-8")
    assert "width, height = EXPANDED_SIZE" in source
    assert '"localpilot.native_avatar"' in source
    assert "COMPACT_SIZE" not in source


def test_pywebview_dependency_is_constrained_to_verified_major_range():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "pywebview>=6.2.1,<7" in project["project"]["dependencies"]
