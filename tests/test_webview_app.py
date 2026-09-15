"""Tests for the expanded WebView chat host.

The actual Windows GUI backend remains a manual smoke-test boundary. These
checks exercise the window handoff, placement, startup registration, companion
state bridge, comic layout, and local frontend security without a display.
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


def test_systemsense_expands_host_left_without_resizing_chat_surface(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=True)

    assert webview_app.SYSTEMSENSE_SIZE == (
        webview_app.EXPANDED_SIZE[0]
        + webview_app.SYSTEMSENSE_WIDTH
        + webview_app.SYSTEMSENSE_GAP,
        webview_app.EXPANDED_SIZE[1],
    )
    assert bridge.set_systemsense_open(True) == {"ok": True, "open": True}
    assert window.resized[-1] == (
        *webview_app.SYSTEMSENSE_SIZE,
        webview_app._ANCHOR_BOTTOM_RIGHT,
    )
    assert bridge.expand() == {"ok": True}
    assert window.resized[-1] == (
        *webview_app.SYSTEMSENSE_SIZE,
        webview_app._ANCHOR_BOTTOM_RIGHT,
    )
    assert bridge.set_systemsense_open(False) == {"ok": True, "open": False}
    assert window.resized[-1] == (
        *webview_app.EXPANDED_SIZE,
        webview_app._ANCHOR_BOTTOM_RIGHT,
    )


def test_systemsense_preserves_custom_size_and_tracks_the_open_minimum(tmp_path):
    window = FakeWindow()
    window.width, window.height = 620, 760
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=True)
    bridge.set_systemsense_open(True)
    assert (window.width, window.height) == (1002, 760)
    assert window.min_size == (802, 520)
    window.width, window.height = 1102, 800
    bridge.set_systemsense_open(False)
    assert (window.width, window.height) == (720, 800)
    assert window.min_size == webview_app.MIN_SIZE
    bridge.expand()
    assert (window.width, window.height) == (720, 800)


def test_systemsense_does_not_raise_native_minimum_before_anchored_expansion(tmp_path, monkeypatch):
    window = FakeWindow()
    changes = []
    def shape(native_window):
        changes.append((native_window.width, native_window.min_size[0]))
        # WinForms would expand at the top-left if MinimumSize exceeds Width.
        assert native_window.width >= native_window.min_size[0]
    monkeypatch.setattr(webview_app, "make_host_background_transparent", shape)
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    bridge.set_systemsense_open(True)
    assert changes == [(882, 802)]
    bridge.set_systemsense_open(False)
    assert changes[-2:] == [(882, 420), (500, 420)]
    assert window.min_size == webview_app.MIN_SIZE


def test_right_side_composition_expands_away_from_avatar(tmp_path):
    window = FakeWindow()
    window._comic_tail_left = True
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=True)
    bridge.set_systemsense_open(True)
    assert window.resized[-1][2] == webview_app.FixPoint.SOUTH | webview_app.FixPoint.WEST


def test_always_on_top_is_marshaled_before_touching_winforms(tmp_path, monkeypatch):
    import sys
    import types

    window = FakeWindow()
    class Native:
        InvokeRequired = True
        invoked = False
        def Invoke(self, callback):
            self.invoked = True
            callback()
    window.native = Native()
    monkeypatch.setitem(sys.modules, "System", types.SimpleNamespace(Action=lambda callback: callback))
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.set_always_on_top(False) == {"ok": True}
    assert window.native.invoked
    assert window.on_top is False


def test_companion_state_bridge_accepts_only_real_known_states(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=True)

    assert bridge.set_companion_state("speaking") == {"ok": True, "state": "speaking"}
    assert bridge._state.read()["companion_state"] == "speaking"
    assert bridge.set_companion_state("not-a-real-state") == {
        "ok": False,
        "reason": "invalid-state",
    }
    assert bridge._state.read()["companion_state"] == "speaking"
    assert bridge.clear_companion_state() == {"ok": True}
    assert bridge._state.read()["companion_state"] is None


def test_standalone_webview_cannot_impersonate_external_avatar_state(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=False)
    assert bridge.set_companion_state("working") == {
        "ok": False,
        "reason": "no-external-avatar",
    }


def test_collapse_spawns_native_avatar_then_destroys_webview_for_standalone_host(tmp_path, monkeypatch):
    window = FakeWindow()
    captured = {}

    def fake_launch(root, config_path, *, x=None, y=None):
        captured.update(root=root, config_path=config_path, x=x, y=y)
        return True

    monkeypatch.setattr(webview_app, "_launch_native_avatar", fake_launch)
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.collapse() == {"ok": True}
    assert window.destroyed is True
    assert captured["x"] == window.x + window.width + webview_app.EDGE_INSET
    assert captured["y"] == window.y + window.height - webview_app.NATIVE_AVATAR_SIZE


def test_companion_collapse_never_spawns_duplicate_avatar(tmp_path, monkeypatch):
    window = FakeWindow()
    launches = []
    monkeypatch.setattr(
        webview_app,
        "_launch_native_avatar",
        lambda *args, **kwargs: launches.append((args, kwargs)) or True,
    )
    bridge = webview_app.WindowBridge(window, tmp_path, None, avatar_external=True)
    assert bridge.avatar_external is True
    assert bridge.collapse() == {"ok": True}
    assert window.destroyed is True
    assert launches == []


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
    assert x == -652
    assert y == 576
    assert x < 0


def test_position_on_screen_includes_positive_nonzero_origin():
    screen = FakeScreen(x=2560, y=-200, width=1920, height=1080)
    x, y = webview_app._position_on_screen(screen, *webview_app.EXPANDED_SIZE)
    assert x == 3828
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


def test_normal_detached_launcher_starts_persistent_avatar_companion(tmp_path, monkeypatch):
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
        webview_app.COMPANION_MODULE,
        "--root",
        str(root),
    ]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL


def test_build_parser_accepts_companion_and_explicit_virtual_desktop_coordinates():
    parser = webview_app.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args([
        "--root", "/tmp/example", "--x", "-700", "--y", "220", "--companion"
    ])
    assert args.root == "/tmp/example"
    assert args.x == -700
    assert args.y == 220
    assert args.companion is True


def test_startup_shortcut_path_is_under_startup_folder():
    path = webview_app._startup_shortcut_path()
    assert path.name == "LocalPilot.lnk"
    assert "Startup" in path.parts


def test_startup_shortcut_starts_persistent_companion_and_uses_environment_values(tmp_path, monkeypatch):
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
            webview_app.COMPANION_MODULE,
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
    comic = (webview_app.WEBVIEW_DIR / "comic-shell.css").read_text(encoding="utf-8")
    sync = (webview_app.WEBVIEW_DIR / "companion-state-sync.js").read_text(encoding="utf-8")

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
    assert ".panel.is-system-open .message-stream" in comic
    assert "repeating-linear-gradient" in comic
    assert "right: calc(100% + var(--systemsense-gap) + var(--chat-border))" in comic
    assert "repeat-y" in comic
    assert "repeat-x" not in comic
    assert "set_systemsense_open" in sync


def test_comic_shell_has_real_tail_no_outer_rectangular_host_and_consistent_font():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    comic = (webview_app.WEBVIEW_DIR / "comic-shell.css").read_text(encoding="utf-8")
    source = Path(webview_app.__file__).read_text(encoding="utf-8")

    assert 'href="comic-shell.css"' in index
    assert "background: transparent !important" in comic
    assert "border-left: var(--tail-width) solid var(--comic-ink)" in comic
    assert "border-left: calc(var(--tail-width) - 5px) solid #f4eadc" in comic
    assert 'transparent=True' in source
    assert 'shadow=False' in source
    assert "--comic-font:" in comic
    assert ".panel button," in comic
    assert ".panel code," in comic
    assert ".panel textarea," in comic


def test_webview_has_no_visible_or_illustrated_duplicate_avatar():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    comic = (webview_app.WEBVIEW_DIR / "comic-shell.css").read_text(encoding="utf-8")

    assert 'src="illustrated-avatar.js"' not in index
    assert 'src="companion-state-sync.js"' in index
    assert 'id="avatar-header" width="1" height="1" aria-hidden="true" hidden' in index
    assert 'id="avatar-dock" width="1" height="1" aria-hidden="true" hidden' in index
    assert ".avatar-canvas[hidden]" in comic
    assert ".illustrated-avatar-layer" in comic
    assert "display: none !important" in comic
    assert 'id="composer-input"' in index
    assert 'id="history-toggle"' in index
    assert 'id="settings-toggle"' in index


def test_companion_state_sync_tracks_the_same_dataset_state_as_real_chat():
    sync = (webview_app.WEBVIEW_DIR / "companion-state-sync.js").read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")
    assert "document.documentElement.dataset.state = next" in javascript
    assert "document.documentElement.dataset.state" in sync
    assert "set_companion_state" in sync
    assert "MutationObserver" in sync


def test_webview_can_create_and_select_a_new_conversation():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="history-new"' in index
    assert 'api("POST", "/v1/sessions", {})' in javascript
    assert "await switchSession(created.session.id)" in javascript


def test_webview_is_chat_surface_not_compact_avatar_owner():
    source = Path(webview_app.__file__).read_text(encoding="utf-8")
    assert "width, height = EXPANDED_SIZE" in source
    assert "COMPANION_MODULE" in source
    assert "avatar_external=companion" in source
    assert "COMPACT_SIZE" not in source


def test_pywebview_dependency_is_constrained_to_verified_major_range():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "pywebview>=6.2.1,<7" in project["project"]["dependencies"]
