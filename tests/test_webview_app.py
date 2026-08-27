"""Tests for localpilot.webview_app.

Scope, deliberately: this module's window-chrome logic (resize/on_top/
startup-registration/config-opening) can be exercised without a real GUI
backend by faking the pywebview Window object, but actually *creating* a
window needs a display server this environment doesn't have — matching the
existing project's own testing boundary for Windows-only surfaces (see
tools/windows.py, which has the same gap, and tests/test_windows_actions.py
for the established pattern of testing decision logic without the OS call).

One thing this file deliberately does NOT do: monkeypatch os.name to "nt" to
exercise the Windows-only branch of set_start_with_windows(). Doing so would
make _startup_shortcut_path() construct a path through pathlib in a context
where a later pathlib.Path() call could try to instantiate a real
WindowsPath on Linux, which pathlib refuses at runtime — this is exactly the
failure mode already present in tests/test_windows_actions.py on this
platform (see the review notes). The not-windows guard clause is tested
directly on the real, unpatched platform instead, which exercises the real
early-return without touching that trap.
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

    def resize(self, width: int, height: int, fix_point=None) -> None:
        self.resized.append((width, height, fix_point))

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


def test_expand_resizes_to_the_expanded_size_anchored_bottom_right(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    result = bridge.expand()
    assert result == {"ok": True}
    assert len(window.resized) == 1
    width, height, fix_point = window.resized[0]
    assert (width, height) == webview_app.EXPANDED_SIZE
    assert fix_point == webview_app._ANCHOR_BOTTOM_RIGHT


def test_collapse_resizes_to_the_compact_size(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    result = bridge.collapse()
    assert result == {"ok": True}
    width, height, _ = window.resized[0]
    assert (width, height) == webview_app.COMPACT_SIZE


def test_compact_window_is_small_avatar_sized():
    assert webview_app.COMPACT_SIZE == (144, 144)
    assert webview_app.MIN_SIZE[0] <= webview_app.COMPACT_SIZE[0]
    assert webview_app.MIN_SIZE[1] <= webview_app.COMPACT_SIZE[1]


def test_set_always_on_top_sets_the_real_window_property(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    assert bridge.set_always_on_top(True) == {"ok": True}
    assert window.on_top_value is True
    assert bridge.set_always_on_top(False) == {"ok": True}
    assert window.on_top_value is False


def test_start_with_windows_refuses_on_non_windows_without_touching_pathlib(tmp_path):
    # Exercises the real platform (whatever it is) rather than monkeypatching
    # os.name — see module docstring for why.
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    if webview_app.os.name != "nt":
        assert bridge.set_start_with_windows(True) == {"ok": False, "reason": "not-windows"}


def test_get_start_with_windows_reports_disabled_when_no_shortcut_exists(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    result = bridge.get_start_with_windows()
    assert result["ok"] is True
    assert result["enabled"] is False


def test_open_config_file_without_a_config_path_reports_a_clear_reason(tmp_path):
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, None)
    result = bridge.open_config_file()
    assert result == {"ok": False, "reason": "no-config-path"}


def test_open_config_file_fails_gracefully_rather_than_raising(tmp_path):
    config_path = tmp_path / "localpilot.toml"
    config_path.write_text("[model]\n", encoding="utf-8")
    window = FakeWindow()
    bridge = webview_app.WindowBridge(window, tmp_path, str(config_path))
    # No assertion on ok/not-ok here: whether this succeeds depends on
    # whether a default file-open handler exists in the environment running
    # the test, which varies. The contract under test is narrower — it must
    # not raise, and must always return a dict.
    result = bridge.open_config_file()
    assert isinstance(result, dict)
    assert "ok" in result


def test_bridge_payload_carries_no_more_than_the_frontend_needs():
    class FakeClient:
        base_url = "http://127.0.0.1:8765"
        token = "secret-token"

    payload = webview_app._bridge_payload(FakeClient(), "/some/config.toml")
    assert payload == {
        "baseUrl": "http://127.0.0.1:8765",
        "token": "secret-token",
        "hasConfigPath": True,
    }
    assert webview_app._bridge_payload(FakeClient(), None)["hasConfigPath"] is False


def test_initial_position_degrades_gracefully_without_a_display_backend():
    # In a headless environment (no GTK/Qt/WebView2 runtime), pywebview's own
    # screen enumeration raises rather than returning an empty list — this
    # asserts _initial_position()'s fallback actually catches that, using
    # the real webview.screens property rather than a mock, since the
    # failure mode itself is what's being verified.
    x, y = webview_app._initial_position(*webview_app.COMPACT_SIZE)
    assert (x is None and y is None) or (isinstance(x, int) and isinstance(y, int))


def test_position_on_screen_preserves_negative_virtual_desktop_origin():
    screen = FakeScreen(x=-1920, y=160, width=1920, height=1080)
    x, y = webview_app._position_on_screen(screen, *webview_app.COMPACT_SIZE)
    assert x == -168
    assert y == 1072
    assert x < 0


def test_position_on_screen_includes_positive_nonzero_origin():
    screen = FakeScreen(x=2560, y=-200, width=1920, height=1080)
    x, y = webview_app._position_on_screen(screen, *webview_app.COMPACT_SIZE)
    assert x == 4312
    assert y == 712


def test_initial_position_uses_supplied_screen_without_clamping():
    screen = FakeScreen(x=-1600, y=-900, width=1600, height=900)
    assert webview_app._initial_position(*webview_app.COMPACT_SIZE, screen=screen) == (-168, -168)


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
    assert not webview_app._should_detach_gui("localpilot", platform_name="posix")


def test_detached_launcher_uses_pythonw_and_no_stdio(tmp_path, monkeypatch):
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(webview_app, "_desktop_python_executable", lambda: pythonw)
    monkeypatch.setattr(webview_app.subprocess, "Popen", fake_popen)

    root = tmp_path / "repo"
    root.mkdir()
    assert webview_app._launch_detached(root, None) is True
    assert captured["argv"] == [
        str(pythonw),
        "-m",
        "localpilot.webview_app",
        "--root",
        str(root),
    ]
    assert captured["kwargs"]["cwd"] == root
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["close_fds"] is True


def test_build_parser_requires_root_and_defaults_config_to_none():
    parser = webview_app.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["--root", "/tmp/example"])
    assert args.root == "/tmp/example"
    assert args.config is None

    args = parser.parse_args(["--root", "/tmp/example", "--config", "/tmp/example/localpilot.toml"])
    assert args.config == "/tmp/example/localpilot.toml"


def test_startup_shortcut_path_is_under_the_startup_folder():
    path = webview_app._startup_shortcut_path()
    assert path.name == "LocalPilot.lnk"
    assert "Startup" in path.parts


def test_startup_shortcut_uses_static_powershell_and_environment_values(tmp_path, monkeypatch):
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
    assert str(target) not in script
    assert "$([danger])" not in script
    assert "LOCALPILOT_SHORTCUT_PATH" in script
    environment = captured["kwargs"]["env"]
    assert environment["LOCALPILOT_SHORTCUT_PATH"] == str(target.resolve())
    assert environment["LOCALPILOT_WORKING_DIRECTORY"] == str(root.resolve())
    assert environment["LOCALPILOT_ARGUMENTS"] == subprocess.list2cmdline(
        [
            "-m",
            "localpilot.webview_app",
            "--root",
            str(root.resolve()),
            "--config",
            str(config.resolve()),
        ]
    )
    assert captured["kwargs"]["check"] is True
    assert captured["kwargs"]["capture_output"] is True


def test_frontend_is_fully_local_and_uses_a_strict_csp():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    javascript = (webview_app.WEBVIEW_DIR / "app.js").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in index
    assert "http-equiv=\"Content-Security-Policy\"" in index
    assert "default-src 'none'" in index
    assert "script-src 'self'" in index
    assert "style-src 'self'" in index
    assert "'unsafe-inline'" not in index
    assert "'unsafe-eval'" not in index
    assert "style=" not in index
    assert ".innerHTML" not in javascript
    assert ".style." not in javascript
    assert "PID 18422" not in index
    assert "restarts: 0" not in index


def test_compact_frontend_has_no_card_chrome():
    css = (webview_app.WEBVIEW_DIR / "app.css").read_text(encoding="utf-8")
    start = css.index("  .dock {")
    end = css.index("\n  }", start)
    dock = css[start:end]
    assert "width: 128px; min-height: 128px;" in dock
    assert "background: transparent;" in dock
    assert "border: none;" in dock
    assert "box-shadow: none;" in dock
    assert ".preview-line { display: none; }" in css


def test_pywebview_dependency_is_constrained_to_the_verified_major_range():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "pywebview>=6.2.1,<7" in project["project"]["dependencies"]
