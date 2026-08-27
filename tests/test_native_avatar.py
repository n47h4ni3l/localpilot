from __future__ import annotations

import subprocess
from pathlib import Path

from localpilot import native_avatar
from localpilot.desktop_state import DesktopUIState


def test_saved_avatar_position_preserves_negative_virtual_coordinates():
    state = {"avatar_x": -640, "avatar_y": 120, "always_on_top": True}
    assert native_avatar._initial_avatar_position(state, (0, 0, 1920, 1080)) == (-640, 120)


def test_unsaved_avatar_anchors_inside_primary_work_area():
    state = {"avatar_x": None, "avatar_y": None, "always_on_top": True}
    x, y = native_avatar._initial_avatar_position(state, (100, -200, 2020, 880))
    assert x == 2020 - native_avatar.AVATAR_SIZE - native_avatar.EDGE_INSET
    assert y == 880 - native_avatar.AVATAR_SIZE - native_avatar.EDGE_INSET


def test_chat_keeps_bottom_right_anchor_to_avatar():
    x, y = native_avatar._chat_position_from_avatar(1700, 820)
    width, height = native_avatar.EXPANDED_SIZE
    assert x + width == 1700 + native_avatar.AVATAR_SIZE
    assert y + height == 820 + native_avatar.AVATAR_SIZE


def test_launch_webview_uses_detached_gui_process_and_absolute_coordinates(tmp_path, monkeypatch):
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(native_avatar, "_desktop_python_executable", lambda: pythonw)

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(native_avatar.subprocess, "Popen", fake_popen)
    root = tmp_path / "repo"
    root.mkdir()

    assert native_avatar._launch_webview(root, None, x=-500, y=240) is True
    assert captured["argv"] == [
        str(pythonw),
        "-m",
        "localpilot.webview_app",
        "--root",
        str(root),
        "--x",
        "-500",
        "--y",
        "240",
    ]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL
    assert captured["kwargs"]["shell"] is False


def test_desktop_ui_state_round_trips_avatar_position_and_on_top(tmp_path):
    state = DesktopUIState(tmp_path)
    assert state.read()["always_on_top"] is True
    state.update(avatar_x=-900, avatar_y=50, always_on_top=False)
    assert state.read() == {
        "avatar_x": -900,
        "avatar_y": 50,
        "always_on_top": False,
    }


def test_native_avatar_uses_windows_transparent_color_not_webview_transparency():
    source = Path(native_avatar.__file__).read_text(encoding="utf-8")
    assert 'wm_attributes("-transparentcolor", _TRANSPARENT_KEY)' in source
    assert "overrideredirect(True)" in source
    assert "webview.create_window" not in source
