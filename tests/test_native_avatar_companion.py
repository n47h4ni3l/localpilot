from __future__ import annotations

import queue
import subprocess
import threading
import pytest

from localpilot import native_avatar_companion, webview_app


def test_chat_sits_left_of_default_bottom_right_avatar():
    work_area = (0, 0, 1920, 1040)
    avatar_x = 1920 - 128
    avatar_y = 1040 - 128
    x, y = native_avatar_companion._chat_position_from_avatar(
        avatar_x,
        avatar_y,
        work_area,
    )
    assert x + webview_app.EXPANDED_SIZE[0] + native_avatar_companion.CHAT_GAP == avatar_x
    assert y + webview_app.EXPANDED_SIZE[1] == avatar_y + 128


def test_chat_moves_to_right_side_when_avatar_is_near_left_edge():
    work_area = (0, 0, 1920, 1040)
    x, y = native_avatar_companion._chat_position_from_avatar(0, 400, work_area)
    assert x == 128 + native_avatar_companion.CHAT_GAP
    assert 0 <= y <= 1040 - webview_app.EXPANDED_SIZE[1]


@pytest.mark.parametrize("scale", [1, 1.25, 1.5, 1.75, 2, 3])
def test_chat_handoff_uses_physical_size_on_scaled_and_negative_origin_displays(scale):
    area = (-4000, -200, 0, 2500)
    x, y = native_avatar_companion._chat_position_from_avatar(-200, 2200, area, scale=scale)
    assert x + round(500 * scale) + native_avatar_companion.CHAT_GAP == -200
    assert y + round(640 * scale) == 2200 + 128


def test_launch_chat_marks_webview_as_existing_avatar_companion(tmp_path, monkeypatch):
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(native_avatar_companion, "_desktop_python_executable", lambda: pythonw)

    class FakeProcess:
        pass

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(native_avatar_companion.subprocess, "Popen", fake_popen)
    root = tmp_path / "repo"
    root.mkdir()
    process = native_avatar_companion._launch_chat(root, None, x=111, y=222)

    assert isinstance(process, FakeProcess)
    assert captured["argv"] == [
        str(pythonw),
        "-m",
        "localpilot.webview_app",
        "--root",
        str(root),
        "--x",
        "111",
        "--y",
        "222",
        "--companion",
        "--physical-position",
    ]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL


def test_persistent_avatar_class_overrides_old_handoff_close():
    assert "_finish_chat_handoff" not in native_avatar_companion.NativeAvatarCompanion.__dict__
    assert "open_chat" in native_avatar_companion.NativeAvatarCompanion.__dict__
    source = native_avatar_companion.NativeAvatarCompanion.open_chat.__code__.co_names
    assert "_launch_chat" in source
    assert "close" not in source


def test_live_chat_presentation_state_drives_the_one_visible_native_avatar():
    class FakeProcess:
        def poll(self):
            return None

    class FakeStateStore:
        def read(self):
            return {"companion_state": "speaking"}

    app = native_avatar_companion.NativeAvatarCompanion.__new__(
        native_avatar_companion.NativeAvatarCompanion
    )
    app._events = queue.Queue()
    app._events.put("working")
    app._broker_runtime_state = "idle"
    app._chat_process = FakeProcess()
    app.state_store = FakeStateStore()
    app.runtime_state = "idle"
    app._stop = threading.Event()
    app._stop.set()
    app._draw = lambda: None

    app._drain_events()

    assert app._broker_runtime_state == "working"
    assert app.runtime_state == "speaking"


def test_native_avatar_falls_back_to_broker_state_when_chat_is_closed():
    class FakeProcess:
        def poll(self):
            return 0

    class FakeStateStore:
        def read(self):
            return {"companion_state": "speaking"}

    app = native_avatar_companion.NativeAvatarCompanion.__new__(
        native_avatar_companion.NativeAvatarCompanion
    )
    app._events = queue.Queue()
    app._events.put("thinking")
    app._broker_runtime_state = "idle"
    app._chat_process = FakeProcess()
    app.state_store = FakeStateStore()
    app.runtime_state = "idle"
    app._stop = threading.Event()
    app._stop.set()
    app._draw = lambda: None

    app._drain_events()

    assert app._broker_runtime_state == "thinking"
    assert app.runtime_state == "thinking"
