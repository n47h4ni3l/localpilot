from __future__ import annotations

import subprocess

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
