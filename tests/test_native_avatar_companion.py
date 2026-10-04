from __future__ import annotations

import queue
import subprocess
import threading
from types import SimpleNamespace
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
    source = native_avatar_companion.NativeAvatarCompanion._open_chat.__code__.co_names
    assert "_launch_chat" in source
    assert "close" not in source


def test_relaunch_focuses_live_chat_without_launching_or_closing_it(monkeypatch):
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)
    app._chat_launch_lock = threading.Lock()
    app._update_handoff_started = False
    app._stop = threading.Event()
    calls = []
    app._chat_is_alive = lambda: True
    app._focus_chat = lambda: calls.append("focus-existing")
    monkeypatch.setattr(native_avatar_companion, "_launch_chat", lambda *args, **kwargs: pytest.fail("Duplicate chat launch"))
    app.open_chat()
    assert calls == ["focus-existing"]


def test_final_update_admission_preserves_chat_opened_during_preparation():
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)
    app._chat_launch_lock = threading.Lock()
    app._update_handoff_started = False
    app._stop = threading.Event()
    app._chat_is_alive = lambda: True
    assert app._commit_update_launch(lambda: pytest.fail("An open composer must defer handoff")) is False
    assert app._update_handoff_started is False


def test_successful_update_admission_prevents_a_new_chat_from_opening():
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)
    app._chat_launch_lock = threading.Lock()
    app._update_handoff_started = False
    app._stop = threading.Event()
    app._chat_is_alive = lambda: False
    calls = []
    app._open_chat = lambda: calls.append("opened")
    assert app._commit_update_launch(lambda: True) is True
    app.open_chat()
    assert calls == []


def test_failed_updater_spawn_keeps_chat_available():
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)
    app._chat_launch_lock = threading.Lock()
    app._update_handoff_started = False
    app._stop = threading.Event()
    app._chat_is_alive = lambda: False
    calls = []
    app._open_chat = lambda: calls.append("opened")
    assert app._commit_update_launch(lambda: False) is False
    app.open_chat()
    assert calls == ["opened"]


def test_avatar_close_preserves_a_live_chat_process_and_draft():
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)

    class LiveChat:
        def poll(self):
            return None

        def terminate(self):
            pytest.fail("A desktop close must not terminate the chat process")

    app._chat_process = LiveChat()
    calls = []
    app._focus_chat = lambda: calls.append("focus-existing")
    app.close()
    assert calls == ["focus-existing"]


@pytest.mark.skipif(native_avatar_companion.os.name != "nt", reason="Windows exact-window focus")
@pytest.mark.parametrize("owner_reused", [False, True])
def test_focus_follows_verified_venv_python_child_and_excludes_browser_windows(tmp_path, monkeypatch, owner_reused):
    root = tmp_path.resolve()
    chat = SimpleNamespace(pid=101, create_time=lambda: 20.0, cmdline=lambda: ["pythonw.exe", "-m", "localpilot.webview_app", "--root", str(root)])
    browser = SimpleNamespace(pid=102, create_time=lambda: 30.0, cmdline=lambda: ["msedgewebview2.exe", "--root", str(root)])
    owner = SimpleNamespace(pid=100, create_time=lambda: 11.0 if owner_reused else 10.0, children=lambda **kwargs: [chat, browser])
    processes = {100: owner, 101: chat, 102: browser}
    calls = []

    def window_pid(hwnd, result):
        result._obj.value = {1001: 101, 1002: 102}[hwnd.value]

    def enumerate_windows(callback, parameter):
        for hwnd in [1002, 1001]:
            if not callback(hwnd, parameter):
                break

    def window_title(hwnd, target, size):
        target.value = "LocalPilot"

    user32 = SimpleNamespace(
        EnumWindows=enumerate_windows, GetWindowThreadProcessId=window_pid,
        GetWindowTextW=window_title,
        IsWindowVisible=lambda hwnd: False,
        ShowWindow=lambda hwnd, command: calls.append(("restore", hwnd.value)),
        SetForegroundWindow=lambda hwnd: calls.append(("focus", hwnd.value)),
    )
    monkeypatch.setattr(native_avatar_companion.psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(native_avatar_companion.ctypes, "windll", SimpleNamespace(user32=user32))
    app = native_avatar_companion.NativeAvatarCompanion.__new__(native_avatar_companion.NativeAvatarCompanion)
    app._chat_process = SimpleNamespace(pid=100)
    app._chat_process_created_at = 10.0
    app.project_root = root
    app._focus_chat()
    assert calls == ([] if owner_reused else [("restore", 1001), ("focus", 1001)])


def test_main_acquires_owner_before_broker_and_draws_avatar_before_initial_chat(tmp_path, monkeypatch):
    calls = []

    class FakeOwner:
        def __init__(self, *args):
            calls.append("owner-created")

        def acquire(self):
            calls.append("owner-acquired")
            return True

        def close(self):
            calls.append("owner-released")

    class FakeRoot:
        def after(self, delay, callback):
            calls.append("chat-scheduled")

    class FakeAvatar:
        def __init__(self, *args, **kwargs):
            calls.append("avatar-created")
            self.root = FakeRoot()

        def open_chat(self):
            pass

        def run(self):
            calls.append("avatar-running")

    monkeypatch.setattr(native_avatar_companion, "DesktopInstance", FakeOwner)
    monkeypatch.setattr(native_avatar_companion, "assert_no_legacy_desktop", lambda root, **kwargs: None)
    monkeypatch.setattr(native_avatar_companion._avatar._legacy, "load_config", lambda path: object())
    monkeypatch.setattr(native_avatar_companion._avatar._legacy, "ensure_broker", lambda *args, **kwargs: calls.append("broker-ready"))
    monkeypatch.setattr(native_avatar_companion, "NativeAvatarCompanion", FakeAvatar)
    native_avatar_companion.main(tmp_path, open_chat=True)
    assert calls == ["owner-created", "owner-acquired", "broker-ready", "avatar-created", "chat-scheduled", "avatar-running", "owner-released"]


def test_repeated_owner_launch_returns_before_broker_or_gui_creation(tmp_path, monkeypatch):
    calls = []

    class FakeOwner:
        def __init__(self, *args):
            pass

        def acquire(self):
            return False

        def request_activation(self):
            calls.append("activate-existing")
            return True

    monkeypatch.setattr(native_avatar_companion, "DesktopInstance", FakeOwner)
    monkeypatch.setattr(native_avatar_companion._avatar._legacy, "ensure_broker", lambda *args, **kwargs: pytest.fail("Duplicate broker startup"))
    monkeypatch.setattr(native_avatar_companion, "NativeAvatarCompanion", lambda *args, **kwargs: pytest.fail("Duplicate avatar startup"))
    native_avatar_companion.main(tmp_path, open_chat=True)
    assert calls == ["activate-existing"]


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
