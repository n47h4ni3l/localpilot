from __future__ import annotations

import ctypes
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from localpilot import desktop_updater as updater


class Process:
    def __init__(self, root, role, pid, events, *, config=None):
        self.pid = pid
        self.role = role
        self.root = root
        self.config = config or root / "localpilot.toml"
        self.events = events
        self.alive = True
        self.created_at = float(pid)
        self.child_processes = []
        self.stubborn = False

    def cmdline(self):
        return ["pythonw.exe", "-m", "localpilot." + self.role, "--root", str(self.root), "--config", str(self.config)]

    def cwd(self):
        return str(self.root)

    def exe(self):
        return str(self.root / "pythonw.exe")

    def create_time(self):
        return self.created_at

    def children(self):
        return self.child_processes

    def wait(self, timeout):
        self.events.append("exit:" + self.role)
        if self.stubborn:
            raise psutil.TimeoutExpired(timeout, self.pid)
        self.alive = False
        return 0

    def terminate(self):
        raise AssertionError("Graceful update must not terminate a process")

    def kill(self):
        raise AssertionError("Graceful update must not kill a process")


def setup(monkeypatch, tmp_path, roles, *, busy=False):
    root = tmp_path / "localpilot"
    root.mkdir()
    (root / "localpilot.toml").write_text('[selfdev]\nenabled = false\n', encoding="utf-8")
    events = []
    processes = [Process(root, role, 100 + index, events) for index, role in enumerate(roles)]
    by_pid = {process.pid: process for process in processes}

    def lookup(pid):
        process = by_pid[pid]
        if not process.alive:
            raise psutil.NoSuchProcess(pid)
        return process

    monkeypatch.setattr(updater.psutil, "process_iter", lambda: iter(processes))
    monkeypatch.setattr(updater.psutil, "Process", lookup)

    class Client:
        def __init__(self, *args):
            pass

        def request(self, method, route, payload=None, **kwargs):
            assert kwargs.get("authenticated", True)
            broker = next(process for process in processes if process.role == "broker")
            if method == "GET":
                events.append("broker-status")
                return {"pid": broker.pid, "pending_requests": int(busy)}
            if busy:
                raise urllib.error.HTTPError("http://localhost", 409, "active response", None, None)
            events.append("broker-shutdown")
            return {"pid": broker.pid, "status": "stopping"}

    def close(identity):
        events.append("close:" + identity.role)

    monkeypatch.setattr(updater, "BrokerClient", Client)
    monkeypatch.setattr(updater, "_request_window_close", close)
    workers = [process for process in processes if process.role == "background_worker"]
    data_dir = root / "localpilot-data"
    data_dir.mkdir()
    if workers:
        (data_dir / "background-worker.pid").write_text(json.dumps({"pid": workers[0].pid, "root": str(root)}), encoding="utf-8")
    return root, data_dir, processes, events


def test_exact_root_ownership_rejects_prefix_and_lookalike_module(tmp_path):
    root = tmp_path / "localpilot"
    other = tmp_path / "localpilot-other"
    process = Process(other, "background_worker", 123, [])
    assert not updater._belongs_to_localpilot(process, root)
    process = Process(root, "broker_extra", 123, [])
    assert not updater._belongs_to_localpilot(process, root)
    process = Process(root, "broker", 123, [])
    assert updater._belongs_to_localpilot(process, root)


def test_explicit_other_root_cannot_be_overridden_by_current_directory(tmp_path):
    root = tmp_path / "localpilot"
    process = Process(root / "other", "broker", 123, [])
    process.cwd = lambda: str(root)
    assert not updater._belongs_to_localpilot(process, root)


@pytest.mark.parametrize(("arguments", "role"), [
    (["--config", "desktop", "doctor"], None),
    (["--config=desktop", "doctor"], None),
    (["--config=custom.toml", "desktop"], "avatar"),
    (["--config", "custom.toml", "desktop"], "avatar"),
    (["--config=custom.toml", "desktop", "--tkinter"], "chat"),
    (["--config", "custom.toml"], None),
])
def test_cli_process_role_uses_the_subcommand_after_configuration(arguments, role):
    assert updater._process_role(["pythonw.exe", "-m", "localpilot.cli", *arguments]) == role
    assert updater._process_role(["localpilot.exe", *arguments]) == role


def test_running_worker_is_not_evidence_that_the_desktop_launched(monkeypatch, tmp_path):
    worker = Process(tmp_path, "background_worker", 123, [])
    monkeypatch.setattr(updater.psutil, "process_iter", lambda: iter([worker]))
    assert updater._localpilot_process_running(tmp_path) is False
    broker = Process(tmp_path, "broker", 321, [])
    monkeypatch.setattr(updater.psutil, "process_iter", lambda: iter([broker]))
    assert updater._localpilot_process_running(tmp_path) is False
    chat = Process(tmp_path, "webview_app", 456, [])
    monkeypatch.setattr(updater.psutil, "process_iter", lambda: iter([worker, chat]))
    assert updater._localpilot_process_running(tmp_path) is True


@pytest.mark.parametrize("owner_pid_as_text", [False, True])
def test_stop_is_cooperative_and_closes_chat_before_avatar(monkeypatch, tmp_path, owner_pid_as_text):
    root, data_dir, processes, events = setup(
        monkeypatch, tmp_path, ["broker", "runtime_worker", "native_avatar_companion", "webview_app", "background_worker"]
    )
    if owner_pid_as_text:
        owner_path = data_dir / "background-worker.pid"
        owner = json.loads(owner_path.read_text())
        owner["pid"] = str(owner["pid"])
        owner_path.write_text(json.dumps(owner))

    updater._stop_localpilot_processes(root)

    assert events[:3] == ["broker-status", "close:chat", "exit:webview_app"]
    assert events.index("exit:webview_app") < events.index("close:avatar")
    assert events.index("exit:native_avatar_companion") < events.index("broker-shutdown")
    request = json.loads((data_dir / "background-worker.stop").read_text())
    assert request["target_pid"] == processes[-1].pid
    assert all(not process.alive for process in processes)


def test_active_broker_refusal_preserves_ui_and_worker(monkeypatch, tmp_path):
    root, data_dir, processes, events = setup(
        monkeypatch, tmp_path, ["broker", "webview_app", "background_worker"], busy=True
    )

    with pytest.raises(RuntimeError, match="finish the response"):
        updater._stop_localpilot_processes(root)

    assert events == ["broker-status"]
    assert all(process.alive for process in processes)
    assert not (data_dir / "background-worker.stop").exists()


def test_config_mismatch_refuses_before_any_shutdown(monkeypatch, tmp_path):
    root, _data_dir, processes, events = setup(monkeypatch, tmp_path, ["broker"])
    processes[0].config = root / "other.toml"

    with pytest.raises(RuntimeError, match="another configuration"):
        updater._stop_localpilot_processes(root)

    assert events == []


def test_skipped_worker_is_not_requested_to_stop(monkeypatch, tmp_path):
    root, data_dir, processes, events = setup(monkeypatch, tmp_path, ["background_worker"])

    updater._stop_localpilot_processes(root, include_background_worker=False)

    assert events == []
    assert processes[0].alive
    assert not (data_dir / "background-worker.stop").exists()


def test_missing_worker_owner_refuses_before_closing_ui(monkeypatch, tmp_path):
    root, data_dir, _processes, events = setup(monkeypatch, tmp_path, ["broker", "webview_app", "background_worker"])
    (data_dir / "background-worker.pid").unlink()

    with pytest.raises(RuntimeError, match="PID ownership is unavailable"):
        updater._stop_localpilot_processes(root)

    assert events == []


def test_window_veto_or_timeout_never_force_terminates_process(monkeypatch, tmp_path):
    root, _data_dir, processes, events = setup(monkeypatch, tmp_path, ["webview_app", "native_avatar_companion"])
    processes[0].stubborn = True

    with pytest.raises(RuntimeError, match="no process was force-terminated"):
        updater._stop_localpilot_processes(root, timeout=.1)

    assert events == ["close:chat", "exit:webview_app"]
    assert all(process.alive for process in processes)


def test_reused_pid_is_not_a_current_owned_instance(monkeypatch, tmp_path):
    process = Process(tmp_path, "webview_app", 123, [])
    identity = updater._ProcessIdentity(process, "chat", process.create_time(), process.exe(), tuple(process.cmdline()))
    process.created_at += 1
    monkeypatch.setattr(updater.psutil, "Process", lambda pid: process)

    assert updater._current_process(identity) is False


def test_concurrent_desktop_after_shutdown_snapshot_refuses_source_update(monkeypatch, tmp_path):
    root, _data_dir, processes, events = setup(monkeypatch, tmp_path, ["webview_app"])
    later = Process(root, "native_avatar_companion", 500, events)
    snapshots = iter([processes, [later]])
    monkeypatch.setattr(updater.psutil, "process_iter", lambda: iter(next(snapshots)))
    original_lookup = updater.psutil.Process
    monkeypatch.setattr(updater.psutil, "Process", lambda pid: later if pid == later.pid else original_lookup(pid))

    with pytest.raises(RuntimeError, match="appeared during graceful shutdown"):
        updater._stop_localpilot_processes(root)

    assert later.alive is True
    assert events == ["close:chat", "exit:webview_app"]


def test_late_broker_refusal_does_not_stop_worker_or_force_kill_runtime(monkeypatch, tmp_path):
    root, data_dir, processes, events = setup(monkeypatch, tmp_path, ["broker", "runtime_worker", "background_worker"])

    class Client:
        def __init__(self, *args):
            pass

        def request(self, method, route, payload=None, **kwargs):
            if method == "GET":
                return {"pid": processes[0].pid, "pending_requests": 0}
            raise urllib.error.HTTPError("http://localhost", 409, "A new response is active", None, None)

    monkeypatch.setattr(updater, "BrokerClient", Client)

    with pytest.raises(RuntimeError, match="broker did not stop safely"):
        updater._stop_localpilot_processes(root)

    assert all(process.alive for process in processes)
    assert not (data_dir / "background-worker.stop").exists()
    assert events == []


def test_helper_is_waited_for_through_verified_runtime_child_identity(monkeypatch, tmp_path):
    root, _data_dir, processes, events = setup(monkeypatch, tmp_path, ["broker", "runtime_worker"])
    helper = Process(root, "hardware_helper", 500, events)
    helper.exe = lambda: str(root / "localpilot" / "_hardware" / "win-x64" / "LocalPilot.SystemSense.HardwareProvider.exe")
    processes[1].child_processes = [helper]
    original_lookup = updater.psutil.Process
    monkeypatch.setattr(updater.psutil, "Process", lambda pid: helper if pid == helper.pid else original_lookup(pid))

    updater._stop_localpilot_processes(root)

    assert events[-1] == "exit:hardware_helper"
    assert helper.alive is False


@pytest.mark.parametrize(("role", "caption", "window_class", "visible", "should_close", "window_reassigned"), [
    ("chat", "LocalPilot", "WindowsForms10.Window.8.app.0.example", False, True, False),
    ("chat", "LocalPilot", "WindowsForms10.Window.8.app.0.example", True, True, False),
    ("chat", "Nestra · Desktop", "TkTopLevel", False, True, False),
    ("avatar", "Nestra", "TkTopLevel", True, True, False),
    ("avatar", "Nestra", "TkTopLevel", False, False, False),
    ("chat", "LocalPilot", "WindowsForms10.Window.8.app.0.example", False, False, True),
])
def test_wm_close_targets_only_verified_desktop_windows_even_when_chat_is_hidden(
    monkeypatch, tmp_path, role, caption, window_class, visible, should_close, window_reassigned,
):
    (tmp_path / "localpilot.toml").write_text('[agent]\nname = "Nestra"\n', encoding="utf-8")
    configuration = updater.load_config(tmp_path / "localpilot.toml")
    # The Windows API branch is mocked on every platform; keep pathlib's
    # platform selection outside the temporary os.name override below.
    monkeypatch.setattr(updater, "load_config", lambda path: configuration)
    monkeypatch.setattr(updater, "_process_path", lambda value, cwd: tmp_path / "localpilot.toml")
    process = Process(tmp_path, "webview_app" if role == "chat" else "native_avatar_companion", 123, [])
    identity = updater._ProcessIdentity(process, role, process.create_time(), process.exe(), tuple(process.cmdline()))
    monkeypatch.setattr(updater.psutil, "Process", lambda pid: process)
    posted = []
    windows = {
        1: (123, caption, window_class, visible),
        2: (456, caption, window_class, True),
        3: (123, caption, "MS_WebView2Helper", True),
        4: (123, "Internal helper", window_class, True),
        5: (123, "", "TkChild", False),
    }
    if window_reassigned:
        current_process = updater._current_process
        checks = []

        def current(identity):
            checks.append(identity)
            if len(checks) == 2:
                windows[1] = (456, caption, window_class, visible)
            return current_process(identity)

        monkeypatch.setattr(updater, "_current_process", current)

    def owner(hwnd, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulong)).contents.value = windows[hwnd.value][0]

    def title(hwnd, buffer, length):
        buffer.value = windows[hwnd.value][1]

    def native_class(hwnd, buffer, length):
        buffer.value = windows[hwnd.value][2]

    def enumerate_windows(callback, parameter):
        for hwnd in windows:
            callback(hwnd, parameter)

    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=SimpleNamespace(
        GetWindowThreadProcessId=owner,
        GetWindowTextW=title,
        GetClassNameW=native_class,
        IsWindowVisible=lambda hwnd: windows[hwnd.value][3],
        PostMessageW=lambda hwnd, message, wparam, lparam: posted.append((hwnd.value, message)),
        EnumWindows=enumerate_windows,
    )), raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(updater.os, "name", "nt")
        updater._request_window_close(identity)

    assert posted == ([(1, 0x0010)] if should_close else [])
