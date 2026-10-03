from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import psutil
import pytest

import localpilot.desktop as desktop
from localpilot.config import Config


class FakeClient:
    def __init__(self, *, healthy=True, elevated=False):
        self.available = healthy
        self.status = {"pid": 123, "elevated": elevated}
        self.calls = []
        self.shutdown_error = None

    def healthy(self):
        return self.available

    def request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload, kwargs))
        if path == "/v1/broker/status":
            return dict(self.status)
        if self.shutdown_error:
            raise self.shutdown_error
        return {"status": "stopping", "pid": self.status["pid"]}


def _setup(monkeypatch, *, desktop_elevated=True, broker_elevated=False, healthy=True):
    client = FakeClient(healthy=healthy, elevated=broker_elevated)
    operations = []
    monkeypatch.setattr(desktop, "BrokerClient", lambda *args: client)
    monkeypatch.setattr(desktop, "current_process_elevated", lambda: desktop_elevated)

    class Process:
        def __init__(self, pid):
            assert pid == 123

        def wait(self, timeout):
            operations.append("old-broker-exited")
            client.available = False

    def launch(argv, **kwargs):
        operations.append("new-broker-started")
        client.available = True
        client.status.update(pid=456, elevated=True)
        return SimpleNamespace(pid=456)

    monkeypatch.setattr(desktop.psutil, "Process", Process)
    monkeypatch.setattr(desktop.subprocess, "Popen", launch)
    return client, operations


def test_elevated_desktop_replaces_unelevated_broker_only_after_exact_pid_exits(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch)

    assert desktop.ensure_broker(tmp_path, Config()) is client

    assert operations == ["old-broker-exited", "new-broker-started"]
    assert [(method, path) for method, path, *_ in client.calls] == [
        ("GET", "/v1/broker/status"),
        ("POST", "/v1/broker/shutdown"),
        ("GET", "/v1/broker/status"),
    ]
    assert all(call[3].get("authenticated", True) for call in client.calls)


@pytest.mark.parametrize("desktop_elevated", [False, None])
def test_normal_and_nonwindows_desktops_keep_existing_broker(monkeypatch, tmp_path, desktop_elevated):
    client, operations = _setup(monkeypatch, desktop_elevated=desktop_elevated)

    assert desktop.ensure_broker(tmp_path, Config()) is client

    assert operations == []
    assert client.calls == []


def test_elevated_desktop_reuses_existing_elevated_broker(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch, broker_elevated=True)

    assert desktop.ensure_broker(tmp_path, Config()) is client

    assert operations == []
    assert len(client.calls) == 1


def test_elevated_desktop_does_not_interrupt_active_response(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch)
    client.shutdown_error = urllib.error.HTTPError("http://localhost", 409, "busy", None, None)

    with pytest.raises(RuntimeError, match="Let the response finish"):
        desktop.ensure_broker(tmp_path, Config())

    assert client.available is True
    assert operations == []


def test_elevated_desktop_cannot_silently_attach_to_legacy_or_unknown_broker(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch, broker_elevated=None)

    with pytest.raises(RuntimeError, match="administrator state is unavailable"):
        desktop.ensure_broker(tmp_path, Config())

    assert operations == []
    assert len(client.calls) == 1


def test_elevated_desktop_does_not_launch_replacement_when_old_process_will_not_exit(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch)

    class Process:
        def __init__(self, pid):
            assert pid == 123

        def wait(self, timeout):
            raise psutil.TimeoutExpired(timeout, pid=123)

    monkeypatch.setattr(desktop.psutil, "Process", Process)

    with pytest.raises(RuntimeError, match="did not stop safely"):
        desktop.ensure_broker(tmp_path, Config())

    assert operations == []


def test_new_broker_must_inherit_elevated_token(monkeypatch, tmp_path):
    client, operations = _setup(monkeypatch, healthy=False)

    def launch(*args, **kwargs):
        operations.append("new-broker-started")
        client.available = True
        client.status["elevated"] = False

    monkeypatch.setattr(desktop.subprocess, "Popen", launch)

    with pytest.raises(RuntimeError, match="did not inherit administrator rights"):
        desktop.ensure_broker(tmp_path, Config())

    assert operations == ["new-broker-started"]
