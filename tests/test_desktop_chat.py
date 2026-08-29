import io
import http.client
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from localpilot.broker import BrokerApp, BrokerHTTPServer, load_or_create_broker_token
from localpilot.chat_store import ChatStore
from localpilot.config import Config, load_config
from localpilot.agent import LocalPilotAgent
from localpilot.runtime_supervisor import RuntimeSupervisor
from localpilot.runtime_worker import RuntimeWorker


class _FakeRuntime:
    def __init__(self):
        self.sent = []
        self.running = True
        self.pid = 321
        self.restart_calls = []

    def send(self, message):
        self.sent.append(message)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def restart(self, **kwargs):
        self.restart_calls.append(kwargs)
        self.running = False
        return True


def _broker(tmp_path):
    config = Config()
    app = BrokerApp(tmp_path, config)
    app.runtime = _FakeRuntime()
    return app


def test_broker_cors_allows_only_loopback_webview_origins(tmp_path):
    app = _broker(tmp_path)
    server = BrokerHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        origin = "http://127.0.0.1:43123"
        connection.request(
            "OPTIONS",
            "/v1/sessions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 204
        assert response.getheader("Access-Control-Allow-Origin") == origin
        assert response.getheader("Access-Control-Allow-Headers") == "Authorization, Content-Type"

        connection.request("GET", "/health", headers={"Origin": origin})
        response = connection.getresponse()
        response.read()
        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == origin

        connection.request(
            "GET",
            "/v1/sessions",
            headers={"Origin": origin, "Authorization": f"Bearer {app.token}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert payload == {"sessions": []}
        assert response.getheader("Access-Control-Allow-Origin") == origin

        connection.request(
            "OPTIONS",
            "/v1/sessions",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 403
        assert response.getheader("Access-Control-Allow-Origin") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_broker_exposes_authenticated_read_only_systemsense_summary(tmp_path):
    app = _broker(tmp_path)
    calls = []

    def summary(*, collect_if_missing=True):
        calls.append(collect_if_missing)
        return {
            "enabled": True,
            "captured_at": "2026-08-28T04:00:00+00:00",
            "system_health": "good",
            "cpu_percent": 23.0,
        }

    app.systemsense = SimpleNamespace(summary=summary)
    server = BrokerHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/v1/systemsense/summary")
        response = connection.getresponse()
        response.read()
        assert response.status == 401
        assert calls == []

        connection.request(
            "GET",
            "/v1/systemsense/summary",
            headers={"Authorization": f"Bearer {app.token}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert payload["summary"]["system_health"] == "good"
        assert payload["summary"]["cpu_percent"] == 23.0
        assert calls == [False]

        connection.request(
            "POST",
            "/v1/systemsense/summary",
            headers={"Authorization": f"Bearer {app.token}"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 404
        assert calls == [False]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_broker_systemsense_summary_fails_soft_without_leaking_details(tmp_path):
    app = _broker(tmp_path)

    def fail_summary(*, collect_if_missing=True):
        raise sqlite3.OperationalError("private database path")

    app.systemsense = SimpleNamespace(summary=fail_summary)
    result = app.systemsense_summary()

    assert result == {
        "enabled": True,
        "system_health": "unknown",
        "available": False,
        "error_type": "OperationalError",
    }


def test_chat_history_is_unicode_safe_persistent_and_separate_from_learning(tmp_path):
    path = tmp_path / "localpilot-data" / "chat.sqlite3"
    store = ChatStore(path)
    session = store.create_session("Emoji lab 🤖")
    store.add_message(session["id"], "user", "Hello — snowman ☃️")
    store.add_message(session["id"], "assistant", "Ready 🚀")
    store.append_event("runtime.state", {"state": "success", "glyph": "✅"}, session_id=session["id"])

    reopened = ChatStore(path)

    assert reopened.session(session["id"])["title"] == "Emoji lab 🤖"
    assert [message["content"] for message in reopened.messages(session["id"])] == [
        "Hello — snowman ☃️",
        "Ready 🚀",
    ]
    assert reopened.events_after(0, session_id=session["id"])[0]["payload"]["glyph"] == "✅"
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "knowledge_facts" not in tables
    assert "chat_messages" in tables


def test_broker_replays_completed_session_after_runtime_replacement(tmp_path):
    app = _broker(tmp_path)
    session = app.store.create_session()
    first = app.submit(session["id"], "Remember this emoji: 🧭")
    command = app.runtime.sent[-1]
    assert command["history"] == []

    app._on_runtime_message(
        {
            "kind": "event",
            "request_id": first["request_id"],
            "session_id": session["id"],
            "type": "assistant.delta",
            "payload": {"delta": "Noted "},
        }
    )
    app._on_runtime_message(
        {
            "kind": "event",
            "request_id": first["request_id"],
            "session_id": session["id"],
            "type": "assistant.delta",
            "payload": {"delta": "🧭"},
        }
    )
    app._on_runtime_message(
        {
            "kind": "result",
            "request_id": first["request_id"],
            "session_id": session["id"],
            "answer": "Noted 🧭",
        }
    )

    replacement = _FakeRuntime()
    app.runtime = replacement
    app.submit(session["id"], "What did I ask you to remember?")

    assert replacement.sent[0]["history"] == [
        {"role": "user", "content": "Remember this emoji: 🧭"},
        {"role": "assistant", "content": "Noted 🧭"},
    ]
    assert "knowledge" not in json.dumps(replacement.sent[0]).lower()


def test_runtime_exit_preserves_ui_history_and_marks_only_inflight_answer_failed(tmp_path):
    app = _broker(tmp_path)
    session = app.store.create_session()
    app.store.add_message(session["id"], "user", "Earlier")
    app.store.add_message(session["id"], "assistant", "Completed", status="complete")
    app.submit(session["id"], "Interrupted")

    app._on_runtime_message(
        {
            "kind": "supervisor",
            "type": "runtime.exited",
            "payload": {"returncode": 7, "restart_attempt": 1},
        }
    )

    messages = app.store.messages(session["id"])
    assert [message["status"] for message in messages] == ["complete", "complete", "complete", "error"]
    assert "restarted before this answer completed" in messages[-1]["content"]
    assert any(
        event["type"] == "runtime.state" and event["payload"].get("state") == "restarting"
        for event in app.store.events_after(0)
    )


def test_broker_startup_closes_abandoned_streaming_records(tmp_path):
    store = ChatStore(tmp_path / "localpilot-data" / "chat.sqlite3")
    session = store.create_session()
    message = store.add_message(session["id"], "assistant", "", status="streaming")

    app = BrokerApp(tmp_path, Config())

    recovered = app.store.message(message["id"])
    assert recovered["status"] == "error"
    assert "broker restarted" in recovered["content"]
    assert any(
        event["type"] == "message.failed"
        and event["payload"].get("reason") == "broker_restart_recovery"
        for event in app.store.events_after(0)
    )


def test_broker_request_timeout_is_soft_and_late_result_completes_without_pid_change(tmp_path):
    config = Config()
    config.desktop.request_timeout_seconds = 60
    app = BrokerApp(tmp_path, config)
    app.runtime = _FakeRuntime()
    session = app.store.create_session()
    submitted = app.submit(session["id"], "A bounded request")
    original_pid = app.runtime.pid

    app._expire_request(submitted["request_id"])

    delayed = app.store.message(submitted["assistant"]["id"])
    assert delayed["status"] == "streaming"
    assert app.runtime.running is True
    assert app.runtime.pid == original_pid
    assert app.runtime.restart_calls == []
    assert any(
        event["payload"].get("reason") == "request_timeout"
        for event in app.store.events_after(0)
        if event["type"] == "message.delayed"
    )

    app._on_runtime_message(
        {
            "kind": "result",
            "request_id": submitted["request_id"],
            "session_id": session["id"],
            "answer": "Completed after the soft boundary",
        }
    )

    completed = app.store.message(submitted["assistant"]["id"])
    assert completed["status"] == "complete"
    assert completed["content"] == "Completed after the soft boundary"


def test_runtime_worker_uses_agent_once_and_streams_structured_visible_deltas(tmp_path, monkeypatch):
    observed = []

    class FakeAgent:
        instances = 0

        def __init__(self, config, root, *, event_sink):
            FakeAgent.instances += 1
            self.messages = [{"role": "system", "content": "safe"}]
            self.event_sink = event_sink

        def ask(self, prompt):
            self.event_sink({"type": "tool.started", "payload": {"tool": "get_system_summary"}})
            return "Unicode answer 🤖"

    monkeypatch.setattr("localpilot.runtime_worker.LocalPilotAgent", FakeAgent)
    worker = RuntimeWorker(tmp_path)
    worker._write = observed.append
    command = {
        "kind": "ask",
        "request_id": "request-1",
        "session_id": "session-1",
        "prompt": "Hello",
        "history": [{"role": "user", "content": "Prior"}, {"role": "assistant", "content": "Turn"}],
    }

    worker.handle(command)
    worker.handle({**command, "request_id": "request-2", "prompt": "Again", "history": []})

    assert FakeAgent.instances == 1
    assert any(item.get("type") == "tool.started" for item in observed)
    assert "".join(
        item["payload"]["delta"] for item in observed if item.get("type") == "assistant.delta"
    ) == "Unicode answer 🤖Unicode answer 🤖"
    assert not any("thinking" in json.dumps(item).lower() and "private" in json.dumps(item).lower() for item in observed)


def test_supervisor_launches_worker_with_argv_shell_false_and_utf8(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 321
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

        @staticmethod
        def poll():
            return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("localpilot.runtime_supervisor.subprocess.Popen", fake_popen)
    monkeypatch.setattr("localpilot.runtime_supervisor.threading.Thread", FakeThread)
    supervisor = RuntimeSupervisor(tmp_path)

    supervisor.start()
    supervisor.send({"kind": "ask", "prompt": "emoji 🤖"})

    assert captured["argv"][1:3] == ["-m", "localpilot.runtime_worker"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert json.loads(FakeProcess.stdin.getvalue())["prompt"] == "emoji 🤖"


def test_supervisor_does_not_forward_raw_worker_stderr(tmp_path):
    messages = []
    supervisor = RuntimeSupervisor(tmp_path, on_message=messages.append)
    process = SimpleNamespace(stderr=io.StringIO("private prompt fragment\n"))

    supervisor._read_stderr(process)

    assert messages == [
        {"kind": "supervisor", "type": "runtime.stderr", "payload": {"diagnostic": True}}
    ]
    assert "private prompt fragment" not in json.dumps(messages)


def test_supervisor_replaces_crashed_pid_and_records_crash_recovery(tmp_path, monkeypatch):
    processes = []

    class FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

        def __init__(self, pid, returncode):
            self.pid = pid
            self.returncode = returncode

        def poll(self):
            return None

        def wait(self, timeout=None):
            return self.returncode

    def fake_popen(*args, **kwargs):
        process = FakeProcess(700 + len(processes), 23 if not processes else 0)
        processes.append(process)
        return process

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("localpilot.runtime_supervisor.subprocess.Popen", fake_popen)
    monkeypatch.setattr("localpilot.runtime_supervisor.threading.Thread", FakeThread)
    monkeypatch.setattr("localpilot.runtime_supervisor.time.sleep", lambda seconds: None)
    supervisor = RuntimeSupervisor(tmp_path)

    supervisor.start()
    old_pid = supervisor.pid
    supervisor._watch(processes[0])

    assert old_pid == 700
    assert supervisor.pid == 701
    rows = supervisor.audit.recent("runtime_lifecycle", limit=10)
    crash_exit = next(row for row in rows if row["transition"] == "exited")
    replacement = next(
        row
        for row in rows
        if row["transition"] == "started" and row.get("new_pid") == 701
    )
    assert crash_exit["source"] == "crash_recovery"
    assert crash_exit["old_pid"] == 700
    assert crash_exit["return_code"] == 23
    assert replacement["source"] == "crash_recovery"
    assert replacement["old_pid"] == 700


def test_explicit_restart_is_classified_and_background_source_is_rejected(tmp_path, monkeypatch):
    processes = []

    class FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

        def __init__(self, pid):
            self.pid = pid
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(*args, **kwargs):
        process = FakeProcess(800 + len(processes))
        processes.append(process)
        return process

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("localpilot.runtime_supervisor.subprocess.Popen", fake_popen)
    monkeypatch.setattr("localpilot.runtime_supervisor.threading.Thread", FakeThread)
    supervisor = RuntimeSupervisor(tmp_path)
    supervisor.start()

    with pytest.raises(ValueError, match="Unsupported whole-runtime restart source"):
        supervisor.restart(source="autonomous_background", reason="background_cycle")
    assert processes[0].terminated is False

    assert supervisor.restart(
        source="explicit_api_restart",
        reason="owner_requested_via_api",
        request_id="request-1",
        session_id="session-1",
        message_id=9,
    )
    assert processes[0].terminated is True
    supervisor._watch(processes[0])

    assert supervisor.pid == 801
    rows = supervisor.audit.recent("runtime_lifecycle", limit=10)
    requested = next(row for row in rows if row["transition"] == "restart_requested")
    replacement = next(row for row in rows if row.get("new_pid") == 801)
    assert requested["source"] == "explicit_api_restart"
    assert requested["request_id"] == "request-1"
    assert requested["session_id"] == "session-1"
    assert requested["message_id"] == 9
    assert replacement["source"] == "explicit_api_restart"


def test_broker_token_is_stable_and_desktop_config_refuses_non_loopback(tmp_path):
    config = Config()
    first = load_or_create_broker_token(tmp_path, config)
    second = load_or_create_broker_token(tmp_path, config)
    assert first == second
    assert len(first) >= 32

    config_file = tmp_path / "localpilot.toml"
    config_file.write_text('[desktop]\nhost = "0.0.0.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="loopback-only"):
        load_config(config_file)

    config_file.write_text('[desktop]\nchat_database = "../learning.sqlite3"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="one local filename"):
        load_config(config_file)

    config_file.write_text('[desktop]\nchat_database = "learning.sqlite3"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="remain separate"):
        load_config(config_file)


def test_agent_event_observer_reports_state_without_exposing_reasoning(tmp_path, monkeypatch):
    events = []
    chunks = [
        SimpleNamespace(
            message=SimpleNamespace(
                thinking="private chain of thought",
                content="Visible 🤖",
                tool_calls=[],
            )
        )
    ]
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=lambda **kwargs: iter(chunks)))
    agent = LocalPilotAgent(Config(), tmp_path, event_sink=events.append)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )

    answer = agent.ask("Say hello")

    assert answer == "Visible 🤖"
    assert {event["payload"].get("state") for event in events} >= {"thinking", "speaking", "idle"}
    assert "private chain of thought" not in json.dumps(events)
