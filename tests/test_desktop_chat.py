import io
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localpilot.broker import BrokerApp, load_or_create_broker_token
from localpilot.chat_store import ChatStore
from localpilot.config import Config, load_config
from localpilot.agent import LocalPilotAgent
from localpilot.runtime_supervisor import RuntimeSupervisor
from localpilot.runtime_worker import RuntimeWorker


class _FakeRuntime:
    def __init__(self):
        self.sent = []
        self.running = True

    def send(self, message):
        self.sent.append(message)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def restart(self):
        self.running = False


def _broker(tmp_path):
    config = Config()
    app = BrokerApp(tmp_path, config)
    app.runtime = _FakeRuntime()
    return app


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


def test_broker_request_timeout_fails_message_and_restarts_owned_runtime(tmp_path):
    config = Config()
    config.desktop.request_timeout_seconds = 60
    app = BrokerApp(tmp_path, config)
    app.runtime = _FakeRuntime()
    session = app.store.create_session()
    submitted = app.submit(session["id"], "A bounded request")

    app._expire_request(submitted["request_id"])

    failed = app.store.message(submitted["assistant"]["id"])
    assert failed["status"] == "error"
    assert "timed out after 60 seconds" in failed["content"]
    assert app.runtime.running is False
    assert any(
        event["payload"].get("reason") == "request_timeout"
        for event in app.store.events_after(0)
        if event["type"] == "message.failed"
    )


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
