from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from localpilot.chat_commands import execute_chat_command, parse_chat_command, render_help
from localpilot.config import Config
from localpilot.runtime_worker import RuntimeWorker


class _FakeAgent:
    def __init__(self):
        self.ask_calls = []
        self.teach_calls = []

    def ask(self, prompt, *, interface="direct"):
        self.ask_calls.append((prompt, interface))
        return "model answer"

    def teach(self, lesson, *, topic="general"):
        self.teach_calls.append((lesson, topic))
        return SimpleNamespace(id=7, lesson=lesson)


def _worker(tmp_path: Path, agent: _FakeAgent):
    worker = RuntimeWorker.__new__(RuntimeWorker)
    worker.root = tmp_path
    worker.config = Config()
    worker._agents = {}
    worker._write_lock = threading.Lock()
    worker._active_request_id = None
    worker._active_session_id = None
    worker._agent = lambda _session_id, _history: agent
    messages = []
    worker._write = messages.append
    return worker, messages


def test_parse_chat_command_reserves_only_slash_prefixed_input():
    assert parse_chat_command("hello") is None
    parsed = parse_chat_command("  /TeAcH   Keep answers concise  ")
    assert parsed is not None
    assert parsed.name == "/teach"
    assert parsed.argument == "Keep answers concise"
    assert parse_chat_command("/new").name == "/clear"


def test_help_explains_desktop_commands_and_memory_boundary():
    help_text = render_help()
    for command in ("/help", "/status", "/doctor", "/teach", "/evolve", "/clear"):
        assert command in help_text
    assert "not sent to the language model" in help_text
    assert "LearningMemory" in help_text


def test_teach_uses_explicit_durable_agent_path_without_model_inference(tmp_path):
    agent = _FakeAgent()
    result = execute_chat_command(
        parse_chat_command("/teach Verify current evidence before conclusions"),
        agent=agent,
        config=Config(),
        root=tmp_path,
    )
    assert agent.teach_calls == [("Verify current evidence before conclusions", "chat")]
    assert agent.ask_calls == []
    assert "Teaching #7 saved" in result.text


def test_unknown_slash_command_never_falls_through_to_model(tmp_path):
    agent = _FakeAgent()
    worker, messages = _worker(tmp_path, agent)

    worker.handle(
        {
            "kind": "ask",
            "request_id": "req-1",
            "session_id": "session-1",
            "prompt": "/definitely-not-a-command",
            "history": [],
        }
    )

    assert agent.ask_calls == []
    result = next(message for message in messages if message.get("kind") == "result")
    assert "Unknown command" in result["answer"]
    assert "/help" in result["answer"]


def test_help_command_bypasses_agent_entirely(tmp_path):
    agent = _FakeAgent()
    worker, messages = _worker(tmp_path, agent)

    worker.handle(
        {
            "kind": "ask",
            "request_id": "req-2",
            "session_id": "session-1",
            "prompt": "/help",
            "history": [],
        }
    )

    assert agent.ask_calls == []
    result = next(message for message in messages if message.get("kind") == "result")
    assert "LocalPilot commands" in result["answer"]


def test_normal_desktop_message_still_uses_agent(tmp_path):
    agent = _FakeAgent()
    worker, messages = _worker(tmp_path, agent)

    worker.handle(
        {
            "kind": "ask",
            "request_id": "req-3",
            "session_id": "session-1",
            "prompt": "How are you going?",
            "history": [],
        }
    )

    assert agent.ask_calls == [("How are you going?", "desktop")]
    result = next(message for message in messages if message.get("kind") == "result")
    assert result["answer"] == "model answer"


def test_desktop_assets_expose_palette_and_local_clear_contract():
    root = Path(__file__).resolve().parents[1]
    html = (root / "localpilot" / "webview" / "index.html").read_text(encoding="utf-8")
    script = (root / "localpilot" / "webview" / "desktop-commands.js").read_text(encoding="utf-8")

    assert 'href="desktop-commands.css"' in html
    assert 'src="desktop-commands.js"' in html
    assert 'name: "/teach"' in script
    assert 'name: "/evolve"' in script
    assert 'name: "/clear"' in script
    assert "historyNew.click()" in script
    assert 'addEventListener("keydown"' in script
