from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any

from localpilot.agent import LocalPilotAgent
from localpilot.background_reading import BackgroundLibraryReader
from localpilot.chat_commands import (
    execute_chat_command,
    is_model_backed_chat_command,
    parse_chat_command,
)
from localpilot.systemsense_diagnosis import normalize_diagnosis_scope
from localpilot.systemsense_watch import parse_systemsense_watch_request
from localpilot.config import load_config
from localpilot.systemsense import get_system_sense


class RuntimeWorker:
    """JSONL adapter around the one authoritative LocalPilotAgent implementation."""

    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.config = load_config(config_path)
        self.systemsense = get_system_sense(
            self.config.systemsense,
            self.root / self.config.agent.data_dir,
            project_root=self.root,
            main_branch=self.config.github.main_branch,
        )
        self._agents: dict[str, LocalPilotAgent] = {}
        self._write_lock = threading.Lock()
        self._active_request_id: str | None = None
        self._active_session_id: str | None = None
        self._background_stop = threading.Event()
        self._background_reader: BackgroundLibraryReader | None = None
        self._background_thread: threading.Thread | None = None

    def _write(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._write_lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    def _event_sink(self, event: dict[str, Any]) -> None:
        self._write(
            {
                "kind": "event",
                "request_id": self._active_request_id,
                "session_id": self._active_session_id,
                "type": str(event.get("type") or "runtime.event"),
                "payload": dict(event.get("payload") or {}),
            }
        )

    @staticmethod
    def _conversation_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Replay visible conversation while excluding application-command transcripts.

        Slash commands stay in ChatStore so the owner can see what happened, but
        deterministic command inputs/results are application state, not dialogue
        evidence for the model. Keeping them out of replay also prevents large
        `/status` diagnostics from consuming conversation context after restart.
        """
        replay: list[dict[str, str]] = []
        index = 0
        while index < len(history):
            message = history[index]
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            parsed = parse_chat_command(content) if role == "user" else None
            if parsed is not None and not is_model_backed_chat_command(parsed):
                index += 1
                if index < len(history) and str(history[index].get("role") or "") == "assistant":
                    index += 1
                continue
            if role in {"user", "assistant"} and content.strip():
                replay.append({"role": role, "content": content})
            index += 1
        return replay

    def _agent(self, session_id: str, history: list[dict[str, Any]]) -> LocalPilotAgent:
        agent = self._agents.get(session_id)
        if agent is not None:
            return agent
        # LocalPilotAgent resolves the same process-local SystemSense singleton
        # by database path, preserving its established constructor surface.
        agent = LocalPilotAgent(self.config, self.root, event_sink=self._event_sink)
        agent.messages.extend(self._conversation_history(history))
        self._agents[session_id] = agent
        return agent

    @staticmethod
    def _answer_chunks(answer: str, size: int = 80) -> list[str]:
        return [answer[index : index + size] for index in range(0, len(answer), size)] or [""]

    def _write_state(self, request_id: str, session_id: str, state: str) -> None:
        self._write(
            {
                "kind": "event",
                "request_id": request_id,
                "session_id": session_id,
                "type": "runtime.state",
                "payload": {"state": state},
            }
        )

    def _write_answer(self, request_id: str, session_id: str, answer: str) -> None:
        self._write_state(request_id, session_id, "speaking")
        for delta in self._answer_chunks(answer):
            self._write(
                {
                    "kind": "event",
                    "request_id": request_id,
                    "session_id": session_id,
                    "type": "assistant.delta",
                    "payload": {"delta": delta},
                }
            )
        self._write(
            {
                "kind": "result",
                "request_id": request_id,
                "session_id": session_id,
                "answer": answer,
            }
        )

    def _command_progress(self, command_name: str, message: str) -> None:
        self._write(
            {
                "kind": "event",
                "request_id": self._active_request_id,
                "session_id": self._active_session_id,
                "type": "command.progress",
                "payload": {
                    "command": command_name,
                    "message": str(message)[:500],
                },
            }
        )

    def handle(self, command: dict[str, Any]) -> None:
        if command.get("kind") != "ask":
            raise ValueError("Unsupported runtime command")
        request_id = str(command.get("request_id") or "")
        session_id = str(command.get("session_id") or "")
        prompt = str(command.get("prompt") or "")
        if not request_id or not session_id or not prompt.strip():
            raise ValueError("ask requires request_id, session_id, and prompt")
        self._active_request_id = request_id
        self._active_session_id = session_id
        try:
            parsed = parse_chat_command(prompt)
            if parsed is not None:
                # Most slash commands are deterministic application controls.
                # /diagnose is deliberately different: it gathers fresh raw
                # SystemSense evidence, then asks Astra to reason over that
                # evidence without allowing presentation data or durable memory
                # to become diagnostic authority.
                self._write_state(request_id, session_id, "working")
                if parsed.name == "/diagnose":
                    try:
                        scope = normalize_diagnosis_scope(parsed.argument)
                    except ValueError:
                        self._write_answer(
                            request_id,
                            session_id,
                            "Usage: `/diagnose [signals|thermal|memory|gpu|cpu|storage]`",
                        )
                        return
                    active_agent = self._agent(
                        session_id, list(command.get("history") or [])
                    )
                    message_count = len(active_agent.messages)
                    try:
                        answer = active_agent.diagnose_system(scope)
                    finally:
                        # The large raw evidence package is one-turn reasoning
                        # substrate, not durable chat context. Keep only the
                        # visible command and resulting diagnosis for follow-up.
                        del active_agent.messages[message_count:]
                    active_agent.messages.extend(
                        [
                            {"role": "user", "content": parsed.source},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                    self._write_answer(request_id, session_id, answer)
                    return

                active_agent = (
                    self._agent(session_id, list(command.get("history") or []))
                    if parsed.name == "/teach"
                    else None
                )
                result = execute_chat_command(
                    parsed,
                    agent=active_agent,
                    config=self.config,
                    root=self.root,
                    progress=lambda message: self._command_progress(parsed.name, message),
                )
                self._write_answer(request_id, session_id, result.text)
                return

            watch_intent = parse_systemsense_watch_request(prompt)
            if watch_intent is not None:
                self._write_state(request_id, session_id, "working")
                watch = self.systemsense.start_watch(
                    profile=watch_intent.profile,
                    expires_at=watch_intent.expires_at,
                    label=watch_intent.label,
                )
                agent = self._agent(session_id, list(command.get("history") or []))
                answer = agent.acknowledge_systemsense_watch(prompt, watch)
                self._write_answer(request_id, session_id, answer)
                return

            self._write_state(request_id, session_id, "thinking")
            agent = self._agent(session_id, list(command.get("history") or []))
            answer = agent.ask(prompt, interface="desktop")
            self._write_answer(request_id, session_id, answer)
        except Exception as exc:
            self._write(
                {
                    "kind": "error",
                    "request_id": request_id,
                    "session_id": session_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            self._active_request_id = None
            self._active_session_id = None

    def _start_background_reader(self) -> None:
        if self._background_thread is not None and self._background_thread.is_alive():
            return
        self._background_stop.clear()
        try:
            reader = BackgroundLibraryReader(self.config, self.root)
        except Exception:
            # Background reading is optional and must never prevent the stable
            # operator worker from starting. The reader records later runtime
            # failures itself once successfully constructed.
            return
        self._background_reader = reader
        self._background_thread = threading.Thread(
            target=reader.run_forever,
            args=(self._background_stop,),
            name="localpilot-background-library-reader",
            daemon=True,
        )
        self._background_thread.start()

    def run(self) -> None:
        self._write({"kind": "ready", "pid": __import__("os").getpid()})
        self.systemsense.start()
        self._start_background_reader()
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    self.handle(json.loads(line))
                except Exception as exc:
                    self._write(
                        {
                            "kind": "protocol_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
        finally:
            self._background_stop.set()
            self.systemsense.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-runtime-worker")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    RuntimeWorker(args.root, args.config).run()


if __name__ == "__main__":
    main()
