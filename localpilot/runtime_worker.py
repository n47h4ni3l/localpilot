from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any

from localpilot.agent import LocalPilotAgent
from localpilot.config import load_config


class RuntimeWorker:
    """JSONL adapter around the one authoritative LocalPilotAgent implementation."""

    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.config = load_config(config_path)
        self._agents: dict[str, LocalPilotAgent] = {}
        self._write_lock = threading.Lock()
        self._active_request_id: str | None = None
        self._active_session_id: str | None = None

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

    def _agent(self, session_id: str, history: list[dict[str, Any]]) -> LocalPilotAgent:
        agent = self._agents.get(session_id)
        if agent is not None:
            return agent
        agent = LocalPilotAgent(self.config, self.root, event_sink=self._event_sink)
        for message in history:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role in {"user", "assistant"} and content.strip():
                agent.messages.append({"role": role, "content": content})
        self._agents[session_id] = agent
        return agent

    @staticmethod
    def _answer_chunks(answer: str, size: int = 80) -> list[str]:
        return [answer[index : index + size] for index in range(0, len(answer), size)] or [""]

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
            self._write(
                {
                    "kind": "event",
                    "request_id": request_id,
                    "session_id": session_id,
                    "type": "runtime.state",
                    "payload": {"state": "thinking"},
                }
            )
            agent = self._agent(session_id, list(command.get("history") or []))
            answer = agent.ask(prompt)
            self._write(
                {
                    "kind": "event",
                    "request_id": request_id,
                    "session_id": session_id,
                    "type": "runtime.state",
                    "payload": {"state": "speaking"},
                }
            )
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

    def run(self) -> None:
        self._write({"kind": "ready", "pid": __import__("os").getpid()})
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
