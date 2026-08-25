from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from localpilot.chat_store import ChatStore
from localpilot.config import Config, load_config
from localpilot.runtime_supervisor import RuntimeSupervisor


def broker_token_path(root: str | Path, config: Config) -> Path:
    return (Path(root).resolve() / config.agent.data_dir / "broker-token").resolve()


def load_or_create_broker_token(root: str | Path, config: Config) -> str:
    path = broker_token_path(root, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(token)
        except FileExistsError:
            token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("The LocalPilot broker token is empty")
    return token


class BrokerApp:
    """Stable process boundary for UI continuity, sessions, and replaceable runtimes."""

    def __init__(self, root: str | Path, config: Config, *, config_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.config = config
        data_dir = (self.root / config.agent.data_dir).resolve()
        self.store = ChatStore(data_dir / config.desktop.chat_database)
        self.token = load_or_create_broker_token(self.root, config)
        self._condition = threading.Condition()
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, Any]] = {}
        self.runtime = RuntimeSupervisor(
            self.root,
            config_path=config_path,
            restart_limit=config.desktop.runtime_restart_limit,
            on_message=self._on_runtime_message,
        )

    def start(self) -> None:
        self.runtime.start()

    def stop(self) -> None:
        self.runtime.stop()

    def _event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        event = self.store.append_event(event_type, payload, session_id=session_id)
        with self._condition:
            self._condition.notify_all()
        return event

    def submit(self, session_id: str, content: str) -> dict[str, Any]:
        content = str(content)
        if not content.strip():
            raise ValueError("Message content must not be empty")
        if len(content) > 100_000:
            raise ValueError("Message content exceeds the 100,000 character limit")
        self.store.session(session_id)
        history = self.store.completed_history(session_id)
        user_message = self.store.add_message(session_id, "user", content.strip())
        assistant_message = self.store.add_message(session_id, "assistant", "", status="streaming")
        request_id = str(uuid.uuid4())
        with self._lock:
            self._pending[request_id] = {
                "session_id": session_id,
                "message_id": assistant_message["id"],
                "content": "",
            }
        self._event("message.created", {"message": user_message}, session_id=session_id)
        self._event("message.created", {"message": assistant_message}, session_id=session_id)
        try:
            self.runtime.send(
                {
                    "kind": "ask",
                    "request_id": request_id,
                    "session_id": session_id,
                    "prompt": content.strip(),
                    "history": history,
                }
            )
        except Exception as exc:
            with self._lock:
                self._pending.pop(request_id, None)
            marker = f"[LocalPilot runtime unavailable: {type(exc).__name__}: {exc}]"
            failed = self.store.update_message(assistant_message["id"], marker, status="error")
            self._event(
                "runtime.error",
                {"error_type": type(exc).__name__, "message": str(exc), "message_record": failed},
                session_id=session_id,
            )
            raise
        return {"request_id": request_id, "user": user_message, "assistant": assistant_message}

    def _fail_pending(self, reason: str) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            marker = f"[LocalPilot runtime restarted before this answer completed: {reason}]"
            failed = self.store.update_message(item["message_id"], marker, status="error")
            self._event(
                "message.failed",
                {"message": failed, "reason": reason},
                session_id=item["session_id"],
            )

    def _on_runtime_message(self, message: dict[str, Any]) -> None:
        kind = str(message.get("kind") or "")
        if kind == "ready":
            self._event("runtime.ready", {"pid": message.get("pid")})
            self._event("runtime.state", {"state": "idle"})
            return
        if kind == "supervisor":
            event_type = str(message.get("type") or "runtime.supervisor")
            payload = dict(message.get("payload") or {})
            if event_type == "runtime.exited":
                self._fail_pending(f"worker exited with code {payload.get('returncode')}")
                self._event("runtime.state", {"state": "restarting"})
            elif event_type == "runtime.unavailable":
                self._event("runtime.state", {"state": "error"})
            elif event_type in {"runtime.starting", "runtime.restarting"}:
                self._event("runtime.state", {"state": "restarting", **payload})
            self._event(event_type, payload)
            return

        session_id = str(message.get("session_id") or "") or None
        request_id = str(message.get("request_id") or "")
        if kind == "event":
            event_type = str(message.get("type") or "runtime.event")
            payload = dict(message.get("payload") or {})
            if event_type == "assistant.delta":
                with self._lock:
                    pending = self._pending.get(request_id)
                    if pending is not None:
                        pending["content"] += str(payload.get("delta") or "")
                        updated = self.store.update_message(
                            pending["message_id"], pending["content"], status="streaming"
                        )
                        payload["message_id"] = pending["message_id"]
                        payload["content"] = updated["content"]
            self._event(event_type, payload, session_id=session_id)
            return

        if kind == "result":
            with self._lock:
                pending = self._pending.pop(request_id, None)
            if pending is None:
                return
            answer = str(message.get("answer") or pending["content"])
            completed = self.store.update_message(pending["message_id"], answer, status="complete")
            self._event("message.completed", {"message": completed}, session_id=session_id)
            self._event("runtime.state", {"state": "success"}, session_id=session_id)
            self._event("runtime.state", {"state": "idle"}, session_id=session_id)
            return

        if kind in {"error", "protocol_error"}:
            with self._lock:
                pending = self._pending.pop(request_id, None)
            payload = {
                "error_type": str(message.get("error_type") or "RuntimeError"),
                "message": "The local runtime reported an error.",
            }
            if pending is not None:
                marker = f"[LocalPilot error: {payload['error_type']}: {payload['message']}]"
                failed = self.store.update_message(pending["message_id"], marker, status="error")
                payload["message_record"] = failed
            self._event("runtime.error", payload, session_id=session_id)
            self._event("runtime.state", {"state": "error"}, session_id=session_id)

    def events_after(
        self,
        after_id: int,
        *,
        session_id: str | None,
        wait_seconds: float,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, min(float(wait_seconds), 25.0))
        while True:
            events = self.store.events_after(after_id, session_id=session_id)
            if events or time.monotonic() >= deadline:
                return events
            remaining = deadline - time.monotonic()
            with self._condition:
                self._condition.wait(timeout=max(0.0, remaining))


class BrokerRequestHandler(BaseHTTPRequestHandler):
    server: "BrokerHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.app.token}"
        return secrets.compare_digest(self.headers.get("Authorization", ""), expected)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 1_000_000:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def _route(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urllib.parse.urlparse(self.path)
        parts = [item for item in parsed.path.split("/") if item]
        return parts, urllib.parse.parse_qs(parsed.query)

    def do_GET(self) -> None:
        try:
            parts, query = self._route()
            if parts == ["health"]:
                self._json(
                    HTTPStatus.OK,
                    {"ok": True, "runtime": "running" if self.server.app.runtime.running else "restarting"},
                )
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if parts == ["v1", "sessions"]:
                self._json(HTTPStatus.OK, {"sessions": self.server.app.store.sessions()})
                return
            if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":
                self.server.app.store.session(parts[2])
                self._json(
                    HTTPStatus.OK,
                    {"messages": self.server.app.store.messages(parts[2])},
                )
                return
            if parts == ["v1", "events"]:
                after = int((query.get("after") or ["0"])[0])
                wait = float((query.get("wait") or ["0"])[0])
                session_id = (query.get("session_id") or [None])[0]
                events = self.server.app.events_after(after, session_id=session_id, wait_seconds=wait)
                self._json(HTTPStatus.OK, {"events": events})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:
        try:
            parts, _ = self._route()
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            body = self._body()
            if parts == ["v1", "sessions"]:
                session = self.server.app.store.create_session(str(body.get("title") or "New conversation"))
                self.server.app._event("session.created", {"session": session}, session_id=session["id"])
                self._json(HTTPStatus.CREATED, {"session": session})
                return
            if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":
                result = self.server.app.submit(parts[2], str(body.get("content") or ""))
                self._json(HTTPStatus.ACCEPTED, result)
                return
            if parts == ["v1", "runtime", "restart"]:
                self.server.app.runtime.restart()
                self._json(HTTPStatus.ACCEPTED, {"status": "restarting"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})


class BrokerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: BrokerApp) -> None:
        self.app = app
        super().__init__(address, BrokerRequestHandler)


def serve(root: str | Path, config: Config, *, config_path: str | Path | None = None) -> None:
    app = BrokerApp(root, config, config_path=config_path)
    server = BrokerHTTPServer((config.desktop.host, config.desktop.port), app)
    app.start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        app.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-broker")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve(args.root, load_config(args.config), config_path=args.config)


if __name__ == "__main__":
    main()
