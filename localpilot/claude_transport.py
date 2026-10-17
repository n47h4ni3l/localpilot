"""One-call-lifetime, loopback-only Claude/Ollama protocol validation.

Claude Code 2.1.263 crashes on an unnamed Anthropic tool_use block before
its normal tool checks. The historical malformed block's origin is unproven.
Collect one bounded provider response before forwarding anything, validate tool identities,
and serialize valid JSON as canonical SSE when the CLI requests streaming.
No missing tool name or argument is inferred and no tool is executed here.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import secrets
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit


class ClaudeTransportError(ValueError):
    pass


class _CancellableReader(io.RawIOBase):
    """Keep HTTP parser buffering intact while polling cancellation on Windows."""

    def __init__(self, connection: socket.socket, check: Callable[[], Any]) -> None:
        self.connection = connection.dup()
        self.check = check

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while True:
            self.check()
            if select.select([self.connection], [], [], 0.1)[0]:
                return self.connection.recv_into(buffer)

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            super().close()


class _ResponseSocket:
    def __init__(self, connection: socket.socket, check: Callable[[], Any]) -> None:
        self.connection, self.check = connection, check

    def makefile(self, *_args: Any) -> io.BufferedReader:
        return io.BufferedReader(_CancellableReader(self.connection, self.check))


def _json(data: bytes) -> Any:
    def invalid(value: str) -> None:
        raise ClaudeTransportError("non-finite JSON value")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ClaudeTransportError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(data, parse_constant=invalid, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ClaudeTransportError("invalid provider JSON") from exc


def validate_message(message: Any, request: dict[str, Any], model: str) -> dict[str, Any]:
    """Validate wire structure, never guess names or change assistant content."""
    if not isinstance(message, dict) or message.get("type") != "message":
        raise ClaudeTransportError("provider did not return a message")
    if message.get("model") != model or message.get("role") != "assistant":
        raise ClaudeTransportError("provider model or role differs from the request")
    if not isinstance(message.get("id"), str) or not message["id"]:
        raise ClaudeTransportError("message is missing its identity")
    content = message.get("content")
    if not isinstance(content, list) or len(content) > 128:
        raise ClaudeTransportError("invalid content block list")
    advertised = {
        tool["name"] for tool in request.get("tools", [])
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }
    identities: set[str] = set()
    for block in content:
        if not isinstance(block, dict):
            raise ClaudeTransportError("invalid content block")
        kind = block.get("type")
        if kind == "tool_use":
            name, identity = block.get("name"), block.get("id")
            if not isinstance(name, str) or not name.strip():
                raise ClaudeTransportError("tool_use is missing a name")
            if name not in advertised:
                raise ClaudeTransportError("tool_use names an unadvertised tool")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ClaudeTransportError("tool_use has a missing or duplicate identity")
            identities.add(identity)
            if not isinstance(block.get("input"), dict):
                raise ClaudeTransportError("tool_use input must be an object")
        elif kind == "text":
            if not isinstance(block.get("text"), str):
                raise ClaudeTransportError("text block is missing text")
        elif kind == "thinking":
            if not isinstance(block.get("thinking"), str):
                raise ClaudeTransportError("thinking block is missing text")
            if "signature" in block and not isinstance(block["signature"], str):
                raise ClaudeTransportError("thinking signature must be a string")
        else:
            raise ClaudeTransportError("unsupported provider content block")
    if message.get("stop_reason") not in {
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal",
    }:
        raise ClaudeTransportError("invalid message stop reason")
    if bool(identities) != (message.get("stop_reason") == "tool_use"):
        raise ClaudeTransportError("tool_use blocks and stop reason disagree")
    usage = message.get("usage")
    if not isinstance(usage, dict) or any(
        isinstance(usage.get(key), bool) or not isinstance(usage.get(key), int)
        or usage[key] < 0 for key in ("input_tokens", "output_tokens")
    ):
        raise ClaudeTransportError("invalid provider usage")
    return message


def message_events(message: dict[str, Any]) -> bytes:
    """Serialize validated content in the official Messages API event order."""
    events: list[dict[str, Any]] = []
    head = dict(message, content=[], stop_reason=None, stop_sequence=None)
    head["usage"] = dict(message["usage"], output_tokens=0)
    events.append({"type": "message_start", "message": head})
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        initial = {"type": kind}
        if kind == "tool_use":
            initial.update(id=block["id"], name=block["name"], input={})
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}
        elif kind == "text":
            initial["text"] = ""
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            initial["thinking"] = ""
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        events.append({"type": "content_block_start", "index": index, "content_block": initial})
        events.append({"type": "content_block_delta", "index": index, "delta": delta})
        if kind == "thinking" and block.get("signature"):
            events.append({"type": "content_block_delta", "index": index,
                           "delta": {"type": "signature_delta", "signature": block["signature"]}})
        events.append({"type": "content_block_stop", "index": index})
    events.append({"type": "message_delta", "delta": {
        "stop_reason": message["stop_reason"], "stop_sequence": message.get("stop_sequence")},
        "usage": message["usage"]})
    events.append({"type": "message_stop"})
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                   for event in events).encode("utf-8")


class ClaudeTransport:
    """Private authenticated gateway owned and closed by one CLI invocation."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float,
                 resource_guard: Callable[[], None] | None = None,
                 max_request_bytes: int = 4_000_000, max_response_bytes: int = 2_000_000,
                 max_requests: int = 168) -> None:
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("Claude transport requires a loopback HTTP Ollama endpoint")
        self.host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
        self.port = parsed.port or 80
        self.model, self.resource_guard = model, resource_guard
        self.deadline = time.monotonic() + max(0.1, timeout_seconds)
        self.max_request_bytes, self.max_response_bytes = max_request_bytes, max_response_bytes
        self.max_requests = max_requests
        self.token = secrets.token_hex(32)
        self._stop = threading.Event()
        self._serial = threading.BoundedSemaphore(1)
        self._connection: http.client.HTTPConnection | None = None
        self._provider_socket: socket.socket | None = None
        self._connection_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.requests = 0
        self.retries = 0
        self.failures: list[dict[str, Any]] = []
        self.resource_stop = ""
        self.deadline_stop = False

    def _check(self, resource: bool = True) -> float:
        remaining = self.deadline - time.monotonic()
        if self._stop.is_set() or remaining <= 0:
            self.deadline_stop = remaining <= 0
            raise ClaudeTransportError("invocation transport deadline reached")
        if resource and self.resource_guard is not None:
            try:
                self.resource_guard()
            except Exception as exc:
                self.resource_stop = type(exc).__name__
                raise ClaudeTransportError("invocation resource guard stopped the request") from exc
        return remaining

    def _provider(self, path: str, payload: dict[str, Any], headers: Any) -> tuple[int, bytes]:
        remaining = self._check()
        if self.requests >= self.max_requests:
            raise ClaudeTransportError("invocation provider request limit reached")
        self.requests += 1
        connection = http.client.HTTPConnection(self.host, self.port, timeout=min(remaining, 5.0))
        gateway = self
        class Response(http.client.HTTPResponse):
            def __init__(self, sock: socket.socket, *args: Any, **kwargs: Any) -> None:
                super().__init__(_ResponseSocket(sock, lambda: gateway._check(resource=False)), *args, **kwargs)
        connection.response_class = Response
        response = None
        with self._connection_lock:
            self._connection = connection
        try:
            forwarded = {"Content-Type": "application/json", "Authorization": "Bearer ollama",
                         "anthropic-version": "2023-06-01"}
            if headers.get("anthropic-beta"):
                forwarded["anthropic-beta"] = str(headers["anthropic-beta"])[:4096]
            connection.request("POST", path, json.dumps(payload).encode("utf-8"), forwarded)
            with self._connection_lock:
                # A duplicate remains usable after HTTP/1.0 getresponse closes
                # HTTPConnection's socket object while its makefile still reads.
                self._provider_socket = connection.sock.dup() if connection.sock is not None else None
                if self._stop.is_set():
                    raise ClaudeTransportError("invocation transport is closed")
                if connection.sock is not None:
                    connection.sock.settimeout(self._check())
            response = connection.getresponse()
            chunks: list[bytes] = []
            total = 0
            while True:
                self._check()
                chunk = response.read(min(65536, self.max_response_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise ClaudeTransportError("provider response exceeds byte limit")
            return response.status, b"".join(chunks)
        except TimeoutError:
            self.deadline_stop = time.monotonic() >= self.deadline
            raise
        finally:
            if response is not None:
                response.close()
            connection.close()
            with self._connection_lock:
                if self._connection is connection:
                    self._connection = None
                    if self._provider_socket is not None:
                        self._provider_socket.close()
                    self._provider_socket = None

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        def send(status: int, data: bytes, content_type: str = "application/json") -> None:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            try:
                handler.wfile.write(data)
            except OSError:
                pass

        def error(status: int, message: str) -> None:
            send(status, json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": message}}).encode())

        if not secrets.compare_digest(str(handler.headers.get("Authorization", "")), "Bearer " + self.token):
            error(401, "private Claude transport authorization required")
            return
        if not self._serial.acquire(blocking=False):
            error(409, "Claude transport already has a provider request")
            return
        try:
            self._check()
            path = urlsplit(handler.path)
            if (path.scheme or path.netloc or path.fragment or path.path not in {
                    "/v1/messages", "/v1/messages/count_tokens"} or len(handler.path) > 256):
                error(404, "unsupported Claude transport endpoint")
                return
            if handler.headers.get("Transfer-Encoding"):
                raise ClaudeTransportError("chunked requests are not supported")
            length = int(handler.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_request_bytes:
                raise ClaudeTransportError("request exceeds byte limit or has no body")
            raw = handler.rfile.read(length)
            if len(raw) != length:
                raise ClaudeTransportError("incomplete request body")
            payload = _json(raw)
            if not isinstance(payload, dict) or payload.get("model") != self.model:
                raise ClaudeTransportError("request differs from the accepted model")
            tools = payload.get("tools", [])
            if not isinstance(tools, list) or any(not isinstance(tool, dict) or
                    not isinstance(tool.get("name"), str) or not tool["name"] for tool in tools):
                raise ClaudeTransportError("request contains invalid tool declarations")
            streaming = payload.get("stream") is True and path.path == "/v1/messages"
            upstream = dict(payload, stream=False) if path.path == "/v1/messages" else dict(payload)
            while True:
                status, body = self._provider(handler.path, upstream, handler.headers)
                if status != 200:
                    # The CLI sees real provider failures; no model substitution or retry here.
                    send(status, body)
                    return
                try:
                    response = _json(body)
                    if path.path == "/v1/messages/count_tokens":
                        if (not isinstance(response, dict) or isinstance(response.get("input_tokens"), bool)
                                or not isinstance(response.get("input_tokens"), int) or response["input_tokens"] < 0):
                            raise ClaudeTransportError("invalid token count response")
                    else:
                        validate_message(response, payload, self.model)
                    break
                except ClaudeTransportError as exc:
                    if len(self.failures) < 8:
                        self.failures.append({"reason": str(exc), "sha256": hashlib.sha256(body).hexdigest(),
                                              "bytes": len(body), "forwarded": False})
                    if self.retries >= 1 or path.path != "/v1/messages":
                        raise
                    # Nothing from this response was forwarded, so none of its tool calls ran.
                    self._check()
                    self.retries += 1
            send(200, message_events(response) if streaming else body,
                 "text/event-stream" if streaming else "application/json")
        except (ClaudeTransportError, ValueError, OSError, http.client.HTTPException) as exc:
            error(400, "Claude transport protocol error: " + str(exc)[:200] +
                  "; no tool from the rejected response was forwarded")
        except Exception:
            # A resource guard can raise its own exception type. Fail closed without
            # exposing paths, prompt content, credentials, or a Python traceback.
            error(400, "Claude transport stopped at the resource boundary")
        finally:
            self._serial.release()

    def __enter__(self) -> ClaudeTransport:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:
                gateway._handle(self)

        self._check()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # server_close joins request handlers. Closing the active provider socket
        # below interrupts an outstanding read; client header/body reads have a
        # five-second socket timeout rather than outliving this invocation.
        self._server.daemon_threads = False
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True,
                                        name="localpilot-claude-transport")
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        with self._connection_lock:
            connection = self._connection
            if connection is not None:
                # HTTP/1.0 getresponse detaches its socket from HTTPConnection;
                # retain that socket so cancellation also interrupts body reads.
                if self._provider_socket is not None:
                    try:
                        self._provider_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                connection.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def evidence(self) -> dict[str, Any]:
        result = {"provider_requests": self.requests, "protocol_retries": self.retries,
                  "rejected_responses": list(self.failures)}
        if self.resource_stop:
            result["resource_stop"] = self.resource_stop
        if getattr(self, "cleanup_error", ""):
            result["cleanup_error"] = self.cleanup_error
        return result
