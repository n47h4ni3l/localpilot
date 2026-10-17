from __future__ import annotations

import contextlib
import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from localpilot.claude_transport import ClaudeTransport, ClaudeTransportError, message_events, validate_message


MODEL = "nestra:p1.2"
REQUEST = {"model": MODEL, "stream": True, "max_tokens": 128,
           "messages": [{"role": "user", "content": "read fixture"}],
           "tools": [{"name": "Read", "input_schema": {"type": "object"}}]}


def _message(block=None):
    block = block or {"type": "text", "text": "done"}
    return {"id": "msg_fixture", "type": "message", "role": "assistant", "model": MODEL,
            "content": [block], "stop_reason": "tool_use" if block["type"] == "tool_use" else "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3}}


@contextlib.contextmanager
def _provider(responses):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(payload)
            raw = responses[min(len(observed) - 1, len(responses) - 1)]
            status, value = raw if isinstance(raw, tuple) else (200, raw)
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _call(gateway, payload=REQUEST, *, token=None, path="/v1/messages?beta=true"):
    req = Request(gateway.url + path, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + (token or gateway.token)})
    opener = build_opener(ProxyHandler({}))
    try:
        response = opener.open(req, timeout=4)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read(), response.headers.get("Content-Type")


def test_valid_named_tool_is_forwarded_as_canonical_sse_without_editing_arguments():
    block = {"type": "tool_use", "id": "tool_a", "name": "Read", "input": {"file_path": "ü.txt"}}
    with _provider([_message(block)]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            status, body, content_type = _call(gateway)
            assert status == 200
            assert content_type == "text/event-stream"
            events = [json.loads(row[6:]) for row in body.decode().splitlines() if row.startswith("data: ")]
            assert [row["type"] for row in events] == ["message_start", "content_block_start",
                    "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
            assert events[1]["content_block"]["name"] == "Read"
            assert json.loads(events[2]["delta"]["partial_json"]) == block["input"]
            assert observed[0] == dict(REQUEST, stream=False)
            assert gateway.evidence()["protocol_retries"] == 0


def test_missing_name_is_never_forwarded_and_one_pre_forward_retry_can_recover():
    bad = _message({"type": "tool_use", "id": "bad", "input": {"file_path": "secret"}})
    checks = []
    with _provider([bad, _message()]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3,
                             resource_guard=lambda: checks.append(True)) as gateway:
            status, body, _ = _call(gateway)
            assert status == 200 and b"tool_use" not in body and b"secret" not in body
            assert len(observed) == 2 and observed[0] == observed[1]
            assert gateway.retries == 1 and len(checks) >= 4
            evidence = gateway.evidence()["rejected_responses"]
            assert evidence[0]["forwarded"] is False and len(evidence[0]["sha256"]) == 64
            assert "secret" not in json.dumps(evidence)


def test_repeated_missing_name_fails_explicitly_before_any_stream_is_forwarded():
    bad = _message({"type": "tool_use", "id": "bad", "input": {}})
    with _provider([bad]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            status, body, content_type = _call(gateway)
            assert status == 400 and content_type == "application/json"
            assert b"tool_use is missing a name" in body
            assert b"no tool from the rejected response was forwarded" in body
            assert b"content_block_start" not in body and len(observed) == 2


@pytest.mark.parametrize("block", [
    {"type": "tool_use", "id": "t", "name": "", "input": {}},
    {"type": "tool_use", "id": "t", "name": "Read<|channel|>commentary", "input": {}},
    {"type": "tool_use", "id": "t", "name": "Bash", "input": {}},
    {"type": "tool_use", "id": "t", "name": "Read", "input": "{}"},
    {"type": "tool_use", "name": "Read", "input": {}},
])
def test_invalid_names_arguments_and_ids_are_not_guessed(block):
    with pytest.raises(ClaudeTransportError):
        validate_message(_message(block), REQUEST, MODEL)


def test_duplicate_tool_ids_model_mismatch_and_non_finite_json_reject():
    message = _message({"type": "tool_use", "id": "t", "name": "Read", "input": {}})
    message["content"] *= 2
    with pytest.raises(ClaudeTransportError, match="duplicate"):
        validate_message(message, REQUEST, MODEL)
    with pytest.raises(ClaudeTransportError, match="model"):
        validate_message(dict(_message(), model="claude-cloud"), REQUEST, MODEL)
    with _provider([b'{"bad":NaN}']) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            assert _call(gateway)[0] == 400
            assert len(observed) == 2


def test_authentication_wrong_model_unknown_endpoint_and_oversize_do_not_call_provider():
    with _provider([_message()]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3, max_request_bytes=1024) as gateway:
            assert _call(gateway, token="wrong")[0] == 401
            assert _call(gateway, dict(REQUEST, model="other"))[0] == 400
            assert _call(gateway, path="/other")[0] == 404
            assert _call(gateway, dict(REQUEST, messages=["x" * 2000]))[0] == 400
            assert not observed


def test_provider_http_error_is_preserved_without_protocol_retry():
    with _provider([(404, {"error": {"message": "model missing"}})]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            status, body, _ = _call(gateway)
            assert status == 404 and b"model missing" in body
            assert len(observed) == 1 and gateway.retries == 0


def test_absolute_proxy_request_target_and_fragment_are_rejected_before_provider():
    with _provider([_message()]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            for target in ("http://example.com/v1/messages", "/v1/messages#fragment"):
                connection = http.client.HTTPConnection("127.0.0.1", int(gateway.url.rsplit(":", 1)[1]))
                connection.request("POST", target, json.dumps(REQUEST), {
                    "Authorization": "Bearer " + gateway.token})
                response = connection.getresponse()
                assert response.status == 404
                response.read()
                connection.close()
            assert not observed


def test_resource_guard_can_stop_before_retry_without_forwarding_or_reissuing():
    holder = {}
    def guard():
        gateway = holder.get("gateway")
        if gateway is not None and gateway.failures:
            raise RuntimeError("memory pressure")
    bad = _message({"type": "tool_use", "id": "t", "input": {}})
    with _provider([bad]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3, resource_guard=guard) as gateway:
            holder["gateway"] = gateway
            assert _call(gateway)[0] == 400
            assert gateway.resource_stop == "RuntimeError"
            assert gateway.retries == 0 and len(observed) == 1


def test_response_size_guard_and_shared_retry_budget_are_bounded():
    with _provider([b"x" * 4096]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3, max_response_bytes=1000) as gateway:
            assert _call(gateway)[0] == 400
            assert len(observed) == 1
    bad = _message({"type": "tool_use", "id": "t", "input": {}})
    with _provider([bad, _message(), bad]) as (base_url, observed):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            assert _call(gateway)[0] == 200
            assert _call(gateway)[0] == 400
            assert len(observed) == 3 and gateway.retries == 1


def test_gateway_closes_listener_and_does_not_allow_cloud_or_redirect_endpoints():
    with _provider([_message()]) as (base_url, _):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            port = int(gateway.url.rsplit(":", 1)[1])
        with socket.socket() as connection:
            assert connection.connect_ex(("127.0.0.1", port)) != 0
    for url in ("https://localhost:11434", "http://ollama.com", "http://u:p@127.0.0.1", "http://127.0.0.1/other"):
        with pytest.raises(ValueError):
            ClaudeTransport(base_url=url, model=MODEL, timeout_seconds=3)


def test_shutdown_interrupts_blocked_http10_body_and_rejects_concurrent_provider_calls():
    started, finish = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "10000")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            started.set()
            finish.wait(5)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    server_thread.start()
    gateway = ClaudeTransport(base_url=f"http://127.0.0.1:{server.server_port}", model=MODEL, timeout_seconds=30)
    gateway.__enter__()
    result = []
    caller = threading.Thread(target=lambda: result.append(_call(gateway)), daemon=True)
    caller.start()
    try:
        assert started.wait(2)
        assert _call(gateway)[0] == 409
        before = time.monotonic()
        gateway.close()
        caller.join(timeout=1)
        assert time.monotonic() - before < 2
        assert not caller.is_alive() and gateway.requests == 1
    finally:
        finish.set()
        gateway.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


def test_nonstream_response_content_is_preserved_exactly_and_usage_is_not_fabricated():
    message = _message()
    with _provider([message]) as (base_url, _):
        with ClaudeTransport(base_url=base_url, model=MODEL, timeout_seconds=3) as gateway:
            status, body, content_type = _call(gateway, dict(REQUEST, stream=False))
            assert status == 200 and content_type == "application/json"
            assert json.loads(body) == message
    events = message_events(message).decode()
    assert '"output_tokens": 3' in events and '"input_tokens": 2' in events
