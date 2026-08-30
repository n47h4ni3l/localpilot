import asyncio
from types import SimpleNamespace

import pytest

from localpilot.config import Config
from localpilot.selfdev import SelfDeveloper, developer_chat


def test_supported_model_keeps_thinking():
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return "ok"

    result = developer_chat(
        fake_chat,
        request_think=True,
        model="thinking-model",
        messages=[],
    )

    assert result == "ok"
    assert len(calls) == 1
    assert calls[0]["think"] is True


def test_explicit_reasoning_level_is_preserved():
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return "ok"

    result = developer_chat(
        fake_chat,
        request_think="high",
        model="gpt-oss:20b",
        messages=[],
    )

    assert result == "ok"
    assert calls[0]["think"] == "high"


def test_unsupported_model_retries_without_thinking():
    calls = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        if "think" in kwargs:
            raise RuntimeError("model does not support thinking")
        return "ok"

    result = developer_chat(
        fake_chat,
        request_think=True,
        model="qwen2.5:32b",
        messages=[],
    )

    assert result == "ok"
    assert len(calls) == 2
    assert calls[0]["think"] is True
    assert "think" not in calls[1]


def test_unrelated_model_error_is_not_hidden():
    def fake_chat(**kwargs):
        raise RuntimeError("GPU exploded")

    with pytest.raises(RuntimeError, match="GPU exploded"):
        developer_chat(
            fake_chat,
            request_think=True,
            model="model",
            messages=[],
        )


def test_thinking_can_be_omitted_explicitly():
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return "ok"

    developer_chat(
        fake_chat,
        request_think=False,
        model="qwen2.5:32b",
        messages=[],
    )

    assert "think" not in captured


def test_streaming_chat_is_guarded_unloads_model_and_preserves_transient_thinking():
    calls = []
    guard_calls = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        return iter(
            [
                {"message": {"thinking": "inspect ", "content": "hello ", "tool_calls": []}},
                {
                    "message": {
                        "thinking": "then act",
                        "content": "world",
                        "tool_calls": [{"function": {"name": "inspect"}}],
                    }
                },
            ]
        )

    response = developer_chat(
        fake_chat,
        request_think=False,
        keep_alive=0,
        stream_guard=lambda: guard_calls.append(True),
        model="safe-model",
        messages=[],
    )

    assert response.message["content"] == "hello world"
    assert response.message["thinking"] == "inspect then act"
    assert len(response.message["tool_calls"]) == 1
    assert len(guard_calls) == 2
    assert calls[0]["stream"] is True
    assert calls[0]["keep_alive"] == 0


def test_preemptible_developer_chat_checks_guard_before_first_model_chunk(monkeypatch):
    import ollama

    streams = []
    clients = []

    class BlockingStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    class FakeAsyncClient:
        def __init__(self):
            self.closed = False
            clients.append(self)

        async def chat(self, **kwargs):
            assert kwargs["stream"] is True
            stream = BlockingStream()
            streams.append(stream)
            return stream

        async def close(self):
            self.closed = True

    monkeypatch.setattr(ollama, "AsyncClient", FakeAsyncClient)
    guard_calls = []

    def foreground_guard():
        guard_calls.append(True)
        if len(guard_calls) >= 2:
            raise RuntimeError("foreground activity detected")

    with pytest.raises(RuntimeError, match="foreground activity detected"):
        developer_chat(
            lambda **_kwargs: pytest.fail("sync chat must not be used"),
            request_think=False,
            stream_guard=foreground_guard,
            preempt_before_first_chunk=True,
            guard_poll_seconds=0.01,
            model="safe-model",
            messages=[],
        )

    assert len(guard_calls) >= 2
    assert streams[0].closed is True
    assert clients[0].closed is True


def test_developer_chat_merges_explicit_context_into_existing_options():
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return "ok"

    developer_chat(
        fake_chat,
        request_think=False,
        context_tokens=16384,
        model="qwen2.5:32b",
        messages=[],
        options={"temperature": 0.2},
    )

    assert captured["options"]["temperature"] == 0.2
    assert captured["options"]["num_ctx"] == 16384


def test_live_selfdev_path_preserves_gpt_oss_high_and_context():
    developer = object.__new__(SelfDeveloper)
    developer.config = Config()
    developer.config.model.think = "high"
    developer.config.selfdev.context_tokens = 16384
    developer.config.selfdev.ollama_keep_alive = 0
    developer._check_resources = lambda *args, **kwargs: None
    calls = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        return iter([{"message": {"thinking": "careful", "content": "done", "tool_calls": []}}])

    response = developer._developer_chat(
        fake_chat,
        force=True,
        branch="candidate/test",
        model="gpt-oss:20b",
        messages=[],
        options={"temperature": 0.1},
    )

    assert response.message["content"] == "done"
    assert response.message["thinking"] == "careful"
    assert calls[0]["think"] == "high"
    assert calls[0]["options"]["num_ctx"] == 16384
    assert calls[0]["keep_alive"] == 0


def test_live_ollama_chat_enables_pre_first_chunk_preemption(monkeypatch):
    import localpilot.selfdev as selfdev_module

    developer = object.__new__(SelfDeveloper)
    developer.config = Config()
    developer._check_resources = lambda *args, **kwargs: None
    captured = {}

    def capture_developer_chat(_chat, **kwargs):
        captured.update(kwargs)
        return "ok"

    def default_ollama_chat(**_kwargs):
        return None

    default_ollama_chat.__module__ = "ollama._client"
    default_ollama_chat.__qualname__ = "Client.chat"
    monkeypatch.setattr(selfdev_module, "developer_chat", capture_developer_chat)

    result = developer._developer_chat(
        default_ollama_chat,
        force=False,
        branch="capability-discovery",
        model="qwen2.5:14b",
        messages=[],
    )

    assert result == "ok"
    assert captured["preempt_before_first_chunk"] is True
    assert callable(captured["stream_guard"])


def test_live_qwen_path_keeps_context_when_thinking_is_unsupported():
    developer = object.__new__(SelfDeveloper)
    developer.config = Config()
    developer.config.model.think = "high"
    developer.config.selfdev.context_tokens = 16384
    developer.config.selfdev.ollama_keep_alive = 0
    developer._check_resources = lambda *args, **kwargs: None
    calls = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        if "think" in kwargs:
            raise RuntimeError("model does not support thinking")
        return iter([{"message": {"content": "qwen result", "tool_calls": []}}])

    response = developer._developer_chat(
        fake_chat,
        force=True,
        branch="candidate/test",
        model="qwen2.5:32b",
        messages=[],
        options={"temperature": 0.1},
    )

    assert response.message["content"] == "qwen result"
    assert len(calls) == 2
    assert calls[0]["think"] is True
    assert "think" not in calls[1]
    assert calls[1]["options"]["num_ctx"] == 16384
