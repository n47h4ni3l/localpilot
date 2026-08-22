import pytest

from localpilot.selfdev import developer_chat


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


def test_streaming_chat_is_guarded_and_unloads_model():
    calls = []
    guard_calls = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        return iter(
            [
                {"message": {"content": "hello ", "tool_calls": []}},
                {"message": {"content": "world", "tool_calls": [{"function": {"name": "inspect"}}]}},
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
    assert len(response.message["tool_calls"]) == 1
    assert len(guard_calls) == 2
    assert calls[0]["stream"] is True
    assert calls[0]["keep_alive"] == 0
