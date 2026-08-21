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
