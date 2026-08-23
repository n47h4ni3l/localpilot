import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config


def _response(*, content: str = "", thinking: str = ""):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=[],
        )
    )


def _agent(tmp_path):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    return config, agent


def test_gpt_oss_chat_uses_high_reasoning_and_surfaces_no_final_answer(tmp_path, monkeypatch):
    config, agent = _agent(tmp_path)
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return _response(thinking="internal reasoning was produced")

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("What should you have done differently?")

    assert calls[0]["think"] == "high"
    assert answer == "[LocalPilot completed a high reasoning pass but returned no final answer.]"
    assert "internal reasoning was produced" not in answer


def test_truly_empty_model_response_retries_once_then_returns_answer(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    responses = iter(
        [
            _response(),
            _response(content="I should verify repository interfaces before using them."),
        ]
    )
    call_count = 0

    def fake_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        return next(responses)

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Review the failed implementation.")

    assert call_count == 2
    assert answer.startswith("I should verify repository interfaces")
    assert any(
        isinstance(message, dict)
        and "previous response contained neither" in str(message.get("content", ""))
        for message in agent.messages
    )


def test_truly_empty_model_response_is_visible_after_retry(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: _response()),
    )

    answer = agent.ask("Say something or explicitly remain silent.")

    assert answer == "[LocalPilot returned an empty response after one retry.]"
