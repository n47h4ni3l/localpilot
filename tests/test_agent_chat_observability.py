import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config


def _chunk(*, content: str = "", thinking: str = "", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=list(tool_calls or []),
        )
    )


def _call(name: str, arguments: dict | None = None):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=dict(arguments or {}))
    )


def _agent(tmp_path):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    return config, agent


def test_tool_investigation_is_synthesized_by_same_high_reasoning_model(tmp_path, monkeypatch):
    config, agent = _agent(tmp_path)
    (tmp_path / "known.txt").write_text("verified repository evidence", encoding="utf-8")
    calls = []
    streams = iter(
        [
            [
                _chunk(
                    thinking="private investigation reasoning",
                    tool_calls=[
                        _call(
                            "list_repository_tree",
                            {"path": ".", "depth": 1, "max_entries": 50},
                        )
                    ],
                )
            ],
            [_chunk(content="Hello! How can I help you today?")],
            [
                _chunk(thinking="private synthesis reasoning"),
                _chunk(content="The repository evidence shows known.txt exists."),
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the repository and tell me what you verified.")

    assert answer == "The repository evidence shows known.txt exists."
    assert [call["think"] for call in calls] == ["high", "high", "high"]
    assert calls[0]["stream"] is True
    assert "tools" in calls[0]
    assert "tools" in calls[1]
    assert "tools" not in calls[2]
    assert calls[2]["options"]["num_predict"] == 4096
    synthesis_text = str(calls[2]["messages"])
    assert "Inspect the repository and tell me what you verified." in synthesis_text
    assert "Finding 1" in synthesis_text
    assert "list_repository_tree" in synthesis_text
    assert "known.txt" in synthesis_text
    assert "Hello! How can I help you today?" not in str(agent.messages)
    assert "private investigation reasoning" not in str(agent.messages)
    assert "private synthesis reasoning" not in str(agent.messages)
    assert config.model.think == "high"


def test_reasoning_only_turn_uses_high_reasoning_synthesis_not_low_finalizer(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [_chunk(thinking="private initial reasoning")],
            [
                _chunk(thinking="private second reasoning"),
                _chunk(content="I should answer after reasoning over the request."),
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Think carefully and answer.")

    assert answer == "I should answer after reasoning over the request."
    assert [call["think"] for call in calls] == ["high", "high"]
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert "private initial reasoning" not in str(agent.messages)
    assert "private second reasoning" not in str(agent.messages)


def test_streamed_content_is_accumulated_without_exposing_thinking(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)

    def fake_chat(**kwargs):
        return iter(
            [
                _chunk(thinking="private one "),
                _chunk(thinking="private two"),
                _chunk(content="Visible "),
                _chunk(content="answer."),
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Answer after thinking.")

    assert answer == "Visible answer."
    assert "private one" not in answer
    assert "private one" not in str(agent.messages)


def test_decline_requires_a_specific_reason_and_is_visible(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(
            chat=lambda **kwargs: iter(
                [_chunk(content="DECLINE: the requested fact cannot be verified from available evidence.")]
            )
        ),
    )

    answer = agent.ask("Answer only if the fact is actually known.")

    assert answer == (
        "[LocalPilot chose not to answer: the requested fact cannot be verified from available evidence.]"
    )


def test_empty_high_synthesis_is_visible(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    streams = iter(
        [
            [_chunk(thinking="first private reasoning")],
            [_chunk(thinking="synthesis private reasoning")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Reason and answer.")

    assert answer == (
        "[LocalPilot completed a high evidence-synthesis reasoning pass over its own findings "
        "but returned no final answer.]"
    )
    assert "first private reasoning" not in str(agent.messages)
    assert "synthesis private reasoning" not in str(agent.messages)


def test_truly_empty_stream_retries_once_then_returns_answer(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    streams = iter(
        [
            [],
            [
                _chunk(content="I should verify repository interfaces "),
                _chunk(content="before using them."),
            ],
        ]
    )
    call_count = 0

    def fake_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Review the failed implementation.")

    assert call_count == 2
    assert answer == "I should verify repository interfaces before using them."
    assert any(
        isinstance(message, dict)
        and "previous response contained neither" in str(message.get("content", ""))
        for message in agent.messages
    )


def test_truly_empty_stream_is_visible_after_retry(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(())),
    )

    answer = agent.ask("Say something or explicitly remain silent.")

    assert answer == "[LocalPilot returned an empty response after one retry.]"
