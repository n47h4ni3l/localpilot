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


def _snapshot(messages):
    return [dict(message) if isinstance(message, dict) else message for message in messages]


def test_reasoning_only_turn_continues_in_same_high_context_with_explicit_context(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    calls = []
    message_snapshots = []
    streams = iter(
        [
            [_chunk(thinking="private initial reasoning")],
            [
                _chunk(thinking="private second reasoning"),
                _chunk(content="I should answer after continuing my reasoning."),
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        message_snapshots.append(_snapshot(kwargs["messages"]))
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Think carefully and answer.")

    assert answer == "I should answer after continuing my reasoning."
    assert [call["think"] for call in calls] == ["high", "high"]
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert calls[0]["options"]["num_ctx"] == 32768
    assert calls[1]["options"]["num_ctx"] == 32768
    assert calls[1]["options"]["num_predict"] == 6144
    assert "private initial reasoning" in str(message_snapshots[1])
    assert "private initial reasoning" not in str(agent.messages)
    assert "private second reasoning" not in str(agent.messages)


def test_streamed_content_is_accumulated_without_exposing_thinking(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    captured = []

    def fake_chat(**kwargs):
        captured.append(kwargs)
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
    assert captured[0]["options"]["num_ctx"] == 32768
    assert "private one" not in answer
    assert "private one" not in str(agent.messages)


def test_ordinary_question_is_not_forced_to_use_tools(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk(content="Four.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("What is two plus two?")

    assert answer == "Four."
    assert len(calls) == 1


def test_model_decline_is_delivered_verbatim(tmp_path, monkeypatch):
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
        "DECLINE: the requested fact cannot be verified from available evidence."
    )


def test_empty_same_context_high_answer_is_visible(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    streams = iter(
        [
            [_chunk(thinking="first private reasoning")],
            [_chunk(thinking="continued private reasoning")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Reason and answer.")

    assert answer == (
        "[LocalPilot completed a high same-context answer reasoning pass but returned no usable final answer.]"
    )
    assert "first private reasoning" not in str(agent.messages)
    assert "continued private reasoning" not in str(agent.messages)


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
    assert not any(
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


def test_guide_tool_observation_remains_in_same_context_without_answer_review(tmp_path, monkeypatch):
    config, agent = _agent(tmp_path)
    (tmp_path / "known.txt").write_text("verified repository evidence", encoding="utf-8")
    requests, snapshots = [], []
    draft = "  My model-authored reply remains unchanged.\n"
    responses = iter([
        [_chunk(thinking="private research reasoning", tool_calls=[_call(
            "list_repository_tree", {"path": ".", "depth": 1, "max_entries": 20},
        )])],
        [_chunk(thinking="private post-tool reasoning"), _chunk(content=draft)],
    ])
    def chat(**request):
        requests.append(request)
        snapshots.append(_snapshot(request["messages"]))
        return iter(next(responses))
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    assert agent.ask("Inspect the repository and give your own answer.") == draft
    assert len(requests) == 2
    assert [r["think"] for r in requests] == ["high", "high"]
    assert all(r["options"]["num_ctx"] == 32768 for r in requests)
    assert any(m.get("role") == "tool" and "known.txt" in str(m.get("content")) for m in snapshots[1])
    assert "private research reasoning" not in str(agent.messages)
    assert "private post-tool reasoning" not in str(agent.messages)
    assert agent.audit.latest("model_guide_first_answer_delivered")["draft_preserved_byte_for_byte"]
    assert agent.audit.latest("model_same_context_postvalidation_complete") is None
