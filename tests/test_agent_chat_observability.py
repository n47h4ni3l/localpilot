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


def test_tool_investigation_continues_in_same_high_reasoning_context(tmp_path, monkeypatch):
    config, agent = _agent(tmp_path)
    (tmp_path / "known.txt").write_text("verified repository evidence", encoding="utf-8")
    calls = []
    message_snapshots = []
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
            [
                _chunk(thinking="private post-tool reasoning"),
                _chunk(content="Hello! How can I help you today?"),
            ],
            [
                _chunk(thinking="private final reasoning"),
                _chunk(content="The repository evidence shows known.txt exists."),
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        message_snapshots.append(_snapshot(kwargs["messages"]))
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the repository and tell me what you verified.")

    assert answer == "The repository evidence shows known.txt exists."
    assert [call["think"] for call in calls] == ["high", "high", "high"]
    assert all(call["stream"] is True for call in calls)
    assert all("tools" in call for call in calls)
    assert all(call["options"]["num_ctx"] == 32768 for call in calls)

    final_context = message_snapshots[2]
    assert any(
        message.get("role") == "tool" and "known.txt" in str(message.get("content"))
        for message in final_context
    )
    assert any(
        message.get("role") == "user"
        and "decide whether you have enough verified evidence" in str(message.get("content"))
        for message in final_context
    )
    assert any(
        message.get("role") == "user"
        and "Inspect the repository and tell me what you verified." in str(message.get("content"))
        for message in final_context
    )
    assert "Finding 1" not in str(final_context)
    assert "Hello! How can I help you today?" not in str(final_context)
    assert "private post-tool reasoning" in str(final_context)

    assert "Hello! How can I help you today?" not in str(agent.messages)
    assert "private investigation reasoning" not in str(agent.messages)
    assert "private post-tool reasoning" not in str(agent.messages)
    assert "private final reasoning" not in str(agent.messages)
    assert "decide whether you have enough verified evidence" not in str(agent.messages)
    assert config.model.think == "high"


def test_explicit_repository_and_github_request_cannot_finish_before_both_are_attempted(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    (tmp_path / "known.txt").write_text("verified", encoding="utf-8")
    github_spec = agent.tools["get_github_pull_request"]
    agent.tools["get_github_pull_request"] = SimpleNamespace(
        risk=github_spec.risk,
        fn=lambda number: f"Private GitHub PR #{number} verified",
    )
    calls = []
    message_snapshots = []
    streams = iter(
        [
            [_chunk(content="I cannot inspect the repository because I have no evidence or access.")],
            [
                _chunk(
                    thinking="I should use both evidence sources.",
                    tool_calls=[
                        _call(
                            "list_repository_tree",
                            {"path": ".", "depth": 1, "max_entries": 50},
                        ),
                        _call("get_github_pull_request", {"number": 30}),
                    ],
                )
            ],
            [_chunk(content="I found the requested evidence.")],
            [_chunk(content="Verified: known.txt exists and private GitHub PR #30 was inspected.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        message_snapshots.append(_snapshot(kwargs["messages"]))
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    prompt = (
        "Inspect your actual local repository and your private GitHub repository yourself. "
        "Review PR #30 and use only interfaces you personally verify exist."
    )
    answer = agent.ask(prompt)

    assert answer == "Verified: known.txt exists and private GitHub PR #30 was inspected."
    assert len(calls) == 4
    assert all("tools" in call for call in calls)
    assert "you have not yet attempted" in str(message_snapshots[1]).lower()
    assert "trusted repository" in str(message_snapshots[1]).lower()
    assert "private github" in str(message_snapshots[1]).lower()
    assert any(
        message.get("role") == "tool" and "Private GitHub PR #30 verified" in str(message.get("content"))
        for message in message_snapshots[2]
    )
    assert "I cannot inspect the repository" not in str(agent.messages)
    assert "you have not yet attempted" not in str(agent.messages).lower()


def test_wrong_tool_does_not_satisfy_required_repository_evidence(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [
                _chunk(
                    tool_calls=[_call("get_system_summary")]
                )
            ],
            [_chunk(content="I used a tool, so I can answer now.")],
            [
                _chunk(
                    tool_calls=[
                        _call(
                            "list_repository_tree",
                            {"path": ".", "depth": 1, "max_entries": 50},
                        )
                    ]
                )
            ],
            [_chunk(content="The repository was actually inspected.")],
            [_chunk(content="Final grounded answer.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the actual LocalPilot repository before answering.")

    assert answer == "Final grounded answer."
    assert len(calls) == 5
    assert any(
        event.get("role") == "tool" and event.get("tool_name") == "list_repository_tree"
        for event in agent.messages
        if isinstance(event, dict)
    )


def test_post_tool_review_can_choose_to_collect_more_evidence(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    (tmp_path / "known.txt").write_text("verified repository evidence", encoding="utf-8")
    calls = []
    streams = iter(
        [
            [
                _chunk(
                    tool_calls=[
                        _call(
                            "list_repository_tree",
                            {"path": ".", "depth": 1, "max_entries": 50},
                        )
                    ]
                )
            ],
            [_chunk(content="I have only partial evidence so far.")],
            [
                _chunk(
                    thinking="I need the file contents too.",
                    tool_calls=[
                        _call(
                            "read_repository_file",
                            {"path": "known.txt", "start_line": 1, "end_line": 10},
                        )
                    ],
                )
            ],
            [_chunk(content="I verified both the file's existence and its contents.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the repository and verify known.txt from evidence.")

    assert answer == "I verified both the file's existence and its contents."
    assert len(calls) == 4
    assert all("tools" in call for call in calls)
    assert any(
        message.get("role") == "tool" and "verified repository evidence" in str(message.get("content"))
        for message in agent.messages
        if isinstance(message, dict)
    )


def test_repository_evidence_requirement_fails_visibly_after_two_ignored_recoveries(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk(content="I still cannot inspect it.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the actual LocalPilot repository and review PR #30.")

    assert len(calls) == 3
    assert "did not attempt the relevant available read-only source" in answer
    assert "I still cannot inspect it." not in str(agent.messages)


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
    assert calls[1]["options"]["num_predict"] == 4096
    assert "private initial reasoning" in str(message_snapshots[1])
    assert "private initial reasoning" not in str(agent.messages)
    assert "private second reasoning" not in str(agent.messages)


def test_generic_reset_during_forced_final_answer_gets_one_same_context_retry(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [_chunk(thinking="private initial reasoning")],
            [_chunk(content="Hello! How can I help you today?")],
            [_chunk(content="Here is the answer to the original request.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Think carefully and answer this question.")

    assert answer == "Here is the answer to the original request."
    assert len(calls) == 3
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert "tools" not in calls[2]
    assert "Hello! How can I help you today?" not in str(agent.messages)


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
