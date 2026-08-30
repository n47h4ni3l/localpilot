import json
import sys
from types import SimpleNamespace

import pytest
from ollama import ResponseError

from localpilot.agent import LocalPilotAgent, _PUBLIC_WEB_FETCHES_PER_TURN
from localpilot.audit import AuditLog
from localpilot.config import Config
from localpilot.research import (
    MAX_ARGUMENT_DEPTH,
    MAX_CHECKPOINT_TEXT,
    MAX_EVIDENCE_REFS,
    RESEARCH_NOTEBOOK_TOOL,
    TransientResearchNotebook,
    research_notebook_tool_schema,
)


def _chunk(*, content: str = "", thinking: str = "", tool_calls=None, **runtime):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=list(tool_calls or []),
        ),
        **runtime,
    )


def _call(name: str, arguments: dict | None = None):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=dict(arguments or {}))
    )


def _agent(tmp_path, *, soft=2, hard=4):
    config = Config()
    config.agent.research_soft_tool_rounds = soft
    config.agent.research_hard_tool_rounds = hard
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    return config, agent


def test_operational_self_status_uses_passive_evidence_without_memory_or_tools(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    agent.audit.write(
        "background_worker_cycle_end",
        pid=717,
        sequence=9,
        status="deferred",
        duration_seconds=0.01,
    )
    agent.audit.write(
        "evolve_run_end",
        invocation_id="evolve-9",
        status="candidate_created",
        branch="selfdev/focused-change",
        checks_passed=True,
        summary="Created a focused candidate.",
    )
    monkeypatch.setattr(
        agent,
        "_learning_context",
        lambda _prompt: pytest.fail("operational status must not retrieve semantic memory"),
    )
    calls = []
    snapshots = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        snapshots.append([dict(message) for message in kwargs["messages"]])
        return iter([_chunk(content="The worker is active; its latest cycle deferred. No model weights changed.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "LocalPilot, what is your current branch and commit, why did your runtime restart, and what "
        "have you learned or changed through autonomous evolution progress?"
    )

    assert "latest cycle deferred" in answer
    assert len(calls) == 1
    assert calls[0]["think"] == "low"
    assert "tools" not in calls[0]
    context = str(snapshots[0])
    assert "OPERATIONAL SELF-STATUS ROUTE" in context
    assert "selfdev/focused-change" in context
    assert "model_weights_changed_by_localpilot" in context
    assert "a restart only loads code" in context
    assert "OPERATIONAL SELF-STATUS ROUTE" not in str(agent.messages)
    assert agent.audit.latest("model_operational_self_status_route")[
        "durable_memory_retrieval_skipped"
    ] is True


def test_operational_status_classifier_covers_owner_handover_and_autonomy_questions():
    for prompt in (
        "LocalPilot, give me a handover of what is stable, blocked, and the next decision.",
        "What can you actually do autonomously toward becoming more capable, what is blocking you, and what still requires me?",
    ):
        assert LocalPilotAgent._is_operational_self_status_prompt(prompt) is True


def test_operational_status_context_exposes_real_learning_and_authority_boundaries(tmp_path):
    _, agent = _agent(tmp_path)
    agent.memory.record_human_lesson(
        "Keep verified facts separate from judgment.",
        topic="communication",
        source="owner",
    )

    context = agent._operational_self_status_context()

    assert '"human_lessons":1' in context
    assert '"model_weights_changed_by_localpilot":false' in context
    assert '"candidate_workspace_writes_allowed":true' in context
    assert '"automatic_merge_or_promotion_allowed":false' in context
    assert '"public_web_research_available"' in context


def test_friendly_planning_prompt_is_direct_low_reasoning_without_memory_or_tools(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    monkeypatch.setattr(
        agent,
        "_learning_context",
        lambda _prompt: pytest.fail("direct conversation must not retrieve semantic memory"),
    )
    calls = []
    snapshots = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        snapshots.append([dict(message) for message in kwargs["messages"]])
        return iter([_chunk(content="Start with the two decisions that unblock the rest.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Help me plan a realistic plan for the weekend and prioritize the work.")

    assert answer.startswith("Start with")
    assert calls[0]["think"] == "low"
    assert "tools" not in calls[0]
    assert "DIRECT CONVERSATION ROUTE" in str(snapshots[0])
    assert "DIRECT CONVERSATION ROUTE" not in str(agent.messages)
    assert agent.audit.latest("model_direct_conversation_route") is not None


def test_friendly_personal_advice_skips_systemsense_memory_and_tools(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    monkeypatch.setattr(
        agent,
        "_learning_context",
        lambda _prompt: pytest.fail("friendly advice must not retrieve semantic memory"),
    )
    monkeypatch.setattr(
        agent.systemsense,
        "compact_context",
        lambda: pytest.fail("friendly advice must not receive passive PC telemetry"),
    )
    calls = []
    snapshots = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        snapshots.append([dict(message) for message in kwargs["messages"]])
        return iter(
            [
                _chunk(
                    content=(
                        "Put your phone in another room and make a cup of tea; the small ritual "
                        "gives the evening a clean edge without turning relaxation into another project."
                    )
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "I'm heading away for the weekend and want to switch off properly. Talk to me like a "
        "friend: suggest one small, ordinary thing I could do tonight that might help."
    )

    assert answer.startswith("Put your phone")
    assert calls[0]["think"] == "low"
    assert "tools" not in calls[0]
    context = str(snapshots[0])
    assert "DIRECT CONVERSATION ROUTE" in context
    assert "PC maintenance" in context
    assert "storage_pressure" not in context


def test_ordinary_conversational_question_skips_research_memory_systemsense_and_tools(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    monkeypatch.setattr(
        agent,
        "_learning_context",
        lambda _prompt: pytest.fail("ordinary conversation must not retrieve semantic memory"),
    )
    monkeypatch.setattr(
        agent.systemsense,
        "compact_context",
        lambda: pytest.fail("ordinary conversation must not receive passive PC telemetry"),
    )
    calls = []
    snapshots = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        snapshots.append([dict(message) for message in kwargs["messages"]])
        return iter(
            [
                _chunk(
                    content=(
                        "I think rain makes the outside world feel temporarily farther away, while tea and a book "
                        "give that smaller indoor world a little warmth and shape."
                    )
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "I've always liked the feeling of a rainy afternoon. Why do you think it makes some people want to "
        "read or make tea? Keep it conversational."
    )

    assert answer.startswith("I think rain")
    assert calls[0]["think"] == "low"
    assert "tools" not in calls[0]
    assert "DIRECT CONVERSATION ROUTE" in str(snapshots[0])


def test_casual_conversation_rejects_failed_evidence_search_deflection(tmp_path):
    _, agent = _agent(tmp_path)

    issues = agent._response_behavior_issues(
        "Why do rainy afternoons make some people want tea? Keep it conversational.",
        "I couldn't find any evidence. The library search didn't turn up relevant passages.",
    )

    assert "casual_conversation_replaced_by_evidence_search" in issues


def test_generic_self_maintenance_does_not_replace_invited_ordinary_topic(tmp_path):
    _, agent = _agent(tmp_path)

    issues = agent._response_behavior_issues(
        "How are you? Tell me one ordinary thing you find interesting right now.",
        "I'm keeping my circuits cool and my files tidy!",
    )

    assert "ordinary_interest_invitation_unanswered" in issues


def test_pc_maintenance_does_not_replace_friendly_personal_advice(tmp_path):
    _, agent = _agent(tmp_path)

    issues = agent._response_behavior_issues(
        "Talk to me like a friend and suggest one small thing to help me switch off tonight.",
        "Run a quick disk cleanup to clear temp files and lower the storage pressure.",
    )

    assert "friendly_personal_advice_replaced_by_pc_maintenance" in issues


def test_operational_status_rejects_restart_code_conflation_and_invented_worker_tasks(tmp_path):
    _, agent = _agent(tmp_path)

    issues = agent._response_behavior_issues(
        "What have you learned or changed since we started, and what will your background worker do?",
        "The only change was a runtime restart. No new code was introduced, so it is the same code as "
        "before. The worker will do background reading, health checks, and housekeeping.",
    )

    assert "runtime_restart_conflated_with_code_change" in issues
    assert "unverified_background_worker_task_examples" in issues


def test_operational_status_recovers_from_restart_conflation_without_inventing_worker_tasks(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [
                _chunk(
                    content=(
                        "The only change was a runtime restart and no new code was introduced. "
                        "The worker will do background reading, health checks, and housekeeping."
                    )
                )
            ],
            [
                _chunk(
                    content=(
                        "The current commit identifies the code loaded now. The passive evidence cannot compare "
                        "it with the earlier session; the restart loaded code but was not itself a code change or "
                        "learning. No model weights changed. The latest worker cycle was deferred because the PC "
                        "was in use; its polling interval is not evidence that autonomous work ran."
                    )
                )
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "Tell me what you learned or changed since we started, what was a code change versus a runtime "
        "restart, and what your background worker will do."
    )

    assert "cannot compare it with the earlier session" in answer
    assert "latest worker cycle was deferred" in answer
    assert len(calls) == 2
    recovery = agent.audit.latest("model_same_context_behavior_recovery_complete")
    assert recovery["original_issues"] == [
        "runtime_restart_conflated_with_code_change",
        "unverified_background_worker_task_examples",
    ]
    assert recovery["accepted"] is True
    recovery_context = str(calls[-1]["messages"])
    assert "passive snapshot cannot compare" in recovery_context
    assert "do not invent task categories" in recovery_context


def test_model_can_request_more_evidence_after_post_tool_review(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=3, hard=5)
    (tmp_path / "known.txt").write_text("verified detail", encoding="utf-8")
    calls = []
    streams = iter(
        [
            [_chunk(tool_calls=[_call("list_repository_tree", {"path": ".", "depth": 1})])],
            [_chunk(content="I have some evidence, but I should review whether it is enough.")],
            [_chunk(thinking="I need the file body.", tool_calls=[_call("read_repository_file", {"path": "known.txt", "start_line": 1, "end_line": 5})])],
            [_chunk(content="Verified answer from the tree and file body.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Inspect the repository and verify known.txt from direct evidence.")

    assert answer == "Verified answer from the tree and file body."
    assert len(calls) == 4
    assert all("tools" in item for item in calls)
    assert any(
        message.get("role") == "tool" and "verified detail" in str(message.get("content"))
        for message in agent.messages
        if isinstance(message, dict)
    )


def test_repeated_zero_information_searches_stop_early_and_synthesize(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=12, hard=24)
    original = agent.tools["search_public_web"]
    invocations = 0

    def empty_search(**kwargs):
        nonlocal invocations
        invocations += 1
        return "Public web search: query\nNo bounded HTTPS results were found."

    agent.tools["search_public_web"] = SimpleNamespace(
        risk=original.risk,
        fn=empty_search,
    )
    streams = iter(
        [
            [_chunk(tool_calls=[_call("search_public_web", {"query": "semantic memory"})])],
            [_chunk(tool_calls=[_call("search_public_web", {"query": "knowledge retrieval"})])],
            [_chunk(content="I should stop the empty search path and synthesize the unresolved result.")],
            [_chunk(content="The two searches produced no usable sources, so I cannot establish the research claim. My current hypothesis remains unresolved rather than something I should decorate with invented facts.")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )
    answer = agent.ask("Research a useful topic on the public web and form a view.")

    assert "no usable sources" in answer
    assert invocations == 2
    stagnation = agent.audit.latest("model_research_stagnation_adaptation")
    assert stagnation["tools"] == ["search_public_web"]
    assert agent.audit.latest("model_research_soft_budget") is None


def test_public_web_fetches_have_a_source_limit_below_the_general_hard_budget(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=12, hard=24)
    original = agent.tools["fetch_public_https"]
    invocations = 0

    def successful_fetch(**kwargs):
        nonlocal invocations
        invocations += 1
        return f"Public HTTPS source: {kwargs['url']}\nVerified source text."

    agent.tools["fetch_public_https"] = SimpleNamespace(
        risk=original.risk,
        fn=successful_fetch,
    )
    tool_turns = [
        [_chunk(tool_calls=[_call("fetch_public_https", {"url": f"https://example.com/{index}"})])]
        for index in range(_PUBLIC_WEB_FETCHES_PER_TURN + 1)
    ]
    streams = iter(
        [
            *tool_turns,
            [_chunk(content="I think the inspected sources are sufficient; anything beyond them remains uncertain.")],
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Research this topic on the public web and form a provisional view.")

    assert answer.startswith("I think the inspected sources are sufficient")
    assert invocations == _PUBLIC_WEB_FETCHES_PER_TURN
    limit = agent.audit.latest("model_public_web_fetch_limit")
    assert limit is not None
    assert limit["fetches"] == limit["limit"] == _PUBLIC_WEB_FETCHES_PER_TURN


def test_hard_research_ceiling_blocks_remembered_tool_call(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=1, hard=1)
    (tmp_path / "known.txt").write_text("secret evidence", encoding="utf-8")
    read_calls = 0
    original = agent.tools["read_repository_file"]

    def counted_read(**kwargs):
        nonlocal read_calls
        read_calls += 1
        return original.fn(**kwargs)

    agent.tools["read_repository_file"] = SimpleNamespace(risk=original.risk, fn=counted_read)
    calls = []
    streams = iter(
        [
            [_chunk(tool_calls=[_call("list_repository_tree", {"path": ".", "depth": 1})])],
            [_chunk(thinking="I want one more file.", tool_calls=[_call("read_repository_file", {"path": "known.txt"})])],
            [_chunk(content="I can answer from the evidence I already have; the file body remains unresolved.")],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Inspect the repository and report what you can verify.")

    assert "file body remains unresolved" in answer
    assert read_calls == 0
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert "tools" not in calls[2]
    # The blocked remembered call is transient. Only the legitimate first tool call remains in history.
    persisted_calls = [
        message for message in agent.messages
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(persisted_calls) == 1
    assert "read_repository_file" not in str(persisted_calls[0].get("tool_calls"))
    assert not any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and "hard research safety ceiling" in str(message.get("content"))
        for message in agent.messages
    )


def test_hard_ceiling_cannot_bypass_missing_required_evidence(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=1, hard=1)
    gh = agent.tools["get_github_pull_request"]
    invocations = 0

    def failed_read(number):
        nonlocal invocations
        invocations += 1
        return "GitHub read failed: temporary auth error"

    agent.tools["get_github_pull_request"] = SimpleNamespace(risk=gh.risk, fn=failed_read)
    streams = iter(
        [
            [_chunk(tool_calls=[_call("get_github_pull_request", {"number": 30})])],
            [_chunk(thinking="I still want the PR.", tool_calls=[_call("get_github_pull_request", {"number": 30})])],
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Review PR #30 from private GitHub before answering.")

    assert "hard research ceiling" in answer
    assert "private GitHub" in answer
    assert invocations == 1
    persisted_calls = [
        message for message in agent.messages
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(persisted_calls) == 1


def test_identical_read_only_observation_is_not_executed_twice(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=3, hard=5)
    invocations = 0
    original = agent.tools["list_repository_tree"]

    def counted_tree(**kwargs):
        nonlocal invocations
        invocations += 1
        return original.fn(**kwargs)

    agent.tools["list_repository_tree"] = SimpleNamespace(risk=original.risk, fn=counted_tree)
    repeated = _call("list_repository_tree", {"path": ".", "depth": 1, "max_entries": 200})
    streams = iter(
        [
            [_chunk(tool_calls=[repeated])],
            [_chunk(content="I should review the evidence first.")],
            [_chunk(tool_calls=[repeated])],
            [_chunk(content="The duplicate observation added nothing; I can answer now.")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )
    answer = agent.ask("Inspect the repository and summarize it.")

    assert answer.startswith("The duplicate observation")
    assert invocations == 1
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and "duplicate call produced no new evidence" in str(message.get("content"))
        for message in agent.messages
    )


def test_failed_required_evidence_is_not_treated_as_success(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=3, hard=5)
    gh = agent.tools["get_github_pull_request"]
    outcomes = iter(["GitHub read failed: temporary auth error", "Private GitHub PR #30 verified"])
    agent.tools["get_github_pull_request"] = SimpleNamespace(
        risk=gh.risk,
        fn=lambda number: next(outcomes),
    )
    streams = iter(
        [
            [_chunk(tool_calls=[_call("get_github_pull_request", {"number": 30})])],
            [_chunk(content="I should not answer from a failed read.")],
            [_chunk(tool_calls=[_call("get_github_pull_request", {"number": 30})])],
            [_chunk(content="I now have successful PR evidence.")],
            [_chunk(content="Verified PR #30 after a successful read.")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )
    answer = agent.ask("Review PR #30 from private GitHub before answering.")

    assert answer == "Verified PR #30 after a successful read."


def test_audit_keeps_token_metrics_but_redacts_credentials(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.write(
        "runtime",
        context_tokens=32768,
        prompt_token_count=1234,
        github_token="do-not-store",
        api_key="also-secret",
    )
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))

    assert row["context_tokens"] == 32768
    assert row["prompt_token_count"] == 1234
    assert row["github_token"] == "<redacted>"
    assert row["api_key"] == "<redacted>"


def _checkpoint_args(
    *,
    proposed_path: str,
    evidence_refs: list[str] | None = None,
    unresolved_fact: str = "A second raw source may materially change the current conclusion.",
):
    return {
        "evidence_refs": list(evidence_refs or ["result-001"]),
        "unresolved_fact": unresolved_fact,
        "proposed_tool": "read_repository_file",
        "proposed_arguments": {"path": proposed_path, "start_line": 1, "end_line": 20},
        "result_that_would_change_the_conclusion": "A conflicting canonical value would require reporting the conflict.",
        "new_hypothesis": "",
    }


def test_compact_checkpoint_schema_and_semantic_redundancy_require_distinct_hypothesis():
    schema = research_notebook_tool_schema()["function"]["parameters"]
    assert set(schema["properties"]) == {
        "evidence_refs",
        "unresolved_fact",
        "proposed_tool",
        "proposed_arguments",
        "result_that_would_change_the_conclusion",
        "new_hypothesis",
    }
    assert schema["properties"]["proposed_arguments"]["type"] == "object"
    assert schema["properties"]["evidence_refs"]["maxItems"] == MAX_EVIDENCE_REFS
    assert "proposed_arguments_json" not in str(schema)

    notebook = TransientResearchNotebook(allowed_tools={"search_repository"})
    first = notebook.add_observation(
        tool="search_repository",
        arguments={"query": "model-adaptation-lab"},
        ok=True,
    )
    unsupported = {
        "evidence_refs": ["result-999"],
        "unresolved_fact": "Alternate spelling may find a distinct path.",
        "proposed_tool": "search_repository",
        "proposed_arguments": {"query": "model adaptation lab"},
        "result_that_would_change_the_conclusion": "A new implementation path would change the conclusion.",
    }

    rejected = notebook.submit(unsupported)

    assert rejected.accepted is False
    assert "unknown current-turn ID" in rejected.message

    redundant = dict(unsupported)
    redundant["evidence_refs"] = [first.result_id]

    needs_justification = notebook.submit(redundant)

    assert needs_justification.accepted is False
    assert needs_justification.redundant_with == (first.observation_id,)
    assert "distinct new_hypothesis" in needs_justification.message

    redundant["new_hypothesis"] = "Underscore normalization may expose a path omitted by the hyphenated query."
    justified = notebook.submit(redundant)

    assert justified.accepted is True
    assert notebook.authorizes("search_repository", {"query": "model adaptation lab"})

    next_turn = TransientResearchNotebook(start_at=2)
    second = next_turn.add_observation(tool="search_repository", arguments={"query": "fresh"}, ok=True)
    stale = dict(redundant)
    stale["proposed_arguments"] = {"query": "different"}
    stale_decision = next_turn.submit(stale)
    assert second.observation_id == "obs-002"
    assert stale_decision.accepted is False
    assert "unknown current-turn ID" in stale_decision.message


def test_checkpoint_rejects_non_bare_refs_and_oversized_repetitive_or_nested_payloads_compactly():
    notebook = TransientResearchNotebook(allowed_tools={"read_repository_file"})
    observation = notebook.add_observation(
        tool="read_repository_file", arguments={"path": "known.txt"}, ok=True
    )
    base = _checkpoint_args(proposed_path="next.txt", evidence_refs=[observation.result_id])

    json_string = dict(base)
    json_string["proposed_arguments"] = json.dumps(base["proposed_arguments"])
    assert "must be an object" in notebook.submit(json_string).message

    prose_ref = dict(base)
    prose_ref["evidence_refs"] = ["result-001: copied prose and a long observation history"]
    assert "only bare" in notebook.submit(prose_ref).message

    repeated = dict(base)
    repeated["evidence_refs"] = [observation.result_id] * (MAX_EVIDENCE_REFS + 1)
    repeated_result = notebook.submit(repeated)
    assert repeated_result.accepted is False
    assert len(repeated_result.message) < 140
    assert observation.result_id not in repeated_result.message

    oversized = dict(base)
    oversized["unresolved_fact"] = "x" * (MAX_CHECKPOINT_TEXT + 1)
    oversized_result = notebook.submit(oversized)
    assert oversized_result.accepted is False
    assert len(oversized_result.message) < 140
    assert "x" * 20 not in oversized_result.message

    nested: dict = {"value": "leaf"}
    for _ in range(MAX_ARGUMENT_DEPTH + 1):
        nested = {"nested": nested}
    too_deep = dict(base)
    too_deep["proposed_arguments"] = nested
    assert "exceeds depth" in notebook.submit(too_deep).message


def test_post_soft_unique_observation_requires_matching_information_gain_checkpoint(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=1, hard=4)
    (tmp_path / "first.txt").write_text("first raw fact", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second raw fact", encoding="utf-8")
    second_reads = 0
    original = agent.tools["read_repository_file"]

    def counted_read(**kwargs):
        nonlocal second_reads
        if kwargs.get("path") == "second.txt":
            second_reads += 1
        return original.fn(**kwargs)

    agent.tools["read_repository_file"] = SimpleNamespace(risk=original.risk, fn=counted_read)
    streams = iter(
        [
            [_chunk(tool_calls=[_call("read_repository_file", {"path": "first.txt"})])],
            [_chunk(tool_calls=[_call("read_repository_file", {"path": "second.txt"})])],
            [_chunk(tool_calls=[_call(RESEARCH_NOTEBOOK_TOOL, _checkpoint_args(proposed_path="second.txt"))])],
            [
                _chunk(
                    tool_calls=[
                        _call(
                            "read_repository_file",
                            {"path": "second.txt", "start_line": 1, "end_line": 20},
                        )
                    ]
                )
            ],
            [_chunk(content="The two raw facts were inspected.")],
            [_chunk(content="Final answer grounded in both raw results.")],
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Inspect the repository and reconcile first.txt with second.txt.")

    assert answer == "Final answer grounded in both raw results."
    assert second_reads == 1
    required = agent.audit.latest("model_research_checkpoint_required")
    accepted = agent.audit.latest("model_research_checkpoint")
    assert required is not None and required["unique_call_count"] == 1
    assert accepted is not None and accepted["accepted"] is True
    assert accepted["evidence_ref_count"] == 1


def test_exact_duplicate_cache_still_bypasses_post_soft_checkpoint(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=1, hard=4)
    invocations = 0
    original = agent.tools["list_repository_tree"]

    def counted_tree(**kwargs):
        nonlocal invocations
        invocations += 1
        return original.fn(**kwargs)

    agent.tools["list_repository_tree"] = SimpleNamespace(risk=original.risk, fn=counted_tree)
    repeated = _call("list_repository_tree", {"path": ".", "depth": 1, "max_entries": 200})
    streams = iter(
        [
            [_chunk(tool_calls=[repeated])],
            [_chunk(tool_calls=[repeated])],
            [_chunk(content="The exact duplicate reused the earlier result.")],
            [_chunk(content="Final cache-preserving answer.")],
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda **kwargs: iter(next(streams))),
    )

    answer = agent.ask("Inspect the repository and report the tree.")

    assert answer == "Final cache-preserving answer."
    assert invocations == 1
    assert agent.audit.latest("tool_observation_cache_hit") is not None
    assert agent.audit.latest("model_research_checkpoint_required") is None


def test_misleading_notebook_cannot_replace_raw_evidence_and_late_result_reaches_final_synthesis(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=1, hard=2)
    (tmp_path / "canonical.txt").write_text("CANONICAL VALUE: blue", encoding="utf-8")
    (tmp_path / "late.txt").write_text("LATE RAW RESULT: blue is corroborated", encoding="utf-8")
    snapshots = []
    misleading = _checkpoint_args(
        proposed_path="late.txt",
        unresolved_fact="MISLEADING CHECKPOINT CLAIM: the canonical value may be red.",
    )
    streams = iter(
        [
            [_chunk(tool_calls=[_call("read_repository_file", {"path": "canonical.txt"})])],
            [_chunk(tool_calls=[_call(RESEARCH_NOTEBOOK_TOOL, misleading)])],
            [
                _chunk(
                    tool_calls=[
                        _call(
                            "read_repository_file",
                            {"path": "late.txt", "start_line": 1, "end_line": 20},
                        )
                    ]
                )
            ],
            [_chunk(thinking="I should now reconcile the raw results in the same live context.")],
            [_chunk(content="The canonical value is blue; the late raw result corroborates it.")],
        ]
    )

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect the repository and report the canonical value using all raw evidence.")

    assert answer == "The canonical value is blue; the late raw result corroborates it."
    final_context = str(snapshots[-1])
    assert "CANONICAL VALUE: blue" in final_context
    assert "LATE RAW RESULT: blue is corroborated" in final_context
    assert "MISLEADING CHECKPOINT CLAIM" not in final_context
    assert "TRANSIENT RESEARCH CHECKPOINT" not in final_context
    assert "update_research_notebook" not in final_context
    assert "advisory research soft budget" not in final_context
    assert "MISLEADING CHECKPOINT CLAIM" not in str(agent.messages)
    assert "MISLEADING CHECKPOINT CLAIM" not in (tmp_path / "localpilot-data" / "audit.jsonl").read_text(
        encoding="utf-8"
    )
    assert "MISLEADING CHECKPOINT CLAIM" not in (tmp_path / "localpilot-data" / "learning.sqlite3").read_bytes().decode(
        "latin-1"
    )
    assert not (tmp_path / "localpilot-data" / "evolution-checkpoint.json").exists()
    scrubbed = agent.audit.latest("model_research_notebook_scrubbed")
    assert scrubbed is not None and scrubbed["notebook_entries_retained"] == 0
    assert Config().agent.research_hard_tool_rounds == 24


def test_late_raw_observation_survives_checkpoint_control_stripping(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=1, hard=2)
    (tmp_path / "first.txt").write_text("EARLY RAW AUTHORITY", encoding="utf-8")
    (tmp_path / "late.txt").write_text("LATE RAW AUTHORITY", encoding="utf-8")
    snapshots = []
    checkpoint = _checkpoint_args(proposed_path="late.txt")
    streams = iter(
        [
            [_chunk(tool_calls=[_call("read_repository_file", {"path": "first.txt"})])],
            [_chunk(tool_calls=[_call(RESEARCH_NOTEBOOK_TOOL, checkpoint)])],
            [
                _chunk(
                    tool_calls=[
                        _call(
                            "read_repository_file",
                            {"path": "late.txt", "start_line": 1, "end_line": 20},
                        )
                    ]
                )
            ],
            [_chunk(thinking="The complete raw results are ready for synthesis.")],
            [_chunk(content="Both early and late raw authority are present.")],
        ]
    )

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter(next(streams))

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=fake_chat, ResponseError=ResponseError),
    )

    answer = agent.ask("Inspect both files and answer from their raw contents.")

    assert answer == "Both early and late raw authority are present."
    final_context = str(snapshots[-1])
    assert "EARLY RAW AUTHORITY" in final_context
    assert "LATE RAW AUTHORITY" in final_context
    assert "TRANSIENT RESEARCH CHECKPOINT" not in final_context
    assert "result_that_would_change_the_conclusion" not in final_context


def test_streamed_tool_call_parse_error_recovers_to_valid_compact_checkpoint_and_observation(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=1, hard=2)
    (tmp_path / "first.txt").write_text("FIRST RAW VALUE: blue", encoding="utf-8")
    (tmp_path / "late.txt").write_text("LATE RAW VALUE: blue", encoding="utf-8")
    late_reads = 0
    original = agent.tools["read_repository_file"]

    def counted_read(**kwargs):
        nonlocal late_reads
        if kwargs.get("path") == "late.txt":
            late_reads += 1
        return original.fn(**kwargs)

    agent.tools["read_repository_file"] = SimpleNamespace(risk=original.risk, fn=counted_read)
    snapshots = []
    invocations = []
    checkpoint = _checkpoint_args(proposed_path="late.txt")

    def malformed_stream():
        yield _chunk(thinking="PRIVATE MALFORMED PARTIAL REASONING")
        raise ResponseError(
            "error parsing tool call: invalid character ',' after array element"
        )

    streams = iter(
        [
            iter([_chunk(tool_calls=[_call("read_repository_file", {"path": "first.txt"})])]),
            malformed_stream(),
            iter([_chunk(tool_calls=[_call(RESEARCH_NOTEBOOK_TOOL, checkpoint)])]),
            iter(
                [
                    _chunk(
                        tool_calls=[
                            _call(
                                "read_repository_file",
                                {"path": "late.txt", "start_line": 1, "end_line": 20},
                            )
                        ]
                    )
                ]
            ),
            iter([_chunk(thinking="Both raw values agree.")]),
            iter([_chunk(content="The authoritative raw values are both blue.")]),
        ]
    )

    def fake_chat(**kwargs):
        invocations.append(kwargs)
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return next(streams)

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=fake_chat, ResponseError=ResponseError),
    )

    answer = agent.ask("Inspect both files and report their raw values.")

    assert answer == "The authoritative raw values are both blue."
    assert late_reads == 1
    assert all(snapshot is not None for snapshot in snapshots)
    assert "PRIVATE MALFORMED PARTIAL REASONING" in str(snapshots[2])
    assert "nothing executed or counted" in str(snapshots[2])
    assert "proposed_arguments must be an object" in str(snapshots[2])
    assert all(call["think"] == "high" for call in invocations)
    final_context = str(snapshots[-1])
    assert "FIRST RAW VALUE: blue" in final_context
    assert "LATE RAW VALUE: blue" in final_context
    assert "PRIVATE MALFORMED PARTIAL REASONING" not in final_context
    assert "nothing executed or counted" not in final_context
    assert "TRANSIENT RESEARCH CHECKPOINT" not in final_context
    retained = str(agent.messages)
    audit_text = (tmp_path / "localpilot-data" / "audit.jsonl").read_text(encoding="utf-8")
    learning_text = (tmp_path / "localpilot-data" / "learning.sqlite3").read_bytes().decode(
        "latin-1"
    )
    assert "PRIVATE MALFORMED PARTIAL REASONING" not in retained
    assert "PRIVATE MALFORMED PARTIAL REASONING" not in audit_text
    assert "PRIVATE MALFORMED PARTIAL REASONING" not in learning_text
    assert "invalid character" not in audit_text
    assert "nothing executed or counted" not in audit_text
    assert not (tmp_path / "localpilot-data" / "evolution-checkpoint.json").exists()
    retry = agent.audit.latest("model_tool_call_protocol_recovery_retry")
    assert retry is not None and retry["attempt"] == 1
    state = agent.audit.latest("model_evidence_state")
    assert state is not None and state["tool_rounds"] == 2


def test_tool_call_parse_recovery_exhaustion_is_visible_and_bounded(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path, soft=1, hard=3)
    (tmp_path / "first.txt").write_text("FIRST RAW VALUE", encoding="utf-8")
    calls = []

    def malformed_stream(label):
        def generate():
            yield _chunk(thinking=label)
            raise ResponseError("error parsing tool call: malformed arguments")

        return generate()

    streams = iter(
        [
            iter([_chunk(tool_calls=[_call("read_repository_file", {"path": "first.txt"})])]),
            malformed_stream("PRIVATE RETRY ONE"),
            malformed_stream("PRIVATE RETRY TWO"),
            malformed_stream("PRIVATE RETRY EXHAUSTED"),
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return next(streams)

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=fake_chat, ResponseError=ResponseError),
    )

    answer = agent.ask("Inspect first.txt and continue researching if needed.")

    assert answer == (
        "[LocalPilot could not recover a valid tool call after the bounded protocol retries; "
        "no malformed call was executed.]"
    )
    assert len(calls) == 4
    assert all(call["think"] == "high" for call in calls)
    assert "PRIVATE RETRY" not in str(agent.messages)
    exhausted = agent.audit.latest("model_tool_call_protocol_recovery_exhausted")
    assert exhausted is not None and exhausted["retries"] == exhausted["retry_limit"] == 2


def test_non_tool_protocol_response_error_still_surfaces(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)

    def failed_chat(**kwargs):
        def generate():
            if False:
                yield None
            raise ResponseError("Internal Server Error", 500)

        return generate()

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=failed_chat, ResponseError=ResponseError),
    )

    with pytest.raises(ResponseError, match="Internal Server Error"):
        agent.ask("Answer normally.")


def test_generation_limited_final_reasoning_continues_once_in_same_raw_context(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path, soft=1, hard=1)
    (tmp_path / "known.txt").write_text("VERIFIED RAW EVIDENCE: blue", encoding="utf-8")
    calls = []
    snapshots = []
    first_pass_reasoning = "first final-pass private reasoning"
    first_pass_reasoning += "r" * (18611 - len(first_pass_reasoning))
    streams = iter(
        [
            [_chunk(tool_calls=[_call("read_repository_file", {"path": "known.txt"})])],
            [_chunk(thinking="research is complete from the raw result")],
            [
                _chunk(
                    thinking=first_pass_reasoning,
                    done=True,
                    done_reason="length",
                    prompt_eval_count=18752,
                    eval_count=4096,
                )
            ],
            [
                _chunk(
                    thinking="continuation private reasoning",
                    content="The verified raw evidence says the canonical value is blue.",
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=22900,
                    eval_count=640,
                )
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Inspect known.txt and answer from its raw evidence.")

    assert answer == "The verified raw evidence says the canonical value is blue."
    assert len(calls) == 4
    assert calls[2]["think"] == "high"
    assert calls[3]["think"] is False
    assert "tools" not in calls[2]
    assert "tools" not in calls[3]
    assert calls[2]["options"]["num_predict"] == 4096
    assert calls[3]["options"]["num_predict"] == 2048
    continuation_context = str(snapshots[3])
    assert "VERIFIED RAW EVIDENCE: blue" in continuation_context
    assert "first final-pass private reasoning" in continuation_context
    assert "Render the conclusion of that exact reasoning" in continuation_context
    assert "first final-pass private reasoning" not in str(agent.messages)
    assert "continuation private reasoning" not in str(agent.messages)
    audit_text = (tmp_path / "localpilot-data" / "audit.jsonl").read_text(encoding="utf-8")
    assert "first final-pass private reasoning" not in audit_text
    assert "continuation private reasoning" not in audit_text
    assert "first final-pass private reasoning" not in (
        tmp_path / "localpilot-data" / "learning.sqlite3"
    ).read_bytes().decode("latin-1")
    assert not (tmp_path / "localpilot-data" / "evolution-checkpoint.json").exists()
    completed = agent.audit.latest(
        "model_same_context_generation_limit_continuation_complete"
    )
    assert completed is not None and completed["exhausted"] is False
    audit_rows = [json.loads(line) for line in audit_text.splitlines()]
    first_final_runtime = next(
        row
        for row in audit_rows
        if row.get("event") == "model_stream_complete"
        and row.get("phase") == "same_context_answer"
    )
    assert first_final_runtime["think"] == "high"
    assert first_final_runtime["done_reason"] == "length"
    assert first_final_runtime["runtime_classification"] == "generation_limit"
    assert first_final_runtime["context_tokens"] == 32768
    assert first_final_runtime["prompt_eval_count"] == 18752
    assert first_final_runtime["context_used_percent"] == 57.23
    assert first_final_runtime["num_predict"] == 4096
    assert first_final_runtime["eval_count"] == 4096
    assert first_final_runtime["reasoning_chars"] == 18611
    assert first_final_runtime["content_chars"] == 0
    assert first_final_runtime["tool_calls"] == 0


def test_generation_limit_continuation_is_still_behavior_postvalidated(
    tmp_path, monkeypatch
):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [_chunk(thinking="initial reasoning")],
            [
                _chunk(
                    thinking="long final reasoning",
                    done=True,
                    done_reason="length",
                    prompt_eval_count=1000,
                    eval_count=4096,
                )
            ],
            [
                _chunk(
                    content="DECLINE: No raw tool outputs are available.",
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=5200,
                    eval_count=80,
                )
            ],
            [_chunk(content="The tension I keep returning to is initiative without overreach. My provisional view is that small reversible investigations are the best way to preserve both.")],
        ]
    )
    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "No task list today. You have room to think. What has your attention right now?"
    )

    assert "initiative without overreach" in answer
    recovery = agent.audit.latest("model_same_context_behavior_recovery_complete")
    assert recovery["original_issues"] == ["unwarranted_open_ended_decline"]
    assert recovery["accepted"] is True
    recovery_context = str(calls[-1]["messages"])
    assert "long final reasoning" not in recovery_context
    assert "Finish the owner's answer now" not in recovery_context


def test_second_generation_limit_returns_visible_marker_without_loop(tmp_path, monkeypatch):
    _, agent = _agent(tmp_path)
    calls = []
    streams = iter(
        [
            [_chunk(thinking="initial private reasoning")],
            [
                _chunk(
                    thinking="first final-pass private reasoning",
                    done=True,
                    done_reason="length",
                    prompt_eval_count=18752,
                    eval_count=4096,
                )
            ],
            [
                _chunk(
                    thinking="bounded continuation private reasoning",
                    done=True,
                    done_reason="length",
                    prompt_eval_count=22900,
                    eval_count=8192,
                )
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask("Reason carefully and answer.")

    assert answer == (
        "[LocalPilot's single bounded same-context answer continuation also reached "
        "its generation limit before producing a usable final answer.]"
    )
    assert len(calls) == 3
    assert [call["think"] for call in calls] == ["high", "high", False]
    assert "tools" not in calls[1]
    assert "tools" not in calls[2]
    assert "first final-pass private reasoning" not in str(agent.messages)
    assert "bounded continuation private reasoning" not in str(agent.messages)
    completed = agent.audit.latest(
        "model_same_context_generation_limit_continuation_complete"
    )
    assert completed is not None and completed["exhausted"] is True


def test_generation_limit_continuation_budget_preserves_context_safety_margin(tmp_path):
    _, agent = _agent(tmp_path)

    observed = agent._generation_limit_continuation_budget(
        {"context_tokens": 32768, "prompt_eval_count": 18752, "eval_count": 4096}
    )
    near_ceiling = agent._generation_limit_continuation_budget(
        {"context_tokens": 32768, "prompt_eval_count": 30000, "eval_count": 500}
    )
    too_close = agent._generation_limit_continuation_budget(
        {"context_tokens": 32768, "prompt_eval_count": 31000, "eval_count": 200}
    )

    assert observed == 8192
    assert near_ceiling == 630
    assert 30000 + 500 + near_ceiling + 1638 == 32768
    assert too_close == 0
