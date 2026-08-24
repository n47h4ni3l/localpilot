import json
import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.audit import AuditLog
from localpilot.config import Config
from localpilot.research import RESEARCH_NOTEBOOK_TOOL, TransientResearchNotebook


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


def _checkpoint_args(*, proposed_path: str, fact_pointer: str = "result-001"):
    return {
        "verified_fact_pointers": [f"{fact_pointer}: locate the raw canonical-value result"],
        "unresolved_questions": [f"{fact_pointer}: does a second source reconcile the value?"],
        "inspected_observation_ids": ["obs-001"],
        "unresolved_fact": f"{fact_pointer}: the canonical value needs independent reconciliation",
        "why_current_evidence_is_insufficient": f"{fact_pointer}: only one raw source is present",
        "proposed_tool": "read_repository_file",
        "proposed_arguments_json": json.dumps(
            {"path": proposed_path, "start_line": 1, "end_line": 20}, sort_keys=True
        ),
        "result_that_would_change_the_conclusion": "A conflicting canonical value would require reporting the conflict.",
        "new_hypothesis": "",
    }


def test_notebook_rejects_unsupported_claims_and_requires_redundancy_hypothesis():
    notebook = TransientResearchNotebook()
    first = notebook.add_observation(
        tool="search_repository",
        arguments={"query": "model-adaptation-lab"},
        ok=True,
    )
    unsupported = _checkpoint_args(proposed_path="second.txt", fact_pointer="result-999")
    unsupported["proposed_tool"] = "search_repository"
    unsupported["proposed_arguments_json"] = json.dumps({"query": "model adaptation lab"})

    rejected = notebook.submit(unsupported)

    assert rejected.accepted is False
    assert "unknown current-turn references" in rejected.message

    redundant = dict(unsupported)
    redundant["verified_fact_pointers"] = [f"{first.result_id}: prior search result pointer"]
    redundant["unresolved_questions"] = [f"{first.observation_id}: whether alternate spelling differs"]
    redundant["inspected_observation_ids"] = [first.observation_id]
    redundant["unresolved_fact"] = f"{first.result_id}: alternate spelling remains untested"
    redundant["why_current_evidence_is_insufficient"] = f"{first.result_id}: prior query may be lexical"

    needs_justification = notebook.submit(redundant)

    assert needs_justification.accepted is False
    assert needs_justification.redundant_with == (first.observation_id,)
    assert "distinct new hypothesis" in needs_justification.message

    redundant["new_hypothesis"] = (
        f"{first.observation_id}: underscore normalization may expose paths omitted by the hyphenated query"
    )
    justified = notebook.submit(redundant)

    assert justified.accepted is True
    assert notebook.authorizes("search_repository", {"query": "model adaptation lab"})

    next_turn = TransientResearchNotebook(start_at=2)
    second = next_turn.add_observation(tool="search_repository", arguments={"query": "fresh"}, ok=True)
    stale = dict(redundant)
    stale["proposed_arguments_json"] = json.dumps({"query": "different"})
    stale_decision = next_turn.submit(stale)
    assert second.observation_id == "obs-002"
    assert stale_decision.accepted is False
    assert "unknown current-turn references" in stale_decision.message


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
    assert accepted["verified_pointer_count"] == 1


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
    misleading = _checkpoint_args(proposed_path="late.txt")
    misleading["verified_fact_pointers"] = [
        "obs-001/result-001: MISLEADING NOTEBOOK CLAIM — canonical value is red"
    ]
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
    assert "MISLEADING NOTEBOOK CLAIM" in final_context
    assert "If notebook text conflicts with a raw result, the raw result controls" in final_context
    assert "MISLEADING NOTEBOOK CLAIM" not in str(agent.messages)
    assert "MISLEADING NOTEBOOK CLAIM" not in (tmp_path / "localpilot-data" / "audit.jsonl").read_text(
        encoding="utf-8"
    )
    assert "MISLEADING NOTEBOOK CLAIM" not in (tmp_path / "localpilot-data" / "learning.sqlite3").read_bytes().decode(
        "latin-1"
    )
    assert not (tmp_path / "localpilot-data" / "evolution-checkpoint.json").exists()
    scrubbed = agent.audit.latest("model_research_notebook_scrubbed")
    assert scrubbed is not None and scrubbed["notebook_entries_retained"] == 0
    assert Config().agent.research_hard_tool_rounds == 24


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
    assert calls[3]["think"] == "high"
    assert "tools" not in calls[2]
    assert "tools" not in calls[3]
    assert calls[2]["options"]["num_predict"] == 4096
    assert calls[3]["options"]["num_predict"] == 8192
    continuation_context = str(snapshots[3])
    assert "VERIFIED RAW EVIDENCE: blue" in continuation_context
    assert "first final-pass private reasoning" in continuation_context
    assert "Finish the owner's answer now" in continuation_context
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
    assert [call["think"] for call in calls] == ["high", "high", "high"]
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
