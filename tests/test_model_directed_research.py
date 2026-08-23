import json
import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.audit import AuditLog
from localpilot.config import Config


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
