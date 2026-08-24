import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace

from localpilot.agent import (
    _LEARNING_MEMORY_CHAR_BUDGET,
    _LEARNING_MEMORY_FACT_LIMIT,
    LocalPilotAgent,
)
from localpilot.config import Config
from localpilot.learning import LearningMemory


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
    config.agent.research_soft_tool_rounds = 2
    config.agent.research_hard_tool_rounds = 4
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    return agent


def _record(
    memory: LearningMemory,
    *,
    key: str,
    subject: str,
    summary: str,
    stage: str = "self",
    source_uri: str = "repo://localpilot/agent.py",
    source_digest: str = "stored-digest",
    relationships=(),
):
    memory.upsert_knowledge_fact(
        stage=stage,
        fact_key=key,
        fact_type="architecture_contract",
        subject=subject,
        summary=summary,
        source_uri=source_uri,
        source_kind="python_ast",
        source_digest=source_digest,
        confidence=0.9,
        relationships=relationships,
    )


def _payload(context: str) -> dict:
    return json.loads(context.split("\n", 1)[1])


def test_relevance_search_is_bounded_and_preserves_fact_authority_metadata(tmp_path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    for index in range(20):
        _record(
            memory,
            key=f"operator:{index}",
            subject=f"OperatorComponent{index}",
            summary=f"Operator research memory contract {index} uses source-linked evidence.",
            relationships=(f"symbol:operator:{index}",),
        )
    _record(
        memory,
        key="irrelevant:weather",
        subject="Weather",
        summary="Adelaide forecast and rainfall.",
    )

    results = memory.search_knowledge_facts(
        "Explain the operator research memory contract and its evidence sources",
        limit=8,
    )

    assert len(results) == 8
    assert all(item.fact_key.startswith("operator:") for item in results)
    assert all(item.source_uri == "repo://localpilot/agent.py" for item in results)
    assert all(item.confidence == 0.9 for item in results)
    assert all(item.last_verified_at for item in results)
    assert all(item.stale is False for item in results)
    assert memory.search_knowledge_facts("Adelaide weather tomorrow", limit=8)[0].subject == "Weather"


def test_turn_context_is_small_source_linked_and_irrelevant_queries_get_nothing(tmp_path):
    agent = _agent(tmp_path)
    (tmp_path / "localpilot").mkdir()
    source = tmp_path / "localpilot" / "agent.py"
    source.write_text("class LocalPilotAgent: pass\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for index in range(15):
        _record(
            agent.memory,
            key=f"memory:{index}",
            subject=f"LearningMemory{index}",
            summary="Durable learning memory retrieves bounded facts for the operator research loop. " * 4,
            source_digest=digest,
            relationships=("localpilot/learning.py", "localpilot/agent.py", "extra"),
        )

    context, facts = agent._learning_context(
        "How does the LocalPilot operator research loop use durable learning memory?"
    )
    parsed = _payload(context)

    assert 6 <= len(facts) <= _LEARNING_MEMORY_FACT_LIMIT
    assert len(context) <= _LEARNING_MEMORY_CHAR_BUDGET
    assert parsed["returned_count"] == len(facts)
    assert all(item["source_uri"].startswith("repo://") for item in parsed["facts"])
    assert all(item["source_digest"] == digest for item in parsed["facts"])
    assert all(item["repository_source_digest_status"] == "match" for item in parsed["facts"])
    assert all("confidence" in item and "stale" in item for item in parsed["facts"])
    assert agent._learning_context("Plan a holiday in Adelaide tomorrow") == ("", [])


def test_digest_mismatch_and_stale_state_are_surfaced(tmp_path):
    agent = _agent(tmp_path)
    (tmp_path / "localpilot").mkdir()
    source = tmp_path / "localpilot" / "agent.py"
    source.write_text("AUTO_PROMOTE = False\n", encoding="utf-8")
    _record(
        agent.memory,
        key="safety:auto_promote",
        subject="auto_promote",
        summary="Candidate auto promotion is enabled.",
        source_digest="old-digest",
    )
    agent.memory.invalidate_knowledge_source("repo://localpilot/agent.py", "new-digest")

    context, facts = agent._learning_context(
        "Verify the current candidate auto_promote safety architecture"
    )
    item = _payload(context)["facts"][0]

    assert facts
    assert item["stale"] is True
    assert item["repository_source_digest_status"] == "mismatch"
    assert "live raw tool results control" in context


def test_declared_dependency_query_prioritizes_pyproject_live_check(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["ollama>=0.6.0"]\n', encoding="utf-8")
    _record(
        agent.memory,
        key="file:pyproject.toml",
        subject="pyproject.toml",
        summary="Tracked project file: pyproject.toml.",
        source_uri="repo://pyproject.toml",
        source_digest=hashlib.sha256(pyproject.read_bytes()).hexdigest(),
    )

    context, _ = agent._learning_context(
        "Verify the declared dependency for the Ollama streaming integration."
    )
    targets = _payload(context)["verification_targets"]

    assert targets[0] == {
        "source_uri": "repo://pyproject.toml",
        "reason": "Read the declared dependency before other live repository checks.",
        "tool": "read_repository_file",
        "arguments": {
            "path": "pyproject.toml",
            "start_line": 1,
            "end_line": 120,
        },
    }
    assert targets[1]["tool"] == "search_repository"
    assert targets[1]["arguments"]["query"] == "from ollama import chat"
    assert targets[2]["tool"] == "read_repository_file"
    assert targets[2]["arguments"] == {
        "path": "localpilot/agent.py",
        "start_line": 450,
        "end_line": 580,
    }
    snapshots = []

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter([_chunk(content="pyproject declares ollama>=0.6.0.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Verify the declared dependency for the Ollama streaming integration.")

    assert answer == "pyproject declares ollama>=0.6.0."
    assert "ollama>=0.6.0" in str(snapshots[0])
    assert agent.audit.latest("model_learning_memory_live_verification")["target_count"] == 3
    assert "Memory-guided live verification" not in str(agent.messages)


def test_current_live_raw_result_remains_in_same_context_and_controls_answer(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    (tmp_path / "localpilot").mkdir()
    source = tmp_path / "localpilot" / "agent.py"
    source.write_text("AUTO_PROMOTE = False  # human merge only\n", encoding="utf-8")
    _record(
        agent.memory,
        key="safety:auto_promote",
        subject="auto_promote",
        summary="Candidate auto promotion is enabled.",
        source_digest="contradicted-digest",
    )
    snapshots = []
    streams = iter(
        [
            [_chunk(content="Live source controls: auto promotion is false and merge remains human-only.")],
        ]
    )

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(
        "Inspect the current LocalPilot architecture and verify candidate auto_promote safety."
    )

    assert "auto promotion is false" in answer
    final = snapshots[-1]
    memory_index = next(
        index for index, item in enumerate(final)
        if "durable_study_memory_retrieval" in str(item.get("content"))
    )
    live_index = next(
        index for index, item in enumerate(final)
        if "Memory-guided live verification" in str(item.get("content"))
        and "AUTO_PROMOTE = False" in str(item.get("content"))
    )
    assert memory_index < live_index
    assert "Candidate auto promotion is enabled" in str(final[memory_index])
    assert "Observation ID:" in str(final[live_index])
    assert "Finding 1" not in str(final)


def test_retrieved_memory_is_turn_local_and_never_relearned_or_exposes_reasoning(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    _record(
        agent.memory,
        key="python:subprocess",
        subject="subprocess",
        stage="python",
        summary="Use argv sequences and shell=False for explicit process boundaries.",
        source_uri="https://docs.python.org/3/library/subprocess.html",
    )
    before = len(agent.memory.knowledge_facts(include_stale=True))
    snapshots = []
    thinks = []
    tool_visibility = []

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        thinks.append(kwargs["think"])
        tool_visibility.append("tools" in kwargs)
        return iter([_chunk(content="The retained subprocess contract recommends argv and shell=False.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(
        "Without using any tools, what did Python study retain about subprocess process boundaries?"
    )

    assert "argv" in answer
    assert thinks == ["low"]
    assert tool_visibility == [False]
    assert any("durable_study_memory_retrieval" in str(item) for item in snapshots[0])
    assert "durable_study_memory_retrieval" not in str(agent.messages)
    assert len(agent.memory.knowledge_facts(include_stale=True)) == before
    assert agent.audit.latest("model_learning_memory_retrieved")["fact_count"] == 1
    assert agent.audit.latest("model_learning_memory_scrubbed")["retained_in_messages"] is False
    assert agent.audit.latest("model_learning_memory_live_verification") is None
    with sqlite3.connect(agent.memory.path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(knowledge_facts)").fetchall()
        }
    assert not columns & {
        "prompt", "transcript", "messages", "thinking", "reasoning", "chain_of_thought",
    }


def test_memory_guided_turn_forces_synthesis_after_four_live_observations(
    tmp_path, monkeypatch
):
    config = Config()
    config.agent.research_soft_tool_rounds = 12
    config.agent.research_hard_tool_rounds = 24
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    for index in range(4):
        path = tmp_path / f"source-{index}.txt"
        path.write_text(f"verified {index}\n", encoding="utf-8")
    _record(
        agent.memory,
        key="architecture:operator",
        subject="operator architecture",
        summary="Operator architecture uses targeted repository verification.",
    )
    streams = iter(
        [
            [_chunk(tool_calls=[_call("read_repository_file", {
                "path": f"source-{index}.txt", "start_line": 1, "end_line": 2,
            })])]
            for index in range(4)
        ]
        + [
            [_chunk(tool_calls=[_call("read_repository_file", {
                "path": "source-0.txt", "start_line": 1, "end_line": 2,
            })])],
            [_chunk(content="Four targeted live sources are sufficient for the answer.")],
        ]
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Inspect the current LocalPilot operator architecture.")

    assert answer.startswith("Four targeted")
    assert len(calls) == 6
    assert all("tools" in item for item in calls[:4])
    assert "tools" not in calls[4]
    assert "tools" not in calls[5]
    assert all(item["think"] == "high" for item in calls[:5])
    assert calls[5]["think"] == "low"
    budget = agent.audit.latest("model_learning_memory_research_budget")
    assert budget["soft_tool_rounds"] == 4
    assert budget["hard_tool_rounds"] == 4
    assert agent.audit.latest("model_research_soft_budget") is None
