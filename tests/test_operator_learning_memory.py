import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from localpilot.agent import (
    _LEARNING_MEMORY_CHAR_BUDGET,
    _LEARNING_MEMORY_FACT_LIMIT,
    LocalPilotAgent,
    GUIDE_FIRST_SYSTEM_PROMPT,
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


def _force_stale_repository_retrieval(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    (tmp_path / "localpilot").mkdir()
    (tmp_path / "localpilot" / "agent.py").write_text("# revised source\n", encoding="utf-8")
    claims = {
        "config.model.name": "LocalPilot model configuration selects the Operator model.",
        "AgentConfig.research_hard_tool_rounds": "The Operator repository defines bounded research tool budgets.",
        "StudyEngine.upsert_knowledge_facts": "Repository study persists durable knowledge facts.",
        "LocalPilotAgent.record_human_lesson": "Explicit owner teaching stores a human lesson.",
        "auto_promote": "Candidate auto promotion is enabled.",
        "SystemSenseReader.provider_path": "SystemSense hardware provider configuration loads LibreHardwareMonitor.",
    }
    for subject, summary in claims.items():
        _record(agent.memory, key=f"symbol:{subject}", subject=subject, summary=summary)
    agent.memory.invalidate_knowledge_source("repo://localpilot/agent.py", "changed-digest")
    facts = agent.memory.knowledge_facts(include_stale=True)
    assert len(facts) == 6 and all(fact.stale for fact in facts)
    # Force the audited overbroad retrieval through the real context/target selector.
    monkeypatch.setattr(agent.memory, "search_knowledge_facts", lambda *args, **kwargs: facts)
    return agent, facts


def test_live_systemsense_question_does_not_select_stale_repository_targets(tmp_path, monkeypatch):
    agent, facts = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    prompt = "Check this PC's current CPU and GPU temperatures using SystemSense."
    context, retrieved = agent._learning_context(prompt)
    assert len(retrieved) == 6
    assert _payload(context).get("verification_targets", []) == []
    assert all(item["material_to_request"] is False for item in retrieved)
    assert all("verification_required" not in item for item in retrieved)
    executed = []

    def live_read(**kwargs):
        executed.append("PC")
        return '{"cpu_temperature_c": 63, "gpu_temperature_c": 51}'

    def repository_read(**kwargs):
        executed.append("repository")
        return "Unrelated repository source."

    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=live_read
    )
    agent.tools["search_repository"] = replace(agent.tools["search_repository"], fn=repository_read)
    agent.tools["read_repository_file"] = replace(agent.tools["read_repository_file"], fn=repository_read)
    streams = iter([
        [_chunk(tool_calls=[_call("get_system_sense_summary")])],
        [_chunk(content="The CPU is 63 C and GPU is 51 C.")],
        [_chunk(content="The CPU is 63 C and GPU is 51 C.")],
    ])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=lambda **kwargs: iter(next(streams))))
    assert "63" in agent.ask(prompt)
    assert executed == ["PC"]
    assert agent.audit.latest("model_learning_memory_live_verification") is None
    assert agent.memory.knowledge_facts(include_stale=True) == facts


def test_mixed_stale_memory_supplies_material_targets_without_forced_reads(tmp_path, monkeypatch):
    agent, facts = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    prompt = "Inspect the local repository architecture and verify auto_promote safety."
    context, retrieved = agent._learning_context(prompt)
    targets = _payload(context)["verification_targets"]
    assert len(targets) == 1
    assert targets[0]["arguments"]["query"] == "auto_promote"
    assert [item["subject"] for item in retrieved if item["material_to_request"]] == ["auto_promote"]
    executed = []

    def repository_read(**kwargs):
        executed.append(kwargs["query"])
        return "AUTO_PROMOTE = False  # human merge only"

    agent.tools["search_repository"] = replace(agent.tools["search_repository"], fn=repository_read)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=lambda **kwargs: iter([
        _chunk(content="The live source shows auto promotion is false and merge remains human-only.")
    ])))
    assert "auto promotion is false" in agent.ask(prompt)
    assert executed == []
    assert agent.audit.latest("model_learning_memory_live_verification") is None
    assert agent.memory.knowledge_facts(include_stale=True) == facts


@pytest.mark.parametrize("prompt", [
    "Inspect the repository architecture for HTTP cache eviction.",
    "Review repository Ollama streaming integration.",
    "Explain the architecture of a medieval castle.",
    "Review CPU cache configuration in this PC.",
])
def test_no_material_memory_claims_produce_no_automatic_targets(tmp_path, monkeypatch, prompt):
    agent, _ = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    context, retrieved = agent._learning_context(prompt)
    assert len(retrieved) == 6
    assert _payload(context).get("verification_targets", []) == []
    assert not any(item["material_to_request"] for item in retrieved)


def test_repository_topic_match_selects_claim_without_an_exact_symbol(tmp_path, monkeypatch):
    agent, _ = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    context, _ = agent._learning_context("Inspect the repository Operator research tool budgets.")
    assert [target["arguments"]["query"] for target in _payload(context)["verification_targets"]] == [
        "research_hard_tool_rounds"
    ]


def test_qualified_config_field_selects_its_claim_without_generic_name_matches(tmp_path, monkeypatch):
    agent, _ = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    context, _ = agent._learning_context("Inspect the repository model.name setting.")
    assert [target["arguments"]["query"] for target in _payload(context)["verification_targets"]] == ["name"]


def test_systemsense_repository_implementation_claim_is_material_when_requested(tmp_path, monkeypatch):
    agent, _ = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    context, _ = agent._learning_context(
        "Inspect the repository implementation of SystemSenseReader.provider_path."
    )
    assert [target["arguments"]["query"] for target in _payload(context)["verification_targets"]] == [
        "provider_path"
    ]


def test_named_source_and_symbol_matching_do_not_use_substrings(tmp_path, monkeypatch):
    agent, _ = _force_stale_repository_retrieval(tmp_path, monkeypatch)
    context, _ = agent._learning_context("Verify auto_promoter in the repository.")
    assert _payload(context).get("verification_targets", []) == []
    context, _ = agent._learning_context("Inspect the repository source file agent.py.")
    targets = _payload(context)["verification_targets"]
    assert len(targets) == 4
    assert all(target["arguments"]["path"] == "localpilot/agent.py" for target in targets)


def test_empty_durable_memory_does_not_create_keyword_based_targets(tmp_path):
    agent = _agent(tmp_path)
    assert agent._learning_context("Inspect the LocalPilot Ollama streaming integration architecture.") == ("", [])


def test_relevant_stale_fact_with_unchanged_bytes_still_requires_verification(tmp_path):
    agent = _agent(tmp_path)
    source = tmp_path / "policy.py"
    source.write_text("AUTO_PROMOTE = False\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _record(agent.memory, key="auto_promote", subject="auto_promote",
            summary="Auto promotion is enabled.", source_uri="repo://policy.py", source_digest=digest)
    agent.memory.invalidate_knowledge_source("repo://policy.py", "different-digest")
    context, retrieved = agent._learning_context("Verify auto_promote in the repository.")
    assert retrieved[0]["stale"] and retrieved[0]["repository_source_digest_status"] == "match"
    assert _payload(context)["verification_targets"][0]["arguments"]["query"] == "auto_promote"


def _synthesis_memory_agent(tmp_path, *, claim_count=3):
    agent = _agent(tmp_path)
    agent.config.agent.research_soft_tool_rounds = 4
    (tmp_path / "localpilot").mkdir()
    (tmp_path / "localpilot" / "agent.py").write_text(
        "\n".join(f"class RoutingClaim{index}: pass" for index in range(claim_count)),
        encoding="utf-8",
    )
    for index in range(claim_count):
        _record(
            agent.memory, key=f"symbol:RoutingClaim{index}", subject=f"RoutingClaim{index}",
            summary=f"RoutingClaim{index} is a repository class.",
        )
    agent.memory.invalidate_knowledge_source("repo://localpilot/agent.py", "revised-digest")
    prompt = "Verify " + ", ".join(f"RoutingClaim{index}" for index in range(claim_count))
    prompt += " in the local repository."
    return agent, prompt


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
    targets = _payload(context)["verification_targets"]
    assert targets[0]["tool"] == "search_repository"
    assert targets[0]["arguments"]["query"] == "auto_promote"
    assert len(targets) == 1  # The word architecture must not add unrelated writer checks.


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
        return iter([
            _chunk(
                content=(
                    "The retained subprocess prior recommends argv and shell=False; "
                    "the current CommandRunner remains unverified without a live source read."
                )
            )
        ])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(
        "Without using any tools, what did Python study retain about subprocess process boundaries, "
        "and how should that prior influence but not prove claims about the current LocalPilot "
        "CommandRunner implementation?"
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
    assert agent.audit.latest("model_evidence_acquisition_failed") is None
    with sqlite3.connect(agent.memory.path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(knowledge_facts)").fetchall()
        }
    assert not columns & {
        "prompt", "transcript", "messages", "thinking", "reasoning", "chain_of_thought",
    }


def test_memory_turn_keeps_configured_budget_and_model_chosen_synthesis(
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
            [_chunk(content="Four targeted live sources are sufficient for the answer.")],
        ]
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Inspect the current LocalPilot operator research design.")

    assert answer.startswith("Four targeted")
    assert len(calls) == 6
    assert all("tools" in item for item in calls[:4])
    assert "tools" in calls[4]
    assert "tools" in calls[5]
    assert all(item["think"] == "high" for item in calls[:5])
    assert calls[5]["think"] == "high"
    budget = agent.audit.latest("model_learning_memory_research_budget")
    assert budget is None
    assert agent.config.agent.research_soft_tool_rounds == 12
    assert agent.config.agent.research_hard_tool_rounds == 24
    assert agent.audit.latest("model_research_soft_budget") is None


def test_explicit_study_stage_outranks_loose_cross_stage_matches(tmp_path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    _record(
        memory,
        key="self:ollama_integration",
        subject="LocalPilot Ollama integration",
        summary="Self architecture has Ollama streaming integration and tool calls.",
        stage="self",
    )
    _record(
        memory,
        key="qwen:tool_calling",
        subject="tool_calling",
        summary="Ollama passes tools and Qwen models can return structured tool calls while streaming.",
        stage="qwen",
        source_uri="https://ollama.com/blog/tool-support",
    )

    results = memory.search_knowledge_facts(
        "Distinguish the Qwen tool calling contract in the Ollama streaming integration",
        limit=2,
    )

    assert results[0].stage == "qwen"
