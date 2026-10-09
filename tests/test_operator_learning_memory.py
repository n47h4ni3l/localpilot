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
    SYSTEM_PROMPT,
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


@pytest.mark.parametrize(
    "live_result", ['{"cpu_temperature_c": 63}', "Tool error: sensors unavailable"]
)
@pytest.mark.parametrize("budget", [1, 2, 4])
def test_explicit_pc_evidence_precedes_stale_repository_memory(
    tmp_path, monkeypatch, live_result, budget
):
    agent = _agent(tmp_path)
    agent.config.agent.research_soft_tool_rounds = budget
    agent.config.agent.research_hard_tool_rounds = budget
    for index in range(6):
        _record(
            agent.memory,
            key=f"irrelevant:{index}",
            subject=f"RepositoryComponent{index}",
            summary="Repository architecture contract unrelated to current PC temperatures.",
        )
    agent.memory.invalidate_knowledge_source("repo://localpilot/agent.py", "new-digest")
    facts = agent.memory.knowledge_facts(include_stale=True)
    assert len(facts) == 6 and all(fact.stale for fact in facts)
    # Reproduce the audited retrieval: six stale facts and four automatic reads.
    retrieved = [
        dict(asdict(fact), repository_source_digest_status="mismatch") for fact in facts
    ]
    context = "Retrieved memory\n" + json.dumps({
        "kind": "durable_study_memory_retrieval",
        "facts": retrieved,
        "verification_targets": [
            {
                "tool": "search_repository",
                "arguments": {
                    "query": f"RepositoryComponent{index}",
                    "path": "localpilot/agent.py",
                    "max_results": 10,
                },
            }
            for index in range(4)
        ],
    })
    monkeypatch.setattr(agent, "_learning_context", lambda prompt: (context, retrieved))
    executed = []

    def repository_read(**kwargs):
        executed.append("repository")
        return "Repository contract found."

    def live_read(**kwargs):
        executed.append("PC")
        return live_result

    agent.tools["search_repository"] = replace(
        agent.tools["search_repository"], fn=repository_read
    )
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=live_read
    )
    snapshots = []

    def fake_chat(**kwargs):
        snapshots.append(str(kwargs["messages"]))
        if len(snapshots) == 1:
            return iter([
                _chunk(tool_calls=[_call("search_repository", {"query": "unrelated"})])
            ])
        if len(snapshots) == 2:
            return iter([_chunk(tool_calls=[_call("get_system_sense_summary")])])
        content = (
            "The current CPU temperature is 63 C." if "Tool error" not in live_result
            else "Sensors are unavailable."
        )
        return iter([_chunk(content=content)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Check this PC's current CPU temperature using SystemSense.")
    assert executed[0] == "PC"
    assert "repository" not in executed
    assert agent.audit.latest("model_learning_memory_direct_synthesis") is None
    assert agent.memory.knowledge_facts(include_stale=True) == facts
    if "Tool error" not in live_result:
        assert "63" in answer
        assert "Observation ID:" in snapshots[-1]
        assert agent.audit.latest("model_evidence_state")["tool_rounds"] <= budget
    else:
        assert answer == "Sensors are unavailable."
        assert agent.audit.latest("model_evidence_acquisition_failed")["missing"] == ["Windows/PC state"]
        assert "63" not in answer
    assert agent.audit.latest("model_learning_memory_verification_deferred")["target_count"] == 4
    assert agent.audit.latest("model_live_pc_evidence_priority")["tool_rounds"] == 0


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


def test_mixed_stale_memory_verifies_only_the_material_repository_claim(tmp_path, monkeypatch):
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
    assert executed == ["auto_promote"]
    assert agent.audit.latest("model_learning_memory_live_verification")["target_count"] == 1
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


@pytest.mark.parametrize("source", ["Windows/PC state", "public HTTPS", "private GitHub"])
@pytest.mark.parametrize("source_ok", [True, False])
def test_completed_memory_verification_cannot_bypass_outstanding_direct_source(
    tmp_path, monkeypatch, source, source_ok
):
    agent, prompt = _synthesis_memory_agent(tmp_path)
    tool, arguments, request = {
        "Windows/PC state": (
            "get_system_sense_summary", {},
            "Check this PC's current CPU temperature using SystemSense.",
        ),
        "public HTTPS": (
            "fetch_public_https", {"url": "https://example.org/reference"},
            "Read https://example.org/reference before answering.",
        ),
        "private GitHub": (
            "get_github_pull_request", {"number": 30}, "Review PR #30 on private GitHub.",
        ),
    }[source]
    prompt += " " + request
    required = agent._evidence_requirements(prompt)
    assert required == {"trusted repository", source}
    context, facts = agent._learning_context(prompt)
    assert len(_payload(context)["verification_targets"]) == 3
    assert all(item["material_to_request"] and item["stale"] for item in facts)
    executed = []
    repository_read = agent.tools["search_repository"].fn
    if source == "Windows/PC state":
        # Step 1 normally defers these reads while PC evidence is pending.
        # Expose the explicit PC obligation at completion of the final read to
        # exercise the synthesis invariant independently of that reservation.
        required.remove(source)
        monkeypatch.setattr(agent, "_evidence_requirements", lambda _: required)

    def verify_repository(**kwargs):
        executed.append("repository")
        result = repository_read(**kwargs)
        if len(executed) == 3:
            required.add(source)
        return result

    def read_source(**kwargs):
        executed.append(source)
        if not source_ok:
            return "Tool error: required source unavailable"
        return '{"cpu_temperature_c": 63}' if source == "Windows/PC state" else "Requested source inspected."

    agent.tools["search_repository"] = replace(agent.tools["search_repository"], fn=verify_repository)
    agent.tools[tool] = replace(agent.tools[tool], fn=read_source)
    snapshots = []

    def fake_chat(**kwargs):
        snapshots.append({"tools": bool(kwargs.get("tools")), "messages": str(kwargs["messages"])})
        if kwargs.get("tools") and source not in executed:
            return iter([_chunk(tool_calls=[_call(tool, arguments)])])
        answer = (
            "The CPU temperature is 63 C." if source == "Windows/PC state" and source_ok
            else "Verified the repository symbols and inspected the requested source."
            if source_ok else "The requested source remains unverified."
        )
        return iter([_chunk(content=answer)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(prompt)
    verification = agent.audit.latest("model_learning_memory_live_verification")
    assert verification["target_count"] == 3
    assert verification["succeeded"] == ["trusted repository"]
    assert agent.audit.latest("model_learning_memory_direct_synthesis") is None
    assert snapshots[0]["tools"] is True
    assert executed == ["repository"] * 3 + [source]
    state = agent.audit.latest("model_evidence_state")
    assert state["tool_rounds"] == 4
    assert "Observation ID:" in snapshots[-1]["messages"]
    if source_ok:
        assert set(state["succeeded"]) == required
        assert not answer.startswith("[LocalPilot")
    else:
        failure = agent.audit.latest("model_evidence_acquisition_failed")
        assert failure["missing"] == [source]
        assert source in failure["attempted"] and source in failure["failed"]
        assert "unverified" in answer
        assert agent.audit.latest("model_same_context_postvalidation_complete")["accepted"]


@pytest.mark.parametrize("location_supplied", [False, True])
def test_verified_repository_memory_still_allows_synthesis_when_requirements_satisfied(
    tmp_path, monkeypatch, location_supplied
):
    agent, prompt = _synthesis_memory_agent(tmp_path)
    requirements = {"trusted repository"}
    if location_supplied:
        prompt += " Use my location for context."
        requirements.add("machine location")
        agent.machine_location = SimpleNamespace(
            public_status=lambda: {"enabled": True},
            coarse_model_context=lambda **kwargs: {"source": "test_location_provider", "approximate": True},
        )
    snapshots = []
    answer_text = "The requested repository symbols are present in the verified source."

    def fake_chat(**kwargs):
        snapshots.append({"tools": bool(kwargs.get("tools")), "messages": str(kwargs["messages"])})
        return iter([_chunk(content=answer_text)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    assert agent.ask(prompt) == answer_text
    assert agent._evidence_requirements(prompt) == requirements
    assert set(agent.audit.latest("model_learning_memory_live_verification")["succeeded"]) == requirements
    assert agent.audit.latest("model_learning_memory_direct_synthesis")["target_count"] == 3
    assert snapshots[0]["tools"] is False
    assert "Observation ID:" in snapshots[0]["messages"]
    assert agent.audit.latest("model_same_context_postvalidation_complete")["accepted"] is True


def test_mixed_repository_and_pc_request_keeps_normal_live_evidence_priority(tmp_path, monkeypatch):
    agent, prompt = _synthesis_memory_agent(tmp_path)
    prompt += " Check this PC's current CPU temperature using SystemSense."
    executed = []
    repository_read = agent.tools["search_repository"].fn

    def verify_repository(**kwargs):
        executed.append(kwargs["query"])
        return repository_read(**kwargs)

    def read_pc(**kwargs):
        executed.append("PC")
        return '{"cpu_temperature_c": 63}'

    agent.tools["search_repository"] = replace(agent.tools["search_repository"], fn=verify_repository)
    agent.tools["get_system_sense_summary"] = replace(agent.tools["get_system_sense_summary"], fn=read_pc)

    def fake_chat(**kwargs):
        if kwargs.get("tools") and len(executed) < 4:
            call = _call("get_system_sense_summary") if not executed else _call(
                "search_repository", {"path": "localpilot/agent.py", "query": f"RoutingClaim{len(executed) - 1}"}
            )
            return iter([_chunk(tool_calls=[call])])
        return iter([_chunk(content="The repository symbols were verified and the CPU temperature is 63 C.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    assert "63" in agent.ask(prompt)
    assert executed == ["PC", "RoutingClaim0", "RoutingClaim1", "RoutingClaim2"]
    assert agent.audit.latest("model_learning_memory_verification_deferred")["target_count"] == 3
    assert agent.audit.latest("model_learning_memory_direct_synthesis") is None
    state = agent.audit.latest("model_evidence_state")
    assert state["tool_rounds"] == 4
    assert set(state["succeeded"]) == {"trusted repository", "Windows/PC state"}


def test_completed_memory_verification_at_ceiling_cannot_bypass_missing_source(tmp_path, monkeypatch):
    agent, prompt = _synthesis_memory_agent(tmp_path, claim_count=4)
    prompt += " Read https://example.org/reference before answering."
    model_calls = []

    def fake_chat(**kwargs):
        model_calls.append(kwargs)
        return iter([_chunk(content="The repository symbols were verified; the reference remains unverified.")])

    def forbidden_extra_read(**kwargs):
        pytest.fail("The existing hard ceiling must not be increased for missing evidence.")

    agent.tools["fetch_public_https"] = replace(agent.tools["fetch_public_https"], fn=forbidden_extra_read)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(prompt)
    verification = agent.audit.latest("model_learning_memory_live_verification")
    assert verification["target_count"] == verification["tool_rounds"] == 4
    assert agent.audit.latest("model_learning_memory_direct_synthesis") is None
    assert not model_calls[0].get("tools")
    assert agent.audit.latest("model_evidence_acquisition_failed")["missing"] == ["public HTTPS"]
    assert "unverified" in answer
    assert "reference" in answer or "public source" in answer


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


def test_declared_dependency_query_prioritizes_pyproject_live_check(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["ollama>=0.6.0"]\n', encoding="utf-8")
    (tmp_path / "localpilot").mkdir()
    (tmp_path / "localpilot" / "agent.py").write_text(
        "from ollama import chat\n"
        "def _stream_chat_message():\n"
        "    return chat(stream=True)\n",
        encoding="utf-8",
    )
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
        "reason": "Read the live declared Ollama dependency.",
        "tool": "read_repository_file",
        "arguments": {
            "path": "pyproject.toml",
            "start_line": 1,
            "end_line": 120,
        },
    }
    assert targets[1]["tool"] == "search_repository"
    assert targets[1]["arguments"]["query"] == "from ollama import chat"
    assert targets[2]["tool"] == "search_repository"
    assert targets[2]["arguments"]["query"] == "_stream_chat_message("
    assert targets[3]["tool"] == "read_repository_file"
    assert targets[3]["arguments"] == {
        "path": "localpilot/agent.py",
        "start_line": 650,
        "end_line": 790,
    }
    snapshots = []
    grounded_answer = (
        "pyproject declares ollama>=0.6.0; _stream_chat_message calls chat(**kwargs), "
        "aggregates thinking, content, and tool_calls, and handles the verified ResponseError path."
    )

    def fake_chat(**kwargs):
        snapshots.append([dict(item) for item in kwargs["messages"]])
        return iter([_chunk(content=grounded_answer)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Verify the declared dependency for the Ollama streaming integration.")

    assert answer == grounded_answer
    assert "ollama>=0.6.0" in str(snapshots[0])
    assert "Never invent or rename a version" in str(snapshots[-1])
    assert agent.audit.latest("model_learning_memory_live_verification")["target_count"] == 4
    assert agent.audit.latest("model_learning_memory_direct_synthesis")["target_count"] == 4
    postvalidation = agent.audit.latest("model_same_context_postvalidation_complete")
    assert postvalidation["accepted"] is True
    assert postvalidation["prose_rewritten"] is False
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
            [_chunk(content="Live source controls: auto promotion is false and merge remains human-only.")],
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
    assert "tools" not in calls[4]
    assert "tools" not in calls[5]
    assert all(item["think"] == "high" for item in calls[:5])
    assert calls[5]["think"] == "high"
    budget = agent.audit.latest("model_learning_memory_research_budget")
    assert budget["soft_tool_rounds"] == 4
    assert budget["hard_tool_rounds"] == 4
    assert agent.audit.latest("model_research_soft_budget") is None


def test_operator_information_paths_and_literal_authority_are_explicit():
    assert "Ordinary operator tool observations are turn-local raw evidence" in SYSTEM_PROMPT
    assert "Sharing a database class does not establish an automatic data flow" in SYSTEM_PROMPT
    assert "Never invent a product version, symbol, file, import, call path" in SYSTEM_PROMPT
    assert "has no merge or promotion method" in SYSTEM_PROMPT
    assert LocalPilotAgent._information_authority_risks(
        "After each interaction the operator records lessons."
    ) == ["automatic_operator_learning"]
    assert LocalPilotAgent._information_authority_risks(
        "The operator research loop may invoke upsert_knowledge_facts."
    ) == ["operator_writes_study_facts"]
    assert LocalPilotAgent._information_authority_risks(
        "Ordinary operator observations remain turn-local; the separate StudyEngine writes facts."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The operator loop never calls upsert_knowledge_facts. CommandRunner is not the wrapper "
        "for every tool. GitHub Actions does not merge or promote."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The operator research loop does not write staged facts. A separate durable learning "
        "section explains that StudyEngine alone calls upsert_knowledge_facts."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The operator research loop never writes staged knowledge facts."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The operator research loop collects raw results and, if needed, records lessons or "
        "knowledge facts into LearningMemory."
    ) == ["operator_writes_study_facts"]
    assert LocalPilotAgent._information_authority_risks(
        "Not all tools are wrapped by CommandRunner."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "Reconciliation preserves candidate branch and GitHub history rather than clearing or deleting it."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The candidate branch remains intact while a clean registered matching worktree is removed."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "GitHub Actions automatically merges the pull request."
    ) == ["github_actions_merges"]
    assert LocalPilotAgent._information_authority_risks(
        "Facts are written by record_human_lesson and upsert_knowledge_facts."
    ) == ["human_lesson_as_knowledge_fact"]
    assert LocalPilotAgent._information_authority_risks(
        "record_human_lesson creates a separate HumanLesson, not a knowledge_fact."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "Targeted live verification is performed only when a digest mismatch is detected."
    ) == ["verification_only_on_digest_mismatch"]
    assert LocalPilotAgent._information_authority_risks(
        "The stable operator records observations only via explicit /teach or record_human_lesson."
    ) == ["teach_records_observations"]
    assert LocalPilotAgent._information_authority_risks(
        "The operator's safety policy governs all tool calls."
    ) == ["operator_policy_governs_all_tools"]
    assert LocalPilotAgent._information_authority_risks(
        "The safety policy ensures that any tool call respects candidate boundaries."
    ) == ["operator_policy_governs_all_tools"]
    assert LocalPilotAgent._information_authority_risks(
        "All interactions are governed by the safety policy."
    ) == ["operator_policy_governs_all_tools"]
    assert LocalPilotAgent._information_authority_risks(
        "Normal operator tools use SafetyPolicy; CandidateTools enforce separate confinement."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "LearningMemory is written to only by explicit /teach calls or staged-study updates."
    ) == ["learning_memory_only_teach_study"]
    assert LocalPilotAgent._information_authority_risks(
        "LearningMemory stores lessons, knowledge facts, and candidate-cycle outcomes. It is "
        "updated only through explicit API calls (record_human_lesson, upsert_knowledge_facts)."
    ) == ["learning_memory_only_teach_study"]
    assert LocalPilotAgent._information_authority_risks(
        "LearningMemory is separate from turn-local observations; only explicit writes "
        "(/teach, record_human_lesson, upsert_knowledge_facts) persist data."
    ) == ["learning_memory_only_teach_study"]
    assert LocalPilotAgent._information_authority_risks(
        "LearningMemory stores separate HumanLesson, knowledge_facts, and self-development cycle records."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "After a candidate PR is merged, GitHub Actions run CI."
    ) == ["ci_after_human_merge"]
    assert LocalPilotAgent._information_authority_risks(
        "GitHub Actions CI passes before the authorized human merge."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "The stable operator and developer operate under the normal safety policy."
    ) == ["developer_uses_operator_policy"]
    assert LocalPilotAgent._information_authority_risks(
        "Normal operator tools use SafetyPolicy; self-development uses bounded research and CandidateTools."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "Candidate code is never executed locally; only the operator's own code runs."
    ) == ["developer_local_process_erased"]
    assert LocalPilotAgent._information_authority_risks(
        "Self-development relies on the same safety boundaries that the operator enforces."
    ) == ["developer_uses_operator_policy"]
    assert LocalPilotAgent._information_authority_risks(
        "Candidate changes are never committed to GitHub until a PR is merged."
    ) == ["candidate_commit_after_merge"]
    assert LocalPilotAgent._information_authority_risks(
        "Candidate commit and push precede PR creation, CI, and human merge."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "Candidate code is prohibited from local execution; only the developer process runs locally."
    ) == ["stable_operator_local_process_erased"]
    assert LocalPilotAgent._information_authority_risks(
        "Both stable operator and developer run locally; candidate code does not."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "record_human_lesson is the only place where a human lesson is written, and "
        "upsert_knowledge_facts is the only place where facts are written."
    ) == ["exclusive_learning_writer"]
    assert LocalPilotAgent._information_authority_risks(
        "The inspected operator path calls record_human_lesson; the inspected StudyEngine path "
        "calls upsert_knowledge_facts."
    ) == []
    assert LocalPilotAgent._information_authority_risks(
        "Self-development is triggered by the ResourceGovernor; only the stable operator runs locally; "
        "the candidate branch is cleared after merge."
    ) == [
        "resource_governor_triggers_evolution",
        "candidate_branch_history_cleared",
        "developer_local_process_erased",
    ]
    assert LocalPilotAgent._strip_authority_meta(
        "Grounded answer.\n\nAll statements above are directly supported.\n"
        "No additional classes are inferred."
    ) == "Grounded answer."
    assert LocalPilotAgent._strip_authority_meta(
        "Grounded answer.\n\nThis summary reflects only verified evidence."
    ) == "Grounded answer."


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
