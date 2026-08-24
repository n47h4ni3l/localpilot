from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from localpilot.cli import _show_study_status
from localpilot.config import Config
from localpilot.learning import LearningMemory
from localpilot.study import (
    RepositoryGroundingValidator,
    StudyEngine,
    benchmark_question_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def _metadata(model: str) -> dict:
    return {
        "details": {"family": "qwen2", "quantization_level": "Q4_K_M"},
        "capabilities": ["completion", "tools"],
        "modified_at": "2026-08-23T00:00:00Z",
        "model_info": {"qwen2.context_length": 32768},
    }


def _engine(tmp_path: Path) -> tuple[StudyEngine, LearningMemory]:
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    engine = StudyEngine(ROOT, memory, Config(), model_metadata=_metadata)
    return engine, memory


def test_knowledge_map_persists_provenance_confidence_and_relationships(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.upsert_knowledge_fact(
        stage="self",
        fact_key="symbol:localpilot.cli:main",
        fact_type="symbol",
        subject="localpilot.cli:main",
        summary="CLI entry point.",
        source_uri="repo://localpilot/cli.py",
        source_kind="python_ast",
        source_digest="abc123",
        confidence=0.95,
        relationships=("localpilot/cli.py", "localpilot.config:load_config"),
    )

    fact = LearningMemory(tmp_path / "learning.sqlite3").knowledge_fact(
        "self", "symbol:localpilot.cli:main"
    )

    assert fact is not None
    assert fact.source_uri == "repo://localpilot/cli.py"
    assert fact.source_kind == "python_ast"
    assert fact.confidence == 0.95
    assert fact.last_verified_at
    assert fact.relationships == (
        "localpilot/cli.py",
        "localpilot.config:load_config",
    )


def test_stage_order_requires_measured_improvement(tmp_path: Path):
    engine, _ = _engine(tmp_path)

    with pytest.raises(RuntimeError, match="Stage qwen is locked"):
        engine.baseline("qwen")

    self_outcome = engine.run_stage("self")
    assert self_outcome.state.status == "improved"
    assert engine.baseline("qwen").phase == "baseline"


def test_each_stage_records_baseline_before_post_study_gain(tmp_path: Path):
    engine, memory = _engine(tmp_path)

    outcomes = engine.run_all()

    assert [outcome.stage for outcome in outcomes] == ["self", "qwen", "python"]
    for outcome in outcomes:
        assert outcome.baseline.phase == "baseline"
        assert outcome.latest.phase == "post_study"
        assert outcome.latest.score > outcome.baseline.score
        assert outcome.latest.question_set_digest == outcome.baseline.question_set_digest
        assert memory.curriculum_state(outcome.stage).status == "improved"


def test_held_out_question_identifiers_are_not_persisted_as_study_facts(tmp_path: Path):
    engine, memory = _engine(tmp_path)
    engine.run_stage("self")
    facts = memory.knowledge_facts(stage="self")
    held_out = set(benchmark_question_ids("self"))

    assert held_out
    assert not held_out & {fact.fact_key for fact in facts}
    assert all(
        question_id not in fact.summary
        for question_id in held_out
        for fact in facts
    )


def test_stale_repository_sources_are_invalidated(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("value = 2\n", encoding="utf-8")
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.upsert_knowledge_fact(
        stage="self",
        fact_key="symbol:sample:value",
        fact_type="symbol",
        subject="sample:value",
        summary="Old fact.",
        source_uri="repo://sample.py",
        source_kind="python_ast",
        source_digest="old-digest",
        confidence=1.0,
    )
    engine = StudyEngine(tmp_path, memory, Config(), model_metadata=_metadata)

    assert engine.refresh_stale_sources() == 1
    assert memory.knowledge_facts(stage="self") == []
    assert memory.knowledge_fact("self", "symbol:sample:value").stale


def test_repository_grounding_detects_recent_failure_classes(tmp_path: Path):
    engine, memory = _engine(tmp_path)
    engine.run_stage("self")

    report = RepositoryGroundingValidator(memory).validate(
        {
            "referenced_symbols": ["localpilot.selfdev:ImaginarySupervisor"],
            "referenced_config_fields": ["selfdev.supervisor_interval"],
            "referenced_paths": ["scripts/pre_ci_review.sh"],
            "planned_subsystems": ["checkpoint"],
            "new_runtime_paths": ["localpilot/idle_supervisor.py"],
            "integration_points": ["localpilot.selfdev:missing_hook"],
            "expected_call_relationships": [
                ["localpilot.cli:main", "ImaginarySupervisor"]
            ],
            "required_test_contracts": ["test_missing_supervisor_contract"],
        }
    )
    codes = {issue.code for issue in report.issues}

    assert not report.grounded
    assert codes == {
        "nonexistent_api",
        "nonexistent_config_field",
        "missing_file_or_command",
        "duplicate_existing_subsystem",
        "wrong_integration_point",
        "call_graph_mismatch",
        "missing_test_contract",
    }


def test_status_shows_stage_scores_weak_areas_and_next_lesson(tmp_path: Path):
    engine, memory = _engine(tmp_path)
    engine.baseline("self")
    console = Console(record=True, width=180)

    _show_study_status(console, memory)
    output = console.export_text()

    assert "baseline_recorded" in output
    assert "symbol and capability ownership" in output
    assert "Study self sources" in output


def test_no_gain_records_adaptation_instead_of_completion(tmp_path: Path):
    engine, memory = _engine(tmp_path)
    engine._study_self()

    outcome = engine.run_stage("self")

    assert outcome.baseline.score == outcome.latest.score == 100.0
    assert outcome.state.status == "needs_adaptation"
    assert "do not mark it complete" in outcome.state.next_lesson


def test_source_failure_is_recorded_for_adaptation(tmp_path: Path, monkeypatch):
    engine, _ = _engine(tmp_path)

    def fail_study() -> list[str]:
        raise ValueError("authoritative source changed")

    monkeypatch.setattr(engine, "_study_self", fail_study)
    outcome = engine.run_stage("self")

    assert outcome.state.status == "needs_adaptation"
    assert any("study_source_error" in error for error in outcome.latest.errors)


def test_peer_comparison_scores_transfer_tasks_without_storing_raw_answers(tmp_path: Path):
    def chat(model: str, messages: list[dict[str, str]]) -> str:
        if model == "peer:smart":
            prompt = messages[-1]["content"]
            if "issue_codes" in prompt:
                return '{"issue_codes":["nonexistent_api","nonexistent_config_field","missing_file_or_command","disconnected_code"]}'
            if "subprocess" in prompt:
                return '{"argv":true,"shell_false":true,"timeout":true}'
            return '{"authoritative_local_source":"api/show","resource_metrics":["context","duration"]}'
        return "{}"

    memory = LearningMemory(tmp_path / "learning.sqlite3")
    engine = StudyEngine(
        ROOT,
        memory,
        Config(),
        model_metadata=_metadata,
        model_chat=chat,
    )

    result = engine.compare_models("peer:smart", subject_model="subject:small")

    assert result.peer_score == 100.0
    assert result.subject_score == 0.0
    assert result.resource_cost["raw_responses_stored"] is False
    assert "Peer model scored higher" in result.transferable_lessons[0]


def test_qwen_facts_prefer_local_metadata_and_authoritative_sources(tmp_path: Path):
    engine, memory = _engine(tmp_path)
    engine.run_stage("self")
    outcome = engine.run_stage("qwen")

    assert outcome.latest.score == 100.0
    metadata = memory.knowledge_fact("qwen", "qwen:installed_model_metadata")
    tool_fact = memory.knowledge_fact("qwen", "qwen:tool_calling")
    assert metadata is not None and metadata.source_kind == "local_ollama_metadata"
    assert tool_fact is not None
    assert tool_fact.source_uri.startswith("https://docs.ollama.com/")
    assert tool_fact.confidence == 0.9


def test_qwen_stage_accepts_typed_ollama_show_response(tmp_path: Path, monkeypatch):
    details = SimpleNamespace(family="qwen2", quantization_level="Q4_K_M")
    response = SimpleNamespace(
        details=details,
        model_dump=lambda **kwargs: {
            "details": {
                "family": details.family,
                "quantization_level": details.quantization_level,
            },
            "capabilities": ["completion", "tools"],
            "modified_at": "2026-08-25T00:00:00Z",
            "model_info": {"qwen2.context_length": 32768},
        },
    )
    monkeypatch.setattr("ollama.show", lambda model: response)
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    engine = StudyEngine(ROOT, memory, Config())

    engine.run_stage("self")
    outcome = engine.run_stage("qwen")
    metadata = memory.knowledge_fact("qwen", "qwen:installed_model_metadata")

    assert outcome.state.status == "improved"
    assert metadata is not None
    assert '"family": "qwen2"' in metadata.summary
    assert '"modified_at": "2026-08-25T00:00:00Z"' in metadata.summary


def test_broad_web_research_is_opt_in_and_rejects_private_hosts(tmp_path: Path):
    engine, memory = _engine(tmp_path)

    with pytest.raises(RuntimeError, match="disabled"):
        engine.inspect_web_source("https://example.org/research")

    allowed = StudyEngine(
        ROOT,
        memory,
        Config(),
        allow_web=True,
        model_metadata=_metadata,
    )
    with pytest.raises(ValueError, match="Local/private"):
        allowed.inspect_web_source("https://localhost/research")
    assert memory.knowledge_facts() == []
