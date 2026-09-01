from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from localpilot.config import Config
from localpilot.evolution_orchestrator import (
    EvolutionBudgetExceeded,
    EvolutionRunBudget,
    OpportunityLedger,
    opportunity_similarity,
)
from localpilot.selfdev import CyclePaused, SelfDeveloper


def _task(task_id: str = "capability-reliable-retrieval") -> dict:
    return {
        "id": task_id,
        "title": "Measure reliable memory retrieval",
        "status": "todo",
        "source": "capability_discovery",
        "evolution_class": "improve_cognition",
        "capability_target": "Reliable retrieval from durable learning memory",
        "mission_alignment": "Improve future work without changing ordinary conversation style.",
        "current_frontier": "Retrieval quality is not measured consistently.",
        "why_high_leverage": "Many future tasks depend on recalling verified lessons.",
        "capability_unlocked": "Evidence-backed reuse across future tasks.",
        "next_frontier": "Measure whether retrieved lessons improve task outcomes.",
        "question": "Can a fixed retrieval benchmark reduce missed relevant lessons?",
        "observed_limitation": "Relevant stored lessons are sometimes omitted from planning.",
        "evidence": ["localpilot/learning.py owns durable retrieval."],
        "alternatives": ["Tune lexical retrieval.", "Add a reranking evaluation."],
        "hypothesis": "A fixed benchmark will detect and reduce missed relevant lessons.",
        "expected_complexity": "low",
        "evaluation": {
            "metric": "relevant lesson recall",
            "baseline": "Measure the current fixed-query recall rate.",
            "success_criterion": "Improve recall without reducing precision.",
            "measurement_method": "Run the same fixed queries before and after the candidate.",
        },
        "acceptance": ["Report the before and after recall rate."],
    }


def test_whole_cycle_budget_persists_and_bounds_tool_and_web_calls(tmp_path: Path):
    now = 10.0

    def monotonic() -> float:
        return now

    budget = EvolutionRunBudget(
        invocation_id="run-1",
        state_path=tmp_path / "run.json",
        wall_clock_seconds=30,
        max_tool_calls=3,
        max_web_calls=1,
        monotonic=monotonic,
    )
    budget.consume_tool("read_project_file", "research")
    budget.consume_tool("search_public_web", "research")

    with pytest.raises(EvolutionBudgetExceeded, match="web-call budget"):
        budget.consume_tool("fetch_public_https", "research")

    state = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "budget_exhausted"
    assert state["usage"] == {"tool_calls": 2, "web_calls": 1}
    assert state["limits"]["wall_clock_seconds"] == 30


def test_whole_cycle_budget_stops_long_inference_at_wall_clock_limit(tmp_path: Path):
    now = 0.0
    budget = EvolutionRunBudget(
        invocation_id="run-2",
        state_path=tmp_path / "run.json",
        wall_clock_seconds=30,
        max_tool_calls=4,
        max_web_calls=2,
        monotonic=lambda: now,
    )
    now = 31.0

    with pytest.raises(EvolutionBudgetExceeded, match="wall-clock budget"):
        budget.check("capability-discovery:inference")


def test_opportunity_ledger_persists_queue_and_rejects_near_duplicates(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    first = _task()
    duplicate = _task("capability-retrieval-copy")
    duplicate["title"] = "Benchmark durable memory retrieval"
    duplicate["hypothesis"] = "A fixed retrieval benchmark can reduce missed relevant lessons."

    assert opportunity_similarity(first, duplicate) >= 0.82
    assert ledger.enqueue(first, score=10) is True
    assert ledger.enqueue(duplicate, score=12) is False

    queued = OpportunityLedger(tmp_path / "opportunities.json").next_task()
    assert queued is not None
    assert queued["id"] == first["id"]
    assert queued["evaluation"]["metric"] == "relevant lesson recall"
    assert queued["evidence"] == first["evidence"]


def test_capability_discovery_uses_durable_queue_before_loading_model(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    task = _task()
    assert developer.opportunities.enqueue(task, score=9)

    selected = developer._discover_capability_task(
        developer_model="model-that-must-not-load",
        force=True,
    )

    assert selected["id"] == task["id"]
    event = developer.audit.latest("capability_opportunity_selected")
    assert event["source"] == "durable_queue"


def test_tool_stage_enforces_global_budget_across_multi_call_response(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    developer._budget = EvolutionRunBudget(
        invocation_id="run-3",
        state_path=developer.data_dir / "run.json",
        wall_clock_seconds=60,
        max_tool_calls=1,
        max_web_calls=1,
    )
    monkeypatch.setattr(developer, "_check_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        developer,
        "_developer_chat",
        lambda *_args, **_kwargs: SimpleNamespace(
            message={
                "content": "",
                "tool_calls": [
                    {"function": {"name": "inspect", "arguments": {}}},
                    {"function": {"name": "inspect", "arguments": {}}},
                ],
            }
        ),
    )

    def inspect() -> str:
        return "ok"

    with pytest.raises(CyclePaused, match="tool-call budget"):
        developer._tool_stage(
            chat=lambda **_kwargs: None,
            model="test",
            messages=[],
            functions=[inspect],
            rounds=1,
            force=True,
            branch="capability-discovery",
            stage="capability_discovery",
        )

    assert developer.audit.latest("selfdev_budget_exhausted")["tool"] == "inspect"
