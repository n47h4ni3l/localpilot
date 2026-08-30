import json
from pathlib import Path

import pytest

from localpilot.checkpoint import CheckpointStore, EvolutionCheckpoint
from localpilot.config import Config
from localpilot.evolution import (
    EvolutionClass,
    capability_proposal_score,
    classify_evolution_task,
    normalize_evolution_task,
    parse_capability_proposals,
    select_capability_proposal,
)
from localpilot.github_integration import MainSyncResult
from localpilot.learning import LearningMemory
from localpilot.mission import mission_context
from localpilot.resource import ResourceState
from localpilot.selfdev import DeveloperModelSelection, EvolutionResult, SelfDeveloper


def _proposal(*, complexity: str = "low", measured: bool = True) -> dict:
    evaluation = {
        "metric": "planner task completion rate",
        "baseline": "6 of 10 held-out tasks complete",
        "success_criterion": "at least 8 of 10 tasks complete with no safety regression",
        "measurement_method": "a held-out GitHub CI benchmark",
    }
    if not measured:
        evaluation.pop("baseline")
    return {
        "evolution_class": "improve_cognition",
        "title": "Evaluate structured plan checkpoints",
        "capability_target": "long-horizon planning",
        "mission_alignment": (
            "Improve transferable long-horizon reasoning while preserving human control."
        ),
        "current_frontier": (
            "LocalPilot can resume durable implementation state but still loses some higher-level task structure."
        ),
        "why_high_leverage": (
            "Better long-horizon planning transfers across research, coding, tool learning, and future capabilities."
        ),
        "capability_unlocked": (
            "More reliable multi-session autonomous research and implementation."
        ),
        "next_frontier": (
            "Generalize durable planning into hierarchical self-directed research programs."
        ),
        "question": "Can structured plan checkpoints reduce lost intermediate state?",
        "observed_limitation": "Recent paused cycles lose useful task structure.",
        "evidence": ["Three recent cycles paused before delivery."],
        "alternatives": ["larger context", "structured checkpoints"],
        "hypothesis": "Structured plan checkpoints will raise held-out completion from 60% to at least 80%.",
        "expected_complexity": complexity,
        "evaluation": evaluation,
    }


def test_four_evolution_classes_are_first_class():
    assert classify_evolution_task({"evolution_class": "repair"}) is EvolutionClass.REPAIR
    assert classify_evolution_task({"evolution_class": "extend"}) is EvolutionClass.EXTEND
    assert (
        classify_evolution_task({"evolution_class": "improve_cognition"})
        is EvolutionClass.IMPROVE_COGNITION
    )
    assert classify_evolution_task({"evolution_class": "explore"}) is EvolutionClass.EXPLORE


def test_unmeasured_complexity_is_rejected_and_measured_complexity_is_downranked():
    with pytest.raises(ValueError, match="unmeasured or incomplete"):
        parse_capability_proposals(json.dumps({"proposals": [_proposal(measured=False)]}))

    low, high = parse_capability_proposals(
        json.dumps({"proposals": [_proposal(complexity="low"), _proposal(complexity="high")]})
    )
    assert capability_proposal_score(low) > capability_proposal_score(high)
    assert select_capability_proposal([high, low]) == low


def test_capability_experiment_persists_reviewable_evidence_and_rebuilds_task(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    proposal = parse_capability_proposals(json.dumps(_proposal()))[0]
    task = proposal.task("capability-planning-test")

    experiment_id = memory.record_experiment(task)
    cycle_id = memory.start_cycle(
        task_id=task["id"],
        branch="localpilot/candidate-capability-planning-test",
        everyday_model="daily",
        developer_model="developer",
        workspace=tmp_path / "candidate",
        is_worktree=True,
    )
    memory.attach_experiment_cycle(task["id"], cycle_id, "localpilot/candidate-capability-planning-test")
    memory.update_experiment_outcome(
        task["id"],
        status="candidate_pending_validation",
        outcome="pending_ci",
        before_evidence="6 of 10",
        after_evidence="CI benchmark added",
        reusable_lesson="Use held-out tasks for planner changes.",
    )

    reopened = LearningMemory(memory.path)
    experiment = reopened.latest_experiment()
    assert experiment is not None
    assert experiment.id == experiment_id
    assert experiment.hypothesis.startswith("Structured plan checkpoints")
    assert experiment.before_evidence == "6 of 10"
    rebuilt = reopened.experiment_task(task["id"])
    assert rebuilt is not None
    assert rebuilt["evaluation"]["success_criterion"].startswith("at least 8")
    proposed_context = reopened.discovery_context()
    assert proposed_context["capabilities"][0]["known_limitation"].startswith("Recent paused")
    reopened.update_experiment_review(task["id"], validation_state="passed", merged=True)
    context = reopened.discovery_context()
    assert context["capabilities"][0]["name"] == "long-horizon planning"
    assert reopened.latest_experiment().status == "validated"
    assert not reopened.schema_columns() & {
        "reasoning",
        "chain_of_thought",
        "thinking",
        "prompt",
        "transcript",
        "messages",
        "raw_tokens",
    }


def test_mission_and_frontier_are_durable_and_reviewable(tmp_path: Path):
    mission = mission_context()
    assert "general-purpose personal intelligence" in mission["mission"]
    assert "transferable general capability" in mission["evolution_objective"]
    assert any("human" in item for item in mission["non_goals"])

    memory = LearningMemory(tmp_path / "learning.sqlite3")
    proposal = parse_capability_proposals(json.dumps(_proposal()))[0]
    task = proposal.task("capability-frontier-test")
    memory.record_experiment(task)

    frontier = memory.latest_frontier()
    assert frontier is not None
    assert frontier.task_id == task["id"]
    assert frontier.current_frontier == task["current_frontier"]
    assert "multi-session" in frontier.capability_unlocked
    assert "hierarchical" in frontier.next_frontier

    rebuilt = memory.experiment_task(task["id"])
    assert rebuilt is not None
    assert rebuilt["mission_alignment"] == frontier.mission_alignment
    assert rebuilt["current_frontier"] == frontier.current_frontier
    assert rebuilt["next_frontier"] == frontier.next_frontier

    context = memory.discovery_context()
    assert context["frontiers"][0]["task_id"] == task["id"]


def test_checkpoint_preserves_capability_contract_across_resume(tmp_path: Path):
    task = normalize_evolution_task(
        parse_capability_proposals(json.dumps(_proposal()))[0].task("capability-resume")
    )
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=11,
        task=task,
        branch="localpilot/candidate-capability-resume",
        workspace=tmp_path / "candidate",
        milestone="research_complete",
        next_action="Implement the measured candidate.",
    )
    store.save(checkpoint)

    loaded = store.load()
    assert loaded is not None
    assert loaded.evolution_class == "improve_cognition"
    assert loaded.capability_target == "long-horizon planning"
    assert "60% to at least 80%" in loaded.hypothesis
    assert "planner task completion rate" in loaded.evaluation_plan


def test_idle_without_blocking_candidate_discovers_its_own_growth_question(
    tmp_path: Path, monkeypatch
):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    discovered = normalize_evolution_task(
        parse_capability_proposals(json.dumps(_proposal()))[0].task("capability-self-chosen")
    )
    captured = {}

    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: MainSyncResult(True, False, "current"),
    )
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: ResourceState(1000, 5, 20, True, "idle", True, True),
    )
    monkeypatch.setattr(developer.governor, "apply_process_priority", lambda idle: None)
    monkeypatch.setattr(developer, "_reconcile_candidates", lambda: None)
    monkeypatch.setattr(developer, "_repair_local_candidate", lambda force=False: None)
    monkeypatch.setattr(developer, "_repair_failed_candidate", lambda force=False: None)
    monkeypatch.setattr(developer, "_load_next_task", lambda: None)
    monkeypatch.setattr(developer.memory, "has_outstanding_candidate", lambda: False)
    monkeypatch.setattr(
        developer,
        "_select_developer_model",
        lambda: DeveloperModelSelection("developer", 1, 1.0, "fits"),
    )
    monkeypatch.setattr(
        developer,
        "_discover_capability_task",
        lambda **_kwargs: discovered,
    )
    monkeypatch.setattr(developer, "_candidate_workspace", lambda _branch: (workspace, True))

    def continue_candidate(**kwargs):
        captured.update(kwargs)
        return EvolutionResult("paused", kwargs["branch"], workspace, "paused")

    monkeypatch.setattr(developer, "_continue_candidate", continue_candidate)

    result = developer._run_once(force=True)

    assert result.status == "paused"
    assert captured["task"]["source"] == "capability_discovery"
    assert captured["task"]["hypothesis"]
    assert captured["task"]["evaluation"]["baseline"]
    assert developer.memory.latest_experiment().capability_target == "long-horizon planning"


def test_outstanding_candidate_gate_prevents_parallel_experiment(tmp_path: Path, monkeypatch):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: MainSyncResult(True, False, "current"),
    )
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: ResourceState(1000, 5, 20, True, "idle", True, True),
    )
    monkeypatch.setattr(developer.governor, "apply_process_priority", lambda idle: None)
    monkeypatch.setattr(developer, "_reconcile_candidates", lambda: None)
    monkeypatch.setattr(developer, "_repair_local_candidate", lambda force=False: None)
    monkeypatch.setattr(developer, "_repair_failed_candidate", lambda force=False: None)
    monkeypatch.setattr(developer, "_load_next_task", lambda: None)
    monkeypatch.setattr(developer.memory, "has_outstanding_candidate", lambda: True)
    monkeypatch.setattr(
        developer,
        "_discover_capability_task",
        lambda **_kwargs: pytest.fail("discovery must remain behind the one-candidate gate"),
    )

    result = developer._run_once(force=True)

    assert result.status == "idle"
    assert "One candidate" in result.summary


def test_candidate_review_reconciliation_runs_even_when_owner_activity_defers_model_work(
    tmp_path: Path, monkeypatch
):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    calls = []
    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: MainSyncResult(True, False, "current"),
    )
    monkeypatch.setattr(developer, "_reconcile_candidates", lambda: calls.append("reconciled"))
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: ResourceState(
            0,
            5,
            20,
            False,
            "owner active",
            idle_allowed=False,
            capacity_allowed=True,
            idle_reason="owner active",
        ),
    )
    monkeypatch.setattr(developer.governor, "apply_process_priority", lambda idle: None)

    result = developer._run_once()

    assert result.status == "deferred"
    assert calls == ["reconciled"]


def test_capability_discovery_exposes_permissionless_public_web_research(
    tmp_path: Path, monkeypatch
):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    captured = {}
    monkeypatch.setattr(developer.github, "tracked_project_paths", lambda: set())

    def fake_stage(**kwargs):
        captured["function_names"] = [function.__name__ for function in kwargs["functions"]]
        captured["messages"] = kwargs["messages"]
        return json.dumps({"proposals": [_proposal()]})

    monkeypatch.setattr(developer, "_tool_stage", fake_stage)

    task = developer._discover_capability_task(developer_model="developer", force=True)

    assert task["capability_target"] == "long-horizon planning"
    assert "search_public_web" in captured["function_names"]
    assert "fetch_public_https" in captured["function_names"]
    assert "untrusted evidence" in str(captured["messages"])
