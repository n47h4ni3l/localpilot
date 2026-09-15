from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from localpilot.checkpoint import CheckpointStore, EvolutionCheckpoint, task_fingerprint
from localpilot.config import Config
from localpilot.evolution_orchestrator import EvolutionRunAlreadyActive, EvolutionRunLease
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.selfdev import SelfDeveloper as BaseSelfDeveloper
from localpilot.selfdev_response_parsing import StaticRepairResult, _json_object
from localpilot.selfdev_results import EvolutionResult


def _task() -> dict:
    return {
        "id": "context-token-budget",
        "title": "Improve context token budgeting",
        "acceptance": ["Preserve context tokens across resumed work."],
        "evolution_class": "improve_cognition",
        "capability_target": "Token-aware context budgeting",
        "mission_alignment": "Improve reliable long-form engineering work.",
        "current_frontier": "A fixed context budget is used.",
        "why_high_leverage": "Many future tasks need reliable context use.",
        "capability_unlocked": "Measured context allocation.",
        "next_frontier": "Adaptive context allocation.",
        "question": "Can context tokens be budgeted from task evidence?",
        "observed_limitation": "Long tasks can exhaust context tokens.",
        "evidence": ["Current configuration uses a fixed context token count."],
        "alternatives": ["Keep the fixed budget", "Add measured allocation"],
        "hypothesis": "Measured token allocation reduces truncated work.",
        "evaluation": {
            "metric": "completed long-form tasks",
            "baseline": "fixed token budget",
            "success_criterion": "no regression and fewer truncations",
            "measurement_method": "held-out long-context evaluation",
        },
        "expected_complexity": "medium",
    }


def _developer(tmp_path: Path) -> SelfDeveloper:
    config = Config()
    config.agent.data_dir = "data"
    return SelfDeveloper(config, tmp_path)


def test_json_object_uses_last_complete_outer_object() -> None:
    text = (
        'Example: {"ignore": true}\n'
        'Final answer: {"summary": "ok", "nested": {"value": 3}}\n'
        'done'
    )

    assert _json_object(text) == {
        "summary": "ok",
        "nested": {"value": 3},
    }


def test_checkpoint_keeps_benign_token_language_and_500_changed_paths(tmp_path: Path) -> None:
    task = _task()
    changed = [f"generated/file_{index:03d}.py" for index in range(500)]
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=7,
        task=task,
        branch="localpilot/candidate-context-token-budget-20260915-000000",
        workspace=tmp_path,
        milestone="static_checks",
        files_inspected=changed,
        files_changed=changed,
        git_head="a" * 40,
        git_state_digest="b" * 64,
    )
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(checkpoint)
    loaded = store.load()

    assert loaded is not None
    assert loaded.capability_target == "Token-aware context budgeting"
    assert loaded.hypothesis == "Measured token allocation reduces truncated work."
    assert loaded.files_changed == tuple(changed)
    assert loaded.files_inspected == tuple(changed)


def test_checkpoint_fingerprint_covers_evaluation_contract() -> None:
    original = _task()
    changed = json.loads(json.dumps(original))
    changed["evaluation"]["success_criterion"] = "strictly better held-out score"

    assert task_fingerprint(original) != task_fingerprint(changed)


def test_checkpoint_rebind_changes_only_cycle_identity(tmp_path: Path) -> None:
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=11,
        task=_task(),
        branch="localpilot/candidate-context-token-budget-20260915-000000",
        workspace=tmp_path,
        milestone="local_static_repair",
        git_head="a" * 40,
        git_state_digest="b" * 64,
    )

    rebound = checkpoint.rebind_cycle(12)

    assert rebound.cycle_id == 12
    assert rebound.branch == checkpoint.branch
    assert rebound.git_head == checkpoint.git_head
    assert rebound.git_state_digest == checkpoint.git_state_digest
    assert rebound.task_fingerprint == checkpoint.task_fingerprint


def test_evolution_run_lease_blocks_live_owner_and_then_releases(tmp_path: Path) -> None:
    path = tmp_path / "evolution-run.lock"
    first = EvolutionRunLease(path, "first")
    second = EvolutionRunLease(path, "second")

    first.acquire()
    try:
        with pytest.raises(EvolutionRunAlreadyActive, match="already active"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    assert second.acquired is True
    second.release()
    assert not path.exists()


def test_evolution_run_lease_recovers_stale_owner(tmp_path: Path) -> None:
    path = tmp_path / "evolution-run.lock"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "invocation_id": "dead",
                "pid": 999_999_999,
                "process_create_time": 1.0,
            }
        ),
        encoding="utf-8",
    )

    lease = EvolutionRunLease(path, "replacement")
    lease.acquire()
    try:
        assert lease.acquired is True
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["invocation_id"] == "replacement"
    finally:
        lease.release()


def test_production_evolve_defers_when_another_process_owns_lease(tmp_path: Path) -> None:
    developer = _developer(tmp_path)
    lease = EvolutionRunLease(tmp_path / "data" / "evolution-run.lock", "owner")
    lease.acquire()
    try:
        result = developer.run_once(force=True)
    finally:
        lease.release()

    assert result.status == "deferred"
    assert "already active" in result.summary


def test_same_worktree_policy_retry_preserves_checkpoint(tmp_path: Path, monkeypatch) -> None:
    developer = _developer(tmp_path)
    assert type(developer) is SelfDeveloper
    assert developer.retry_candidate.__func__ is SelfDeveloper.retry_candidate
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    branch = "localpilot/candidate-context-token-budget-20260915-000000"
    cycle_id = developer.memory.start_cycle(
        task_id=_task()["id"],
        branch=branch,
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )
    developer.memory.finish_cycle(
        cycle_id,
        status="candidate_needs_work",
        summary="Static repair did not complete.",
        reusable_lesson="Preserve framework failure evidence.",
        checks_passed=False,
        pushed=False,
    )
    developer.memory.record_write_integrity_failure(
        cycle_id,
        "Framework policy blocked the structured static-repair contract.",
    )
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=cycle_id,
        task=_task(),
        branch=branch,
        workspace=workspace,
        milestone="local_static_repair",
        files_changed=["localpilot/example.py"],
        git_head="a" * 40,
        git_state_digest="b" * 64,
        static_check_status="failed",
        static_check_failures=["example failure"],
    )
    assert checkpoint.branch == branch
    developer.checkpoints.save(checkpoint)
    monkeypatch.setattr(
        developer.github,
        "worktree_for_branch",
        lambda candidate_branch: workspace if candidate_branch == branch else None,
    )

    result = developer.retry_candidate(
        branch,
        reason="Framework output contract failed; retry the same candidate.",
    )
    restored = developer.checkpoints.load()
    rebind_event = developer.audit.latest("candidate_policy_retry_checkpoint_rebind")

    assert result.resume_mode == "resume_existing_worktree"
    assert result.branch == branch
    assert rebind_event is not None, (
        f"wrapper returned without checkpoint-rebind evidence; checkpoint={checkpoint.branch!r}, "
        f"result={result.branch!r}, mode={result.resume_mode!r}"
    )
    assert rebind_event["status"] == "preserved", rebind_event
    assert restored is not None
    assert restored.cycle_id == result.retry_cycle_id
    assert restored.branch == branch
    assert restored.milestone == "local_static_repair"
    assert restored.git_state_digest == checkpoint.git_state_digest


def test_structured_static_repair_failure_becomes_durable_policy_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    branch = "localpilot/candidate-repair-contract-20260915-000000"
    cycle_id = developer.memory.start_cycle(
        task_id="repair-contract",
        branch=branch,
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )

    def failed_repair(_self, *args, **kwargs):
        return StaticRepairResult(
            "static_checks=failed",
            False,
            "Structured static repair failed: ValueError: Model response did not contain a valid JSON object.",
            3,
        )

    monkeypatch.setattr(BaseSelfDeveloper, "_repair_static_failures", failed_repair)
    developer._repair_static_failures(cycle_id=cycle_id, branch=branch)
    durable = developer.memory.candidate_for_cycle(cycle_id)

    assert durable is not None
    assert "Framework policy/orchestration failure" in durable.write_integrity_failure
    assert "structured static-repair contract failed" in durable.write_integrity_failure


def test_clean_terminal_candidate_worktree_is_removed(tmp_path: Path, monkeypatch) -> None:
    developer = _developer(tmp_path)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    branch = "localpilot/candidate-clean-failure-20260915-000000"
    removed: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        BaseSelfDeveloper,
        "_continue_candidate",
        lambda _self, **_kwargs: EvolutionResult(
            "failed", branch, workspace, "pre-write failure", False
        ),
    )
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda _branch: workspace)
    monkeypatch.setattr(developer.github, "candidate_changed_paths", lambda _workspace: [])
    monkeypatch.setattr(developer.github, "branch_has_candidate_commit", lambda _workspace: False)

    def remove(candidate_branch, *, expected_workspace):
        removed.append((candidate_branch, Path(expected_workspace)))
        return SimpleNamespace(ok=True, stdout="removed", stderr="")

    monkeypatch.setattr(developer.github, "remove_candidate_worktree", remove)
    result = developer._continue_candidate(
        branch=branch,
        workspace=workspace,
        is_worktree=True,
    )

    assert removed == [(branch, workspace.resolve())]
    assert "worktree removed" in result.summary
