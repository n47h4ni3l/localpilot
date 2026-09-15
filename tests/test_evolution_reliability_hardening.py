from __future__ import annotations

import json
from pathlib import Path

import pytest

from localpilot.checkpoint import CheckpointStore, EvolutionCheckpoint, task_fingerprint
from localpilot.evolution_orchestrator import EvolutionRunAlreadyActive, EvolutionRunLease
from localpilot.selfdev_response_parsing import _json_object


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
