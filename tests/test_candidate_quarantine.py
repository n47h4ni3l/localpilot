from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from localpilot.checkpoint import EvolutionCheckpoint
from localpilot.config import Config
from localpilot.evolution_reliability import CandidateQuarantinedError, SelfDeveloper
from localpilot.implementation_backend import ImplementationPreflight, ImplementationResult, ImplementationStatus
from localpilot.selfdev import CandidateTools
from localpilot.selfdev_results import CyclePaused, EvolutionResult


def fixture(tmp_path: Path):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    branch = "localpilot/candidate-unsafe"
    task = {"id": "unsafe", "title": "Repair retained file", "acceptance": []}
    cycle = developer.memory.start_cycle(
        task_id=task["id"], branch=branch, everyday_model="test", developer_model="test",
        workspace=tmp_path, is_worktree=True,
    )
    developer.memory.finish_cycle(cycle, status="candidate_pending_validation", summary="Original failed evidence",
                                  reusable_lesson="Preserve this", checks_passed=False, pushed=True, validation_state="failed")
    return developer, cycle, branch, task


def test_quarantine_preserves_original_record_and_cannot_be_reopened_by_reconciliation(tmp_path):
    developer, cycle, branch, task = fixture(tmp_path)
    before = developer.memory.candidate_for_cycle(cycle)
    developer.memory.quarantine_candidate(cycle, reason="Confirmed confinement violation")
    developer.memory.update_candidate_review(cycle, validation_state="failed", merged=False, pull_request_url="old-pr")
    developer.memory.finish_cycle(cycle, status="candidate_needs_work", summary="retry", reusable_lesson="retry",
                                  checks_passed=False, pushed=False)
    after = developer.memory.candidate_for_cycle(cycle)
    assert after.status == "quarantined" and after.pushed == before.pushed
    assert after.validation_state == before.validation_state and after.summary == before.summary
    assert not developer.memory.pending_candidates() and not developer.memory.failed_candidates()
    assert not developer.memory.local_candidates() and not developer.memory.has_outstanding_candidate()
    assert task["id"] in developer.memory.rejected_task_ids()  # Hold the original task out of rediscovery.
    with sqlite3.connect(developer.memory.path) as connection:
        saved = connection.execute("SELECT reason, prior_record_json FROM candidate_quarantines WHERE cycle_id=?", (cycle,)).fetchone()
    assert saved[0] == "Confirmed confinement violation"
    assert json.loads(saved[1])["summary"] == "Original failed evidence"
    developer.memory.quarantine_candidate(cycle, reason="Second observation")
    assert developer.memory.candidate_for_cycle(cycle).status == "quarantined"


def test_confirmed_backend_confinement_failure_is_retained_once_and_discovery_can_continue(tmp_path, monkeypatch):
    developer, cycle, branch, task = fixture(tmp_path)
    source = tmp_path / "broken.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    tools = CandidateTools(tmp_path)
    checkpoint = EvolutionCheckpoint.create(cycle_id=cycle, task=task, branch=branch, workspace=tmp_path, milestone="ci_repair")
    developer.checkpoints.save(checkpoint)
    original = developer.checkpoints.path.read_bytes()
    calls = []

    class Backend:
        def preflight(self):
            return ImplementationPreflight(True, "claude_code", "test", "test-double", context_tokens=65536)

        def run(self, request, *, repair_pass=0):
            calls.append(request)
            return ImplementationResult(ImplementationStatus.CONFINEMENT_VIOLATION, "claude_code", "test",
                                        "Candidate exceeds the file limit")

    monkeypatch.setattr(developer, "_admit_claude_unit", lambda **kwargs: None)
    monkeypatch.setattr(developer, "_claude_code_backend", lambda **kwargs: Backend())
    with pytest.raises(CandidateQuarantinedError):
        developer._run_claude_code_implementation(
            chat=None, developer_model="test", task=task, branch=branch, workspace=tmp_path,
            tools=tools, cycle_id=cycle, research="", grounding_plan={"referenced_paths": ["broken.py"]},
            grounding_evidence=[], evolution_context="", lessons=[], force=False,
        )
    result = developer._finalize_candidate_quarantine(EvolutionResult("failed", branch, tmp_path, "Confinement failure", False))
    assert result.status == "quarantined" and not result.tests_passed
    assert source.read_text() == "VALUE = 1\n" and not developer.checkpoints.path.exists()
    archives = list(developer.data_dir.glob("*.quarantined.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == original
    monkeypatch.setattr(developer.github, "sync_trusted_main", lambda: SimpleNamespace(ok=True, updated=False, summary="Current"))
    monkeypatch.setattr(developer, "_reconcile_candidates", lambda: None)
    monkeypatch.setattr(developer.governor, "sample", lambda: SimpleNamespace(allows_selfdev=lambda **kwargs: True))
    monkeypatch.setattr(developer.governor, "apply_process_priority", lambda **kwargs: None)
    monkeypatch.setattr(developer, "_load_next_task", lambda: None)
    monkeypatch.setattr(developer, "_select_developer_model", lambda: SimpleNamespace(model="test"))

    def discovery(**kwargs):
        raise CyclePaused("Reached fresh capability discovery")

    monkeypatch.setattr(developer, "_discover_capability_task", discovery)
    assert "fresh capability discovery" in developer._run_once().summary
    assert len(calls) == 1


def test_quarantine_does_not_touch_an_unrelated_checkpoint(tmp_path):
    developer, cycle, branch, task = fixture(tmp_path)
    other = EvolutionCheckpoint.create(cycle_id=cycle + 1, task=task, branch="another", workspace=tmp_path, milestone="research")
    developer.checkpoints.save(other)
    original = developer.checkpoints.path.read_bytes()
    developer.memory.quarantine_candidate(cycle, reason="Unsafe workspace")
    developer._finalize_candidate_quarantine(EvolutionResult("failed", branch, tmp_path, "Unsafe", False))
    assert developer.checkpoints.path.read_bytes() == original


def test_quarantine_refuses_missing_or_merged_cycle(tmp_path):
    developer, cycle, branch, task = fixture(tmp_path)
    with pytest.raises(ValueError):
        developer.memory.quarantine_candidate(cycle + 100, reason="Missing")
    developer.memory.update_candidate_review(cycle, validation_state="passed", merged=True, pull_request_url="merged-pr")
    with pytest.raises(ValueError):
        developer.memory.quarantine_candidate(cycle, reason="Already merged")


def test_reconciliation_cannot_overwrite_a_quarantine_committed_after_its_read(tmp_path, monkeypatch):
    developer, cycle, branch, task = fixture(tmp_path)
    original_connect = developer.memory._connect
    raced = []

    class Connection:
        def __init__(self):
            self.connection = original_connect()

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, sql, parameters=()):
            if "SET validation_state = ?" in sql and not raced:
                raced.append(True)
                developer.memory.quarantine_candidate(cycle, reason="Confinement observed concurrently")
            return self.connection.execute(sql, parameters)

    monkeypatch.setattr(developer.memory, "_connect", Connection)
    developer.memory.update_candidate_review(cycle, validation_state="failed", merged=False, pull_request_url="old-pr")
    assert raced and developer.memory.candidate_for_cycle(cycle).status == "quarantined"
