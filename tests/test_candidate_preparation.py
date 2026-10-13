import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from localpilot.config import Config, GitHubConfig
from localpilot.evolution import capability_task_id, parse_capability_proposals
from localpilot.github_integration import GitHubIntegration, MainSyncResult
from localpilot.learning import LearningMemory
from localpilot.resource import ResourceState
from localpilot.selfdev import DeveloperModelSelection, EvolutionResult, SelfDeveloper


def task(task_id="capability-preparation-test"):
    return {
        "id": task_id, "title": "Measure verified retrieval", "source": "capability_discovery",
        "evolution_class": "improve_cognition", "capability_target": "verified retrieval",
        "mission_alignment": "Improve reusable verified reasoning while preserving human control.",
        "current_frontier": "Stored evidence is not retrieved consistently.",
        "why_high_leverage": "Future tasks rely on verified memory retrieval.",
        "capability_unlocked": "More reliable reuse of verified evidence.",
        "next_frontier": "Measure transfer to longer planning tasks.",
        "question": "Can a fixed benchmark reduce missed retrieval evidence?",
        "observed_limitation": "Relevant verified lessons are missed.",
        "evidence": ["localpilot/learning.py owns retrieval."],
        "alternatives": ["Tune existing retrieval.", "Evaluate a reranker."],
        "hypothesis": "Fixed retrieval tests will reduce missed evidence.",
        "expected_complexity": "low",
        "evaluation": {"metric": "relevant recall", "baseline": "Measure current recall.",
                       "success_criterion": "Improve recall without reducing precision.",
                       "measurement_method": "Identical fixed before/after queries in CI."},
    }


def developer_at(root):
    config = Config()
    config.agent.data_dir = "data"
    return SelfDeveloper(config, root)


@pytest.mark.parametrize("ledger_status", [None, "active", "selected", "superseded", "preparation_failed"])
def test_unstarted_proposal_resumes_after_reopen_without_model(tmp_path, ledger_status):
    developer = developer_at(tmp_path)
    original = task()
    developer.memory.record_experiment(original)
    if ledger_status:
        developer.opportunities.enqueue(original, score=10)
        developer.opportunities.update(original["id"], ledger_status)
    before = developer.memory.experiment_for_task(original["id"])

    reopened = developer_at(tmp_path)
    recovered = reopened._discover_capability_task(developer_model="must-not-load", force=False)

    assert recovered["id"] == original["id"]
    assert recovered["evaluation"] == original["evaluation"]
    assert recovered["hypothesis"] == original["hypothesis"]
    assert reopened.memory.experiment_for_task(original["id"]) == before
    assert reopened.audit.latest("capability_preparation_resumed")["prior_attempts_preserved"]


@pytest.mark.parametrize("status", ["validated", "rejected", "quarantined", "candidate_active"])
def test_terminal_or_started_experiment_cannot_recover(tmp_path, status):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_experiment(task())
    memory.update_experiment_outcome(task()["id"], status=status)
    assert memory.preparation_recovery_tasks() == []


def test_unattached_cycle_prevents_recovery(tmp_path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_experiment(task())
    memory.start_cycle(task_id=task()["id"], branch="localpilot/candidate-original",
                       everyday_model="daily", developer_model="developer")
    assert memory.experiment_for_task(task()["id"]).cycle_id is None
    assert memory.preparation_recovery_tasks() == []


def test_retries_bounded_and_all_preparation_evidence_retained(tmp_path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_experiment(task())
    for attempt in range(3):
        assert memory.preparation_recovery_tasks()
        pid = memory.start_candidate_preparation(task()["id"], f"localpilot/candidate-{attempt}",
                                                tmp_path / str(attempt))
        if attempt < 2:
            memory.finish_candidate_preparation(pid, error=f"checkout failure {attempt}")
        # Third attempt simulates abrupt process exit, preserving 'preparing'.
    assert LearningMemory(memory.path).preparation_recovery_tasks() == []
    with sqlite3.connect(memory.path) as connection:
        rows = connection.execute("SELECT status, error FROM candidate_preparations ORDER BY id").fetchall()
    assert rows == [("failed", "checkout failure 0"), ("failed", "checkout failure 1"), ("preparing", "")]
    assert memory.experiment_for_task(task()["id"]).status == "proposed"


def test_completed_duplicate_is_still_rejected(tmp_path, monkeypatch):
    developer = developer_at(tmp_path)
    proposal = parse_capability_proposals(json.dumps(task()))[0]
    original = proposal.task(capability_task_id(proposal))
    developer.memory.record_experiment(original)
    developer.memory.update_experiment_outcome(original["id"], status="validated")
    monkeypatch.setattr(developer.github, "tracked_project_paths", lambda: set())
    monkeypatch.setattr(developer, "_tool_stage", lambda **kwargs: json.dumps({"proposals": [task()]}))
    with pytest.raises(ValueError, match="previously attempted hypotheses"):
        developer._discover_capability_task(developer_model="fake", force=True)


def test_abandoned_proposal_does_not_resume(tmp_path, monkeypatch):
    developer = developer_at(tmp_path)
    original = task()
    developer.memory.record_experiment(original)
    developer.opportunities.enqueue(original, score=10)
    developer.opportunities.update(original["id"], "abandoned")
    monkeypatch.setattr(developer.github, "tracked_project_paths", lambda: set())
    monkeypatch.setattr(developer, "_tool_stage", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("new discovery")))
    with pytest.raises(RuntimeError, match="new discovery"):
        developer._discover_capability_task(developer_model="fake", force=True)
    assert developer.audit.latest("capability_preparation_resumed") is None


@pytest.mark.parametrize("source", ["queue", "model"])
def test_selection_interruption_keeps_proposal_recoverable(tmp_path, monkeypatch, source):
    developer = developer_at(tmp_path)
    if source == "queue":
        developer.opportunities.enqueue(task(), score=10)
    monkeypatch.setattr(developer.github, "tracked_project_paths", lambda: set())
    monkeypatch.setattr(developer, "_tool_stage", lambda **kwargs: json.dumps({"proposals": [task()]}))
    monkeypatch.setattr(developer.opportunities, "update", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        developer._discover_capability_task(developer_model="fake", force=True)
    reopened = developer_at(tmp_path)
    expected = reopened.memory.latest_experiment()
    assert expected is not None and expected.cycle_id is None
    assert reopened._discover_capability_task(developer_model="must-not-load", force=False)["id"] == expected.task_id


def test_workspace_failure_then_success_preserves_attempt_and_uses_same_experiment(tmp_path, monkeypatch):
    developer = developer_at(tmp_path)
    original = task()
    developer.memory.record_experiment(original)
    developer.opportunities.enqueue(original, score=10)
    monkeypatch.setattr(developer.github, "sync_trusted_main", lambda: MainSyncResult(True, False, "current"))
    monkeypatch.setattr(developer.governor, "sample", lambda: ResourceState(1000, 5, 20, True, "idle", True, True))
    monkeypatch.setattr(developer.governor, "apply_process_priority", lambda idle: None)
    monkeypatch.setattr(developer, "_reconcile_candidates", lambda: None)
    monkeypatch.setattr(developer, "_repair_local_candidate", lambda **kwargs: None)
    monkeypatch.setattr(developer, "_repair_failed_candidate", lambda **kwargs: None)
    monkeypatch.setattr(developer, "_load_next_task", lambda: None)
    monkeypatch.setattr(developer, "_select_developer_model", lambda: DeveloperModelSelection("fake", 1, 1.0, "fits"))
    monkeypatch.setattr(developer, "_review_task_repository", lambda *args, **kwargs: {"status": "source_anchored"})
    calls = []

    def workspace(branch):
        calls.append(branch)
        if len(calls) == 1:
            raise RuntimeError("Filename too long")
        path = developer._candidate_destination(branch)
        path.mkdir(parents=True)
        return path, True

    monkeypatch.setattr(developer, "_candidate_workspace", workspace)
    monkeypatch.setattr(developer, "_continue_candidate", lambda **kwargs: EvolutionResult(
        "paused", kwargs["branch"], kwargs["workspace"], "test stopped before inference"))
    first = developer.run_once(force=True)
    assert first.status == "preparation_failed"
    assert developer.memory.latest_experiment().cycle_id is None
    second = developer.run_once(force=True)
    assert second.status == "paused"
    assert calls[0] != calls[1]
    assert developer.memory.latest_experiment().cycle_id is not None
    assert developer.memory.preparation_recovery_tasks() == []
    with sqlite3.connect(developer.memory.path) as connection:
        rows = connection.execute("SELECT branch,status,error FROM candidate_preparations ORDER BY id").fetchall()
    assert rows[0] == (calls[0], "failed", "RuntimeError: Filename too long")
    assert rows[1] == (calls[1], "prepared", "")


def test_short_destination_never_overwrites_existing_workspace(tmp_path, monkeypatch):
    developer = developer_at(tmp_path)
    branch = "localpilot/candidate-" + "x" * 120
    destination = developer._candidate_destination(branch)
    assert len(destination.name) == 26
    assert developer._candidate_destination(branch + "-retry") != destination
    destination.mkdir(parents=True)
    evidence = destination / "retained.txt"
    evidence.write_text("preserve this failure")
    monkeypatch.setattr(developer.github, "create_candidate_worktree", lambda *args: pytest.fail("must not overwrite"))
    with pytest.raises(RuntimeError, match="preserved for inspection"):
        developer._candidate_workspace(branch)
    assert evidence.read_text() == "preserve this failure"


@pytest.mark.skipif(os.name != "nt" or not shutil.which("git"), reason="Git for Windows regression")
def test_real_windows_long_checkout_without_changing_git_settings(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    integration = GitHubIntegration(root, GitHubConfig())
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
    assert git("init").returncode == 0
    assert git("config", "core.longpaths", "false").returncode == 0
    assert git("config", "core.autocrlf", "false").returncode == 0
    short_leaf = "c-" + "a" * 24
    card_base = "training/licenses/p12_fresh/cards/HuggingFaceTB--.md"
    padding = max(65, 280 - len(str(tmp_path / short_leaf / card_base)))
    tracked = Path("training/licenses/p12_fresh/cards") / ("HuggingFaceTB--" + "x" * padding + ".md")
    source = root / tracked
    source.parent.mkdir(parents=True)
    source.write_bytes(b"Provenance must remain intact.\n")
    assert integration._run(["git", "add", "."]).ok
    assert integration._run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                             "commit", "-m", "fixture"]).ok
    destination = tmp_path / ("candidate-" + "x" * 30)
    assert len(str(destination / tracked)) > 260
    failed = git("worktree", "add", "-b", "localpilot/candidate-failure", str(destination), "HEAD")
    assert failed.returncode != 0
    assert "Filename too long" in failed.stderr
    failure_log = tmp_path / "failed-checkout.txt"
    failure_log.write_text(failed.stdout + failed.stderr, encoding="utf-8")
    # Git's own worktree metadata also has a path limit. Test the complete fix:
    # a short leaf for metadata and command-scoped longpaths for tracked files.
    recovered = tmp_path / short_leaf
    assert len(str(recovered / tracked)) > 260
    result = integration.create_candidate_worktree("localpilot/candidate-retry", recovered)
    assert result.ok, result.stderr
    assert hashlib.sha256((recovered / tracked).read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    assert integration._run(["git", "status", "--porcelain"], cwd=recovered).stdout == ""
    assert git("config", "--get", "core.longpaths").stdout.strip() == "false"
    # Git can remove its own failed checkout; our repair does not erase the
    # retained branch or the captured failure evidence.
    assert git("show-ref", "--verify", "refs/heads/localpilot/candidate-failure").returncode == 0
    assert "Filename too long" in failure_log.read_text(encoding="utf-8")
