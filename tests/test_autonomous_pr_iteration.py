from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from localpilot.autonomous_pr_iteration import (
    AutonomousPRLedger, MAX_AUTONOMOUS_STEPS_PER_PR,
)
from localpilot.config import Config
from localpilot.github_integration import CandidateLifecycle, CommandResult
from localpilot.selfdev import EvolutionResult, SelfDeveloper


BRANCH = "localpilot/candidate-incremental-factuality-20261015-123abc"
HEAD = "a" * 40


def test_autonomous_head_receipts_resume_only_after_real_new_commit(tmp_path):
    ledger = AutonomousPRLedger(tmp_path / "private" / "iterations.json")
    assert ledger.can_start(BRANCH, HEAD)
    assert ledger.claim(BRANCH, HEAD)
    assert not ledger.can_start(BRANCH, HEAD)
    assert not AutonomousPRLedger(ledger.path).claim(BRANCH, HEAD)
    for number in range(1, MAX_AUTONOMOUS_STEPS_PER_PR):
        new_head = f"{number:040x}"
        assert ledger.claim(BRANCH, new_head)
        assert not ledger.can_start(BRANCH, new_head)
    assert not ledger.can_start(BRANCH, "f" * 40)
    assert not ledger.claim(BRANCH, "f" * 40)
    assert ledger.path.read_text(encoding="utf-8").count('"head"') == 1


def test_incremental_brief_survives_restart_and_closes_after_push(tmp_path):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    task = {
        "id": "factuality-progress",
        "title": "Factuality measurement",
        "question": "Can the model measure factual error rates?",
        "acceptance": ["Measure the complete factuality baseline."],
        "evaluation": {
            "metric": "hallucination rate", "baseline": "unknown",
            "measurement_method": "Run a reproducible reference benchmark",
        },
        "source": "capability_discovery",
    }
    developer.memory.record_experiment(task)
    cycle = developer.memory.start_cycle(
        task_id=task["id"], branch=BRANCH, everyday_model="nestra",
        developer_model="developer", workspace=tmp_path / "candidate", is_worktree=True,
    )
    receipts = AutonomousPRLedger(
        developer.data_dir / "autonomous-pr-iterations.json"
    )
    acceptance = ["Build one meaningful scoring fixture this time."]
    assert receipts.claim(
        BRANCH, HEAD, cycle_id=cycle,
        task_id=task["id"], acceptance=acceptance,
    )
    assert receipts.active_for_task(task["id"])["acceptance"] == acceptance
    # The actual checkpoint loader must reconstruct the exact per-step task,
    # rather than reverting to the larger original acceptance criteria.
    restored = developer._load_task_by_id(task["id"])
    assert restored["acceptance"] == acceptance
    assert restored["incremental_pr_revision"] is True
    receipts.finish(BRANCH, cycle, "completed")
    assert receipts.active_for_task(task["id"]) is None
    restored = developer._load_task_by_id(task["id"])
    assert restored["acceptance"] != acceptance
    assert not restored.get("incremental_pr_revision")


def test_autonomous_receipts_fail_closed_on_corruption(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version":1,"branches":{"unsafe":{"attempts":true,"head":"bad"}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        AutonomousPRLedger(path).can_start(BRANCH, HEAD)


def _candidate(tmp_path, monkeypatch):
    cfg = Config()
    cfg.agent.data_dir = "data"
    dev = SelfDeveloper(cfg, tmp_path)
    workspace = tmp_path / "existing-candidate"
    workspace.mkdir()
    task = {
        "id": "measure-factuality",
        "title": "Build factuality evaluation",
        "source": "capability_discovery",
        "acceptance": ["Complete the full 95% interval factuality evaluation."],
        "question": "How often does Nestra hallucinate?",
        "evaluation": {
            "metric": "hallucination rate",
            "baseline": "unknown",
            "measurement_method": "Run the factuality benchmark",
        },
    }
    cycle = dev.memory.start_cycle(
        task_id=task["id"], branch=BRANCH,
        everyday_model="nestra", developer_model="developer",
        workspace=workspace, is_worktree=True,
    )
    dev.memory.finish_cycle(
        cycle, status="candidate_pending_validation",
        summary="Initial draft added only typing imports; benchmark still missing.",
        reusable_lesson="Add a meaningful test next.",
        checks_passed=True, pushed=True, validation_state="passed",
    )
    monkeypatch.setattr(dev, "_load_task_by_id", lambda tid: dict(task))
    monkeypatch.setattr(
        dev.github, "candidate_lifecycle",
        lambda branch: CandidateLifecycle(
            "passed", False, "https://github.com/owner/localpilot/pull/220", "open", True,
        ),
    )
    monkeypatch.setattr(dev.github, "worktree_for_branch", lambda branch: workspace)
    monkeypatch.setattr(dev.github, "reviewer_modified_test_paths", lambda workspace: set())
    monkeypatch.setattr(dev, "_select_developer_model", lambda: SimpleNamespace(model="nestra", reason=""))
    monkeypatch.setattr(dev, "_candidate_tools", lambda *args, **kwargs: object())

    def git(args, **kwargs):
        if args[:3] == ["gh", "pr", "list"]:
            return CommandResult(True, json.dumps([
                {"number": 220, "headRefName": BRANCH, "baseRefName": "main", "isCrossRepository": False},
            ]), "", 0)
        if args[:3] == ["git", "status", "--porcelain"]:
            return CommandResult(True, "", "", 0)
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return CommandResult(True, HEAD, "", 0)
        if args[:3] == ["git", "ls-remote", "--heads"]:
            return CommandResult(True, HEAD + " refs/heads/" + BRANCH, "", 0)
        raise AssertionError(args)

    monkeypatch.setattr(dev.github, "_run", git)
    return dev, workspace, task


def test_autonomous_iteration_reuses_existing_pr_and_reduces_per_round_scope(tmp_path, monkeypatch):
    dev, workspace, original = _candidate(tmp_path, monkeypatch)
    observed = []

    def continue_candidate(**kwargs):
        observed.append(kwargs)
        assert kwargs["branch"] == BRANCH
        assert kwargs["workspace"] == workspace
        task = kwargs["task"]
        assert task["incremental_pr_revision"] is True
        assert task["question"] == original["question"]
        assert task["evaluation"] == original["evaluation"]
        assert "entire objective" in task["acceptance"][0]
        assert "Unused imports" in task["acceptance"][1]
        assert "95% interval" in task["acceptance"][3]
        assert "typing imports" in task["acceptance"][4]
        dev.memory.finish_cycle(
            kwargs["cycle_id"], status="candidate_pending_validation",
            summary="Added first measurable fixture; more steps remain.",
            reusable_lesson="Next: add scorer.", checks_passed=True, pushed=True,
            validation_state="awaiting_pr",
        )
        return EvolutionResult(
            "candidate_pending_validation", BRANCH, workspace, "Increment committed", True,
        )

    monkeypatch.setattr(dev, "_continue_candidate", continue_candidate)
    result = dev._iterate_open_candidate(force=False)
    assert result.status == "candidate_pending_validation"
    assert len(observed) == 1
    assert dev.memory.candidate_for_branch(BRANCH).cycle_id == observed[0]["cycle_id"]
    # Never repeat an unchanged head even if its previous CI check becomes green.
    assert dev._iterate_open_candidate(force=False) is None
    assert len(observed) == 1


def test_autonomous_iteration_fails_closed_on_dirty_or_cross_repo_candidate(tmp_path, monkeypatch):
    dev, workspace, task = _candidate(tmp_path, monkeypatch)
    attempted = []
    monkeypatch.setattr(dev, "_continue_candidate", lambda **kwargs: attempted.append(kwargs))

    def wrong_pr(args, **kwargs):
        if args[:3] == ["gh", "pr", "list"]:
            return CommandResult(True, json.dumps([
                {"number": 220, "headRefName": BRANCH, "baseRefName": "main", "isCrossRepository": True},
            ]), "", 0)
        raise AssertionError("Must refuse before touching Git")

    monkeypatch.setattr(dev.github, "_run", wrong_pr)
    assert dev._iterate_open_candidate(force=False).status == "deferred"
    assert not attempted

    def dirty(args, **kwargs):
        if args[:3] == ["gh", "pr", "list"]:
            return CommandResult(True, json.dumps([
                {"number": 220, "headRefName": BRANCH, "baseRefName": "main", "isCrossRepository": False},
            ]), "", 0)
        if args[:3] == ["git", "status", "--porcelain"]:
            return CommandResult(True, " M localpilot/behavior_eval.py", "", 0)
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return CommandResult(True, HEAD, "", 0)
        if args[:3] == ["git", "ls-remote", "--heads"]:
            return CommandResult(True, HEAD + " refs/heads/" + BRANCH, "", 0)
        raise AssertionError(args)

    monkeypatch.setattr(dev.github, "_run", dirty)
    assert dev._iterate_open_candidate(force=False).status == "deferred"
    assert not attempted


def test_failed_pr_does_not_auto_iterate_and_owner_revision_still_preempts(tmp_path, monkeypatch):
    dev, workspace, task = _candidate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        dev.github, "candidate_lifecycle",
        lambda branch: CandidateLifecycle("failed", False, "url", "open", True),
    )
    assert dev._iterate_open_candidate(force=False) is None
    assert not (dev.data_dir / "autonomous-pr-iterations.json").exists()
