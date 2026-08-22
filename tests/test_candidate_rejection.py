import json
from pathlib import Path

import pytest
from rich.console import Console

from localpilot.checkpoint import EvolutionCheckpoint
from localpilot.cli import _show_status, build_parser
from localpilot.config import Config, GitHubConfig
from localpilot.github_integration import (
    CandidateLifecycle,
    CandidatePullRequest,
    CommandResult,
    GitHubIntegration,
)
from localpilot.selfdev import CandidateRejectionError, SelfDeveloper


BRANCH = "localpilot/candidate-structural-completeness"
PR_URL = "https://github.com/example/localpilot/pull/19"
REASON = (
    "Green CI was insufficient: the candidate referenced missing "
    "scripts/pre_ci_review.sh. Validate structural completeness."
)
TASK = {
    "id": "structural-completeness",
    "title": "Validate candidate structural completeness",
    "status": "todo",
    "evolution_class": "repair",
    "capability_target": "candidate evaluation quality",
    "question": "Can structural validation catch references to missing files?",
    "observed_limitation": "Green CI did not catch a missing referenced script.",
    "hypothesis": "Reference validation will catch missing candidate dependencies.",
    "evaluation": {
        "metric": "missing references detected",
        "baseline": "0 of 1",
        "success_criterion": "1 of 1",
        "measurement_method": "held-out structural fixture in GitHub CI",
    },
}


def _developer(tmp_path: Path) -> tuple[SelfDeveloper, int, Path]:
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    developer.memory.record_experiment(TASK)
    cycle_id = developer.memory.start_cycle(
        task_id=TASK["id"],
        branch=BRANCH,
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )
    developer.memory.attach_experiment_cycle(TASK["id"], cycle_id, BRANCH)
    developer.memory.update_experiment_outcome(
        TASK["id"],
        status="candidate_pending_validation",
        outcome="CI passed; human review pending.",
        before_evidence="Referenced command was not checked.",
        after_evidence="GitHub checks passed.",
        reusable_lesson="CI is a necessary validation boundary.",
    )
    developer.memory.finish_cycle(
        cycle_id,
        status="candidate_pending_validation",
        summary="Candidate pushed and CI passed.",
        reusable_lesson="Keep validation evidence attached to the candidate.",
        checks_passed=True,
        pushed=True,
    )
    developer.memory.update_candidate_review(
        cycle_id,
        validation_state="passed",
        merged=False,
        pull_request_url=PR_URL,
    )
    return developer, cycle_id, workspace


def _resolved() -> CandidatePullRequest:
    return CandidatePullRequest(19, BRANCH, PR_URL)


def test_reject_pr_maps_to_managed_cycle_and_retains_terminal_evidence(
    tmp_path: Path,
    monkeypatch,
):
    developer, cycle_id, workspace = _developer(tmp_path)
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=cycle_id,
        task=TASK,
        branch=BRANCH,
        workspace=workspace,
        milestone="delivery",
    )
    developer.checkpoints.save(checkpoint)
    monkeypatch.setattr(developer.github, "resolve_candidate_pull_request", lambda _pr: _resolved())

    cleanup_observations = []

    def cleanup(branch, *, expected_workspace=None):
        durable = developer.memory.candidate_for_branch(branch)
        cleanup_observations.append((durable.validation_state, expected_workspace))
        return CommandResult(True, "Removed clean local candidate worktree.", "", 0)

    monkeypatch.setattr(developer.github, "remove_candidate_worktree", cleanup)

    result = developer.reject_candidate(19, reason=REASON)
    rejected = developer.memory.candidate_for_branch(BRANCH)

    assert result.already_rejected is False
    assert result.checkpoint_cleared is True
    assert developer.checkpoints.exists() is False
    assert cleanup_observations == [("rejected_by_human", str(workspace.resolve()))]
    assert rejected.status == "rejected_by_human"
    assert rejected.validation_state == "rejected_by_human"
    assert rejected.rejection_prior_validation_state == "passed"
    assert rejected.rejection_pull_request_number == 19
    assert rejected.rejection_reason == REASON
    assert rejected.pull_request_url == PR_URL
    assert rejected.task_id == TASK["id"]
    assert rejected.branch == BRANCH
    assert rejected.merged is False
    assert developer.memory.has_outstanding_candidate() is False
    assert developer.memory.pending_candidates() == []
    assert developer.memory.pending_task_ids() == set()

    experiment = developer.memory.experiment_for_task(TASK["id"])
    assert experiment.status == "rejected_by_human"
    assert "CI passed" in experiment.outcome
    assert REASON in experiment.outcome
    assert REASON in experiment.reusable_lesson
    context = developer.memory.discovery_context()
    assert REASON in json.dumps(context)
    assert REASON in developer.memory.reusable_lessons()[0]

    audit = developer.audit.latest("candidate_rejection")
    assert audit["status"] == "rejected"
    assert audit["prior_validation_state"] == "passed"
    assert audit["github_history_retained"] is True


def test_unrelated_or_unowned_pr_rejection_fails_closed(tmp_path: Path, monkeypatch):
    developer, _, _ = _developer(tmp_path)
    monkeypatch.setattr(
        developer.github,
        "resolve_candidate_pull_request",
        lambda _pr: (_ for _ in ()).throw(
            ValueError("only LocalPilot-managed candidate branches can be rejected")
        ),
    )
    with pytest.raises(CandidateRejectionError, match="LocalPilot-managed"):
        developer.reject_candidate(20, reason="Not an autonomous candidate.")
    assert developer.memory.has_outstanding_candidate() is True

    monkeypatch.setattr(
        developer.github,
        "resolve_candidate_pull_request",
        lambda _pr: CandidatePullRequest(
            21,
            "localpilot/candidate-not-in-memory",
            "https://github.com/example/localpilot/pull/21",
        ),
    )
    with pytest.raises(CandidateRejectionError, match="no durable"):
        developer.reject_candidate(21, reason="Unknown candidate.")


def test_github_pr_resolution_rejects_unrelated_branch(tmp_path: Path, monkeypatch):
    github = GitHubIntegration(tmp_path, GitHubConfig())
    monkeypatch.setattr(github, "gh_available", lambda: True)
    monkeypatch.setattr(
        github,
        "_run",
        lambda *_args, **_kwargs: CommandResult(
            True,
            json.dumps(
                {
                    "number": 19,
                    "headRefName": "feature/manual-change",
                    "url": PR_URL,
                }
            ),
            "",
            0,
        ),
    )
    with pytest.raises(ValueError, match="unrelated branch"):
        github.resolve_candidate_pull_request(19)


def test_repeated_rejection_is_idempotent_and_preserves_original_reason(
    tmp_path: Path,
    monkeypatch,
):
    developer, _, _ = _developer(tmp_path)
    monkeypatch.setattr(developer.github, "resolve_candidate_pull_request", lambda _pr: _resolved())
    monkeypatch.setattr(
        developer.github,
        "remove_candidate_worktree",
        lambda *_args, **_kwargs: CommandResult(True, "No registered candidate worktree.", "", 0),
    )

    first = developer.reject_candidate(19, reason=REASON)
    second = developer.reject_candidate(19, reason="A different later reason.")

    assert first.already_rejected is False
    assert second.already_rejected is True
    assert second.reason == REASON
    assert developer.memory.latest_rejected_candidate().rejection_reason == REASON
    assert developer.audit.latest("candidate_rejection")["status"] == "already_rejected"


def test_reconciliation_cannot_reclassify_explicit_rejection(tmp_path: Path, monkeypatch):
    developer, cycle_id, _ = _developer(tmp_path)
    monkeypatch.setattr(developer.github, "resolve_candidate_pull_request", lambda _pr: _resolved())
    monkeypatch.setattr(
        developer.github,
        "remove_candidate_worktree",
        lambda *_args, **_kwargs: CommandResult(True, "No registered candidate worktree.", "", 0),
    )
    developer.reject_candidate(19, reason=REASON)

    developer.memory.update_candidate_review(
        cycle_id,
        validation_state="pending",
        merged=False,
        pull_request_url=PR_URL,
    )
    developer.memory.update_experiment_review(
        TASK["id"],
        validation_state="pending",
        merged=False,
    )
    monkeypatch.setattr(
        developer.github,
        "candidate_lifecycle",
        lambda _branch: pytest.fail("terminal rejection must not be reconciled"),
    )
    developer._reconcile_candidates()

    assert developer.memory.candidate_for_branch(BRANCH).validation_state == "rejected_by_human"
    assert developer.memory.experiment_for_task(TASK["id"]).status == "rejected_by_human"


def test_closed_pr_without_explicit_rejection_remains_outstanding(tmp_path: Path):
    developer, cycle_id, _ = _developer(tmp_path)
    # A closed, unmerged PR can still report green checks. Reconciliation has
    # no explicit rejection evidence and must therefore leave the gate closed.
    lifecycle = CandidateLifecycle("passed", False, PR_URL)
    developer.memory.update_candidate_review(
        cycle_id,
        validation_state=lifecycle.validation_state,
        merged=lifecycle.merged,
        pull_request_url=lifecycle.pull_request_url,
    )
    assert developer.memory.candidate_for_branch(BRANCH).status == "candidate_pending_validation"
    assert developer.memory.has_outstanding_candidate() is True


def test_rejection_does_not_clear_an_unrelated_checkpoint(tmp_path: Path, monkeypatch):
    developer, cycle_id, workspace = _developer(tmp_path)
    developer.checkpoints.save(
        EvolutionCheckpoint.create(
            cycle_id=cycle_id + 100,
            task={**TASK, "id": "different-task"},
            branch="localpilot/candidate-different-task",
            workspace=workspace,
            milestone="research",
        )
    )
    monkeypatch.setattr(developer.github, "resolve_candidate_pull_request", lambda _pr: _resolved())
    monkeypatch.setattr(
        developer.github,
        "remove_candidate_worktree",
        lambda *_args, **_kwargs: CommandResult(True, "No registered candidate worktree.", "", 0),
    )

    result = developer.reject_candidate(19, reason=REASON)

    assert result.checkpoint_cleared is False
    assert developer.checkpoints.exists() is True
    assert developer.checkpoints.load().task_id == "different-task"


def test_worktree_cleanup_skips_dirty_or_mismatched_local_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    github = GitHubIntegration(root, GitHubConfig())
    monkeypatch.setattr(github, "worktree_for_branch", lambda _branch: workspace.resolve())
    calls = []

    def run(args, cwd=None, timeout=30):
        calls.append((args, cwd))
        return CommandResult(True, " M local-only.txt", "", 0)

    monkeypatch.setattr(github, "_run", run)
    dirty = github.remove_candidate_worktree(BRANCH, expected_workspace=workspace)
    mismatch = github.remove_candidate_worktree(
        BRANCH,
        expected_workspace=tmp_path / "different",
    )

    assert dirty.ok is False
    assert "local changes" in dirty.stderr
    assert mismatch.ok is False
    assert "durable candidate memory" in mismatch.stderr
    assert calls == [
        (["git", "status", "--porcelain", "--untracked-files=all"], workspace.resolve())
    ]


def test_clean_worktree_cleanup_keeps_branch_and_remote_history(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    github = GitHubIntegration(root, GitHubConfig())
    monkeypatch.setattr(github, "worktree_for_branch", lambda _branch: workspace.resolve())
    calls = []

    def run(args, cwd=None, timeout=30):
        calls.append((args, cwd))
        return CommandResult(True, "", "", 0)

    monkeypatch.setattr(github, "_run", run)
    result = github.remove_candidate_worktree(BRANCH, expected_workspace=workspace)

    assert result.ok is True
    assert "branch and GitHub history were retained" in result.stdout
    assert calls == [
        (["git", "status", "--porcelain", "--untracked-files=all"], workspace.resolve()),
        (["git", "worktree", "remove", str(workspace.resolve())], None),
    ]
    assert not any("branch" in args or "push" in args for args, _cwd in calls)


def test_reject_cli_is_prompt_safe_and_status_shows_latest_reason(
    tmp_path: Path,
    monkeypatch,
):
    args = build_parser().parse_args(["reject", "19", "--reason", REASON])
    assert args.command == "reject"
    assert args.pull_request == 19
    assert args.reason == REASON

    developer, _, _ = _developer(tmp_path)
    monkeypatch.setattr(developer.github, "resolve_candidate_pull_request", lambda _pr: _resolved())
    monkeypatch.setattr(
        developer.github,
        "remove_candidate_worktree",
        lambda *_args, **_kwargs: CommandResult(True, "No registered candidate worktree.", "", 0),
    )
    developer.reject_candidate(19, reason=REASON)
    console = Console(record=True, width=140)
    _show_status(console, developer.config, developer.root)
    rendered = console.export_text()
    assert "Last rejected candidate" in rendered
    assert "PR #19" in rendered
    assert "prior CI: passed" in rendered
    assert "scripts/pre_ci_review.sh" in rendered
