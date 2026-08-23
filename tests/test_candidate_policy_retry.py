import json
from pathlib import Path

import pytest
from rich.console import Console

from localpilot.config import Config
from localpilot.cli import _show_status, build_parser
from localpilot.github_integration import CandidateLifecycle, CommandResult
from localpilot.learning import LearningMemory
from localpilot.selfdev import (
    CandidateRetryError,
    DeveloperModelSelection,
    EvolutionResult,
    SelfDeveloper,
)


BRANCH = "localpilot/candidate-model-adaptation-lab-20260823-003946"
TASK = "model-adaptation-lab"
RETRY_BRANCH_SUFFIX = "-policy-retry-"


def _policy_blocked(memory: LearningMemory, workspace: Path) -> int:
    cycle = memory.start_cycle(
        task_id=TASK,
        branch=BRANCH,
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )
    failure = (
        "Candidate delivery blocked because autonomous write attempts were rejected: "
        "training_datasets/: File type is not allowed; file-write limit reached"
    )
    memory.record_write_integrity_failure(cycle, failure)
    memory.finish_cycle(
        cycle,
        status="candidate_needs_work",
        summary=failure,
        reusable_lesson="The candidate architecture was useful.",
        checks_passed=False,
        pushed=False,
    )
    return cycle


def _pushed_policy_blocked(memory: LearningMemory, workspace: Path) -> int:
    cycle = _policy_blocked(memory, workspace)
    memory.finish_cycle(
        cycle,
        status="candidate_pending_validation",
        summary=(
            "Candidate cycle completed. static_checks=passed. "
            "prior_write_integrity_failure=Candidate delivery blocked because "
            "directory writes used a disallowed file type and the file-write limit was reached."
        ),
        reusable_lesson="The candidate architecture remains useful.",
        checks_passed=True,
        pushed=True,
        validation_state="passed",
    )
    return cycle


def test_policy_retry_preserves_history_and_is_idempotent(tmp_path: Path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    prior_id = _policy_blocked(memory, workspace)

    first = memory.authorize_policy_retry(BRANCH, reason="Framework policy blocked valid architecture.")
    repeated = memory.authorize_policy_retry(BRANCH, reason="A different reason must not overwrite history.")

    prior = memory.candidate_for_cycle(prior_id)
    retry = memory.candidate_for_cycle(first.retry_cycle_id)
    assert first.prior_cycle_id == prior_id
    assert repeated.retry_cycle_id == first.retry_cycle_id
    assert repeated.already_authorized is True
    assert prior is not None and prior.status == "policy_blocked"
    assert prior.failure_attribution == "framework_policy"
    assert prior.policy_failure_reason == "Framework policy blocked valid architecture."
    assert prior.retried_by_cycle_id == first.retry_cycle_id
    assert "candidate architecture was useful" in prior.reusable_lesson.lower()
    assert retry is not None and retry.task_id == TASK and retry.branch == BRANCH
    assert retry.human_authorized_retry is True
    assert retry.retry_of_cycle_id == prior_id
    assert retry.local_repair_attempts == 0
    assert retry.write_integrity_failure == ""
    assert len(memory.local_candidates()) == 1


def test_pushed_unmerged_policy_block_gets_fresh_linked_retry_and_clears_old_gate(
    tmp_path: Path,
):
    workspace = tmp_path / "original-pushed-candidate"
    workspace.mkdir()
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    prior_id = _pushed_policy_blocked(memory, workspace)

    first = memory.authorize_policy_retry(
        BRANCH,
        reason="Framework policy blocked valid architecture.",
        remote_branch_verified=True,
        remote_merged=False,
        pull_request_state="none",
    )
    repeated_by_branch = memory.authorize_policy_retry(
        BRANCH,
        reason="Do not create another retry.",
    )
    repeated_by_task = memory.authorize_policy_retry(
        TASK,
        reason="Still do not create another retry.",
    )

    prior = memory.candidate_for_cycle(prior_id)
    retry = memory.candidate_for_cycle(first.retry_cycle_id)
    assert first.prior_branch == BRANCH
    assert first.branch == f"{BRANCH}{RETRY_BRANCH_SUFFIX}{prior_id}"
    assert repeated_by_branch.retry_cycle_id == first.retry_cycle_id
    assert repeated_by_task.retry_cycle_id == first.retry_cycle_id
    assert repeated_by_branch.already_authorized is True
    assert repeated_by_task.already_authorized is True
    assert prior is not None and prior.branch == BRANCH and prior.pushed is True
    assert prior.status == "policy_blocked"
    assert prior.validation_state == "passed" and prior.merged is False
    assert prior.retried_by_cycle_id == first.retry_cycle_id
    assert retry is not None and retry.branch == first.branch
    assert retry.retry_of_cycle_id == prior_id and retry.pushed is False
    assert retry.workspace is None
    assert memory.pending_candidates() == []
    assert [candidate.cycle_id for candidate in memory.local_candidates()] == [retry.cycle_id]
    assert memory.has_outstanding_candidate() is True

    memory.update_candidate_review(
        prior_id,
        validation_state="pending",
        merged=False,
        pull_request_url="https://example.test/pull/17",
    )
    reconciled_prior = memory.candidate_for_cycle(prior_id)
    assert reconciled_prior is not None and reconciled_prior.status == "policy_blocked"
    assert reconciled_prior.validation_state == "passed"


def test_pushed_candidate_without_policy_evidence_is_refused(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    cycle = memory.start_cycle(
        task_id="ordinary-bug",
        branch="localpilot/candidate-ordinary-bug",
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )
    memory.finish_cycle(
        cycle,
        status="candidate_pending_validation",
        summary="Candidate pushed after a normal implementation failure.",
        reusable_lesson="fix the implementation",
        checks_passed=False,
        pushed=True,
        validation_state="failed",
    )

    with pytest.raises(ValueError, match="does not show a framework-policy"):
        memory.authorize_policy_retry(
            "ordinary-bug",
            reason="try again",
            remote_branch_verified=True,
            remote_merged=False,
            pull_request_state="open",
        )


def test_pushed_policy_candidate_requires_verified_remote_state(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    _pushed_policy_blocked(memory, workspace)

    with pytest.raises(ValueError, match="could not be verified"):
        memory.authorize_policy_retry(BRANCH, reason="Framework policy failure.")


def test_policy_blocked_failed_ci_cycle_is_not_selected_for_repair(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    prior_id = _pushed_policy_blocked(memory, workspace)
    memory.update_candidate_review(
        prior_id,
        validation_state="failed",
        merged=False,
        pull_request_url="https://example.test/pull/17",
    )
    retry = memory.authorize_policy_retry(
        BRANCH,
        reason="Framework policy failure.",
        remote_branch_verified=True,
        remote_merged=False,
        pull_request_state="open",
    )

    assert memory.failed_candidates() == []
    assert [candidate.cycle_id for candidate in memory.local_candidates()] == [
        retry.retry_cycle_id
    ]


def test_merged_or_human_rejected_pushed_candidate_is_refused(tmp_path: Path):
    merged_memory = LearningMemory(tmp_path / "merged.sqlite3")
    merged_workspace = tmp_path / "merged-candidate"
    merged_workspace.mkdir()
    merged_cycle = _pushed_policy_blocked(merged_memory, merged_workspace)
    merged_memory.update_candidate_review(
        merged_cycle,
        validation_state="passed",
        merged=True,
        pull_request_url="https://example.test/pull/17",
    )
    with pytest.raises(ValueError, match="already merged or promoted"):
        merged_memory.authorize_policy_retry(
            BRANCH,
            reason="retry merged work",
            remote_branch_verified=True,
            remote_merged=False,
            pull_request_state="closed",
        )

    rejected_memory = LearningMemory(tmp_path / "rejected.sqlite3")
    rejected_workspace = tmp_path / "rejected-candidate"
    rejected_workspace.mkdir()
    rejected_cycle = _pushed_policy_blocked(rejected_memory, rejected_workspace)
    rejected_memory.reject_candidate(
        rejected_cycle,
        pull_request_number=17,
        pull_request_url="https://example.test/pull/17",
        reason="Human rejected this attempt.",
    )
    with pytest.raises(ValueError, match="explicitly rejected by a human"):
        rejected_memory.authorize_policy_retry(
            BRANCH,
            reason="retry rejected work",
            remote_branch_verified=True,
            remote_merged=False,
            pull_request_state="closed",
        )


def test_retry_refuses_unmanaged_or_nonpolicy_failure(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    cycle = memory.start_cycle(
        task_id="ordinary-bug", branch="localpilot/candidate-ordinary-bug",
        everyday_model="daily", developer_model="developer", workspace=workspace,
        is_worktree=True,
    )
    memory.finish_cycle(
        cycle, status="candidate_needs_work", summary="Static syntax error",
        reusable_lesson="fix syntax", checks_passed=False, pushed=False,
    )
    with pytest.raises(ValueError, match="does not show a framework-policy"):
        memory.authorize_policy_retry("ordinary-bug", reason="try again")
    with pytest.raises(ValueError, match="No LocalPilot-managed"):
        memory.authorize_policy_retry("unmanaged", reason="try again")


def test_human_retry_resumes_full_same_objective_in_existing_worktree(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "selfdev-backlog.json").write_text(
        json.dumps({"tasks": [{"id": TASK, "title": "Build a guarded local model adaptation lab", "status": "todo"}]}),
        encoding="utf-8",
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "existing.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    prior_id = _policy_blocked(developer.memory, workspace)
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda branch: workspace if branch == BRANCH else None)

    authorized = developer.retry_candidate(BRANCH, reason="The restrictive framework caused this failure.")
    audit = developer.audit.latest("candidate_policy_retry_authorized")
    seen = {}
    monkeypatch.setattr(developer.github, "candidate_changed_paths", lambda _workspace: ["existing.py"])
    monkeypatch.setattr(developer.github, "branch_has_candidate_commit", lambda _workspace: False)
    monkeypatch.setattr(developer.github, "reviewer_modified_test_paths", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        developer, "_select_developer_model",
        lambda: DeveloperModelSelection("developer", 1, 1.0, "available"),
    )

    def continue_candidate(**kwargs):
        seen.update(kwargs)
        return EvolutionResult("candidate_needs_work", kwargs["branch"], kwargs["workspace"], "continued")

    monkeypatch.setattr(developer, "_continue_candidate", continue_candidate)
    result = developer._repair_local_candidate(force=True)

    assert authorized.prior_cycle_id == prior_id
    assert authorized.resume_mode == "resume_existing_worktree"
    assert audit is not None and audit["failure_attribution"] == "framework_policy"
    assert audit["candidate_idea_at_fault"] is False
    assert audit["history_preserved"] is True
    assert result is not None and result.summary == "continued"
    assert seen["task"]["id"] == TASK
    assert seen["branch"] == BRANCH
    assert seen["workspace"] == workspace
    assert seen["cycle_id"] == authorized.retry_cycle_id
    assert seen["checkpoint"] is None
    assert seen["tools"].files_written == {(workspace / "existing.py").resolve()}


def test_pushed_retry_uses_fresh_main_based_branch_and_resumes_clean_objective(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "selfdev-backlog.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": TASK,
                        "title": "Build a guarded local model adaptation lab",
                        "status": "todo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    old_workspace = tmp_path / "old-pushed-worktree"
    old_workspace.mkdir()
    (old_workspace / "historical.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    prior_id = _pushed_policy_blocked(developer.memory, old_workspace)
    worktrees = {BRANCH: old_workspace.resolve()}
    created = []

    monkeypatch.setattr(
        developer.github,
        "candidate_lifecycle",
        lambda branch: CandidateLifecycle(
            "passed",
            False,
            "https://example.test/pull/17",
            "open",
            True,
        ),
    )
    monkeypatch.setattr(
        developer.github,
        "worktree_for_branch",
        lambda branch: worktrees.get(branch),
    )

    def create_retry(branch, destination):
        destination.mkdir(parents=True)
        worktrees[branch] = destination.resolve()
        created.append((branch, destination.resolve()))
        return CommandResult(True, "created from main", "", 0)

    monkeypatch.setattr(developer.github, "create_policy_retry_worktree", create_retry)
    monkeypatch.setattr(
        developer.github,
        "remove_candidate_worktree",
        lambda *_args, **_kwargs: pytest.fail("historical worktree was removed"),
    )

    authorized = developer.retry_candidate(
        BRANCH,
        reason="The restrictive framework caused this failure.",
    )
    first_audit = developer.audit.latest("candidate_policy_retry_authorized")
    repeated = developer.retry_candidate(
        BRANCH,
        reason="A repeated invocation must use the existing linkage.",
    )

    assert authorized.prior_cycle_id == prior_id
    assert authorized.prior_branch == BRANCH
    assert authorized.branch != BRANCH
    assert authorized.branch.endswith(f"{RETRY_BRANCH_SUFFIX}{prior_id}")
    assert authorized.resume_mode == (
        "fresh_retry_branch_from_trusted_main_preserving_pushed_history"
    )
    assert repeated.retry_cycle_id == authorized.retry_cycle_id
    assert repeated.already_authorized is True
    assert len(created) == 1
    assert (old_workspace / "historical.py").exists()

    retry = developer.memory.candidate_for_cycle(authorized.retry_cycle_id)
    prior = developer.memory.candidate_for_cycle(prior_id)
    assert retry is not None and Path(retry.workspace) == created[0][1]
    assert prior is not None and prior.branch == BRANCH and prior.pushed is True
    assert prior.pull_request_url == "https://example.test/pull/17"
    assert prior.retried_by_cycle_id == retry.cycle_id
    assert first_audit is not None and first_audit["prior_pull_request_state"] == "open"
    assert first_audit["prior_pull_request_url"] == "https://example.test/pull/17"

    seen = {}
    monkeypatch.setattr(developer.github, "candidate_changed_paths", lambda _workspace: [])
    monkeypatch.setattr(developer.github, "branch_has_candidate_commit", lambda _workspace: False)
    monkeypatch.setattr(
        developer.github,
        "reviewer_modified_test_paths",
        lambda *_args, **_kwargs: set(),
    )
    monkeypatch.setattr(
        developer,
        "_select_developer_model",
        lambda: DeveloperModelSelection("developer", 1, 1.0, "available"),
    )

    def continue_candidate(**kwargs):
        seen.update(kwargs)
        return EvolutionResult(
            "candidate_needs_work",
            kwargs["branch"],
            kwargs["workspace"],
            "fresh retry continued",
        )

    monkeypatch.setattr(developer, "_continue_candidate", continue_candidate)
    resumed = developer._repair_local_candidate(force=True)
    assert resumed is not None and resumed.summary == "fresh retry continued"
    assert seen["task"]["id"] == TASK
    assert seen["branch"] == authorized.branch
    assert seen["cycle_id"] == authorized.retry_cycle_id
    assert seen["tools"].files_written == set()


def test_selfdeveloper_retry_fails_closed_for_main_checkout(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    _policy_blocked(developer.memory, project)

    with pytest.raises(CandidateRetryError, match="trusted main checkout"):
        developer.retry_candidate(BRANCH, reason="policy failure")


def test_selfdeveloper_refreshes_remote_merge_state_before_pushed_retry(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    prior_id = _pushed_policy_blocked(developer.memory, workspace)
    monkeypatch.setattr(
        developer.github,
        "worktree_for_branch",
        lambda branch: workspace if branch == BRANCH else None,
    )
    monkeypatch.setattr(
        developer.github,
        "candidate_lifecycle",
        lambda _branch: CandidateLifecycle(
            "passed",
            True,
            "https://example.test/pull/17",
            "merged",
            True,
        ),
    )

    with pytest.raises(CandidateRetryError, match="already been merged or promoted"):
        developer.retry_candidate(BRANCH, reason="policy failure")

    prior = developer.memory.candidate_for_cycle(prior_id)
    assert prior is not None and prior.merged is True
    assert prior.retried_by_cycle_id is None


def test_github_lifecycle_distinguishes_open_no_pr_and_merged(
    tmp_path: Path,
    monkeypatch,
):
    config = Config()
    github = SelfDeveloper(config, tmp_path).github
    monkeypatch.setattr(github, "remote_candidate_branch_exists", lambda _branch: True)
    monkeypatch.setattr(github, "gh_available", lambda: True)
    monkeypatch.setattr(
        github,
        "branch_workflow_state",
        lambda _branch: CandidateLifecycle("passed", False),
    )

    payload = {
        "value": [
            {
                "headRefName": BRANCH,
                "url": "https://example.test/pull/17",
                "state": "OPEN",
                "mergedAt": None,
                "statusCheckRollup": [],
            }
        ]
    }
    monkeypatch.setattr(
        github,
        "_run",
        lambda *_args, **_kwargs: CommandResult(
            True, json.dumps(payload["value"]), "", 0
        ),
    )
    opened = github.candidate_lifecycle(BRANCH)
    assert opened.pull_request_state == "open"
    assert opened.pull_request_url == "https://example.test/pull/17"
    assert opened.merged is False and opened.remote_branch_exists is True

    payload["value"] = []
    no_pr = github.candidate_lifecycle(BRANCH)
    assert no_pr.pull_request_state == "none"
    assert no_pr.pull_request_url is None and no_pr.merged is False

    payload["value"] = [
        {
            "headRefName": BRANCH,
            "url": "https://example.test/pull/17",
            "state": "CLOSED",
            "mergedAt": "2026-08-23T02:00:00Z",
            "statusCheckRollup": [],
        }
    ]
    merged = github.candidate_lifecycle(BRANCH)
    assert merged.pull_request_state == "merged"
    assert merged.merged is True


def test_fresh_retry_worktree_requires_clean_current_trusted_main(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    github = SelfDeveloper(Config(), project).github
    destination = tmp_path / "retry-worktree"
    retry_branch = f"{BRANCH}{RETRY_BRANCH_SUFFIX}17"
    head = "a" * 40
    responses = {
        ("git", "rev-parse", "--show-toplevel"): CommandResult(
            True, str(project.resolve()), "", 0
        ),
        ("git", "branch", "--show-current"): CommandResult(True, "main", "", 0),
        ("git", "status", "--porcelain", "--untracked-files=all"): CommandResult(
            True, "", "", 0
        ),
        ("git", "rev-parse", "--verify", "HEAD^{commit}"): CommandResult(
            True, head, "", 0
        ),
        (
            "git",
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main^{commit}",
        ): CommandResult(True, head, "", 0),
    }
    monkeypatch.setattr(
        github,
        "_run",
        lambda args, **_kwargs: responses[tuple(args)],
    )
    created = []
    monkeypatch.setattr(
        github,
        "create_candidate_worktree",
        lambda branch, path: (
            created.append((branch, path))
            or CommandResult(True, "created", "", 0)
        ),
    )

    result = github.create_policy_retry_worktree(retry_branch, destination)
    assert result.ok is True
    assert created == [(retry_branch, destination)]

    responses[("git", "branch", "--show-current")] = CommandResult(
        True, "feature/untrusted", "", 0
    )
    refused = github.create_policy_retry_worktree(retry_branch, destination)
    assert refused.ok is False
    assert "must be on main" in refused.stderr


def test_unregistered_existing_workspace_is_preserved_and_retry_rebuilds_same_objective(
    tmp_path: Path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    old_workspace = tmp_path / "orphaned-candidate"
    old_workspace.mkdir()
    (old_workspace / "evidence.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    prior_id = _policy_blocked(developer.memory, old_workspace)
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda _branch: None)

    result = developer.retry_candidate(BRANCH, reason="Framework policy blocked construction.")

    retry = developer.memory.candidate_for_cycle(result.retry_cycle_id)
    prior = developer.memory.candidate_for_cycle(prior_id)
    assert result.resume_mode == "rebuild_same_branch_and_objective_preserving_old_workspace"
    assert retry is not None and Path(retry.workspace) != old_workspace
    assert Path(retry.workspace).parent == project / "data" / "retries"
    assert retry.task_id == TASK and retry.branch == BRANCH
    assert prior is not None and prior.workspace == str(old_workspace.resolve())
    assert (old_workspace / "evidence.py").exists()


def test_retry_cli_requires_durable_reason():
    parser = build_parser()
    parsed = parser.parse_args(["retry", BRANCH, "--reason", "Framework policy failure."])
    assert parsed.command == "retry" and parsed.candidate == BRANCH
    with pytest.raises(SystemExit):
        parser.parse_args(["retry", BRANCH])


def test_status_shows_prior_pushed_attempt_and_linked_retry(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    memory = LearningMemory(project / "data" / config.selfdev.learning_database)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    prior_id = _pushed_policy_blocked(memory, workspace)
    retry = memory.authorize_policy_retry(
        BRANCH,
        reason="Framework policy blocked valid architecture.",
        remote_branch_verified=True,
        remote_merged=False,
        pull_request_state="open",
    )

    console = Console(record=True, width=180)
    _show_status(console, config, project)
    rendered = console.export_text()
    assert "Human-authorized policy retry" in rendered
    assert f"prior: {BRANCH} (pushed=True, attribution=framework_policy)" in rendered
    assert f"retry: {retry.branch}" in rendered
    assert f"retry cycle: {retry.retry_cycle_id}; prior cycle: {prior_id}" in rendered
