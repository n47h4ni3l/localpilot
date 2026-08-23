import json
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.cli import build_parser
from localpilot.learning import LearningMemory
from localpilot.selfdev import (
    CandidateRetryError,
    DeveloperModelSelection,
    EvolutionResult,
    SelfDeveloper,
)


BRANCH = "localpilot/candidate-model-adaptation-lab-20260823-003946"
TASK = "model-adaptation-lab"


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


def test_selfdeveloper_retry_fails_closed_for_main_checkout(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    _policy_blocked(developer.memory, project)

    with pytest.raises(CandidateRetryError, match="trusted main checkout"):
        developer.retry_candidate(BRANCH, reason="policy failure")


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
