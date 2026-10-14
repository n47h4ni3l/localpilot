from __future__ import annotations

import json
from types import SimpleNamespace

from localpilot.config import Config, GitHubConfig
from localpilot.github_integration import CandidateLifecycle, CommandResult, GitHubIntegration
from localpilot.pr_revision_feedback import (
    MARKER,
    OwnerRevision,
    OwnerRevisionLedger,
    parse_owner_revision,
)
from localpilot.selfdev import DeveloperModelSelection, EvolutionResult, SelfDeveloper


BRANCH = "localpilot/candidate-owner-review-20261014-a1b2c3d4"


def test_only_verified_owner_comment_with_exact_prefix_triggers() -> None:
    good = {"id": 43, "body": MARKER + "\nImplement the missing benchmark.", "user": {"login": "repo-owner"}}
    parsed = parse_owner_revision(good, owner_login="repo-owner", pr_number=220, branch=BRANCH)
    assert parsed == OwnerRevision(43, 220, BRANCH, "Implement the missing benchmark.")
    assert parse_owner_revision({**good, "user": {"login": "someone-else"}}, owner_login="repo-owner", pr_number=220, branch=BRANCH) is None
    assert parse_owner_revision({**good, "body": "Quoted review: " + MARKER + " hack"}, owner_login="repo-owner", pr_number=220, branch=BRANCH) is None
    assert parse_owner_revision({**good, "body": MARKER + "   "}, owner_login="repo-owner", pr_number=220, branch=BRANCH) is None
    assert parse_owner_revision({**good, "body": MARKER + "a" * 4001}, owner_login="repo-owner", pr_number=220, branch=BRANCH) is None
    assert parse_owner_revision({**good, "id": True}, owner_login="repo-owner", pr_number=220, branch=BRANCH) is None


def test_private_receipt_ledger_prevents_replaying_comment(tmp_path) -> None:
    file = tmp_path / "ignored" / "owner-pr-revisions.json"
    ledger = OwnerRevisionLedger(file)
    command = OwnerRevision(19, 220, BRANCH, "Deliver an executable comparison.")
    assert ledger.begin(command, 81, "capability-measure") is True
    assert ledger.begin(command, 82, "capability-measure") is False
    assert ledger.active_for_task("capability-measure")["feedback"] == command.feedback
    assert ledger.active(BRANCH)["comment_id"] == 19
    reopened = OwnerRevisionLedger(file)
    assert reopened.seen(BRANCH) == {19}
    reopened.finish(19, "completed")
    assert reopened.active(BRANCH) is None
    assert reopened.begin(command, 83, "capability-measure") is False
    assert reopened.begin(OwnerRevision(20, 220, BRANCH, "New request"), 84, "capability-measure")
    assert reopened.seen(BRANCH) == {19, 20}


def test_pr_lookup_rejects_nonowner_cross_repo_wrong_branch_and_closed(monkeypatch, tmp_path) -> None:
    api = GitHubIntegration(tmp_path, GitHubConfig())
    monkeypatch.setattr(api, "gh_available", lambda: True)
    owner = {"id": 101, "user": {"login": "repo-owner"}, "body": MARKER + " Fix incomplete code"}
    nonowner = {"id": 102, "user": {"login": "random"}, "body": MARKER + " Remove guard"}
    ordinary = {"id": 103, "user": {"login": "repo-owner"}, "body": "Please inspect this"}
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["gh", "repo", "view"]:
            return CommandResult(True, '{"nameWithOwner":"repo-owner/localpilot"}', "", 0)
        if args[:3] == ["gh", "pr", "list"]:
            return CommandResult(True, json.dumps([
                {"number": 220, "headRefName": BRANCH, "baseRefName": "main", "isCrossRepository": False},
                {"number": 221, "headRefName": "another", "baseRefName": "main", "isCrossRepository": False},
            ]), "", 0)
        if args[:2] == ["gh", "api"]:
            return CommandResult(True, json.dumps([nonowner, owner, ordinary]), "", 0)
        raise AssertionError(args)

    monkeypatch.setattr(api, "_run", run)
    assert api.owner_revision_requests("unexpected-branch") == []
    assert not calls
    requests = api.owner_revision_requests(BRANCH)
    assert requests == [OwnerRevision(101, 220, BRANCH, "Fix incomplete code")]
    assert calls[-1][-1].startswith("repos/repo-owner/localpilot/issues/220/comments")

    def cross_repo(args, **kwargs):
        if args[:3] == ["gh", "repo", "view"]:
            return CommandResult(True, '{"nameWithOwner":"repo-owner/localpilot"}', "", 0)
        if args[:3] == ["gh", "pr", "list"]:
            return CommandResult(True, json.dumps([
                {"number": 220, "headRefName": BRANCH, "baseRefName": "main", "isCrossRepository": True}
            ]), "", 0)
        raise AssertionError("Never fetch feedback from an ineligible PR")

    monkeypatch.setattr(api, "_run", cross_repo)
    assert api.owner_revision_requests(BRANCH) == []


def test_owner_request_reuses_same_green_pr_branch_exactly_once(monkeypatch, tmp_path) -> None:
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    workspace = tmp_path / "existing-pr-worktree"
    workspace.mkdir()
    task = {
        "id": "factuality-baseline", "title": "Measure factuality",
        "acceptance": ["Implement a real baseline metric."],
        "evaluation": {"metric": "hallucination rate", "baseline": "unknown",
                       "measurement_method": "run the benchmark"},
    }
    cycle = developer.memory.start_cycle(
        task_id=task["id"], branch=BRANCH,
        everyday_model="nestra", developer_model="developer",
        workspace=workspace, is_worktree=True,
    )
    developer.memory.finish_cycle(
        cycle, status="candidate_pending_validation",
        summary="Initial draft", reusable_lesson="Pending review",
        checks_passed=True, pushed=True, validation_state="passed",
    )
    monkeypatch.setattr(developer, "_load_task_by_id",
                        lambda key: {**task, "acceptance": [
                            *task["acceptance"],
                            *([OwnerRevisionLedger(developer.data_dir / "owner-pr-revisions.json").active(BRANCH)["feedback"]]
                              if OwnerRevisionLedger(developer.data_dir / "owner-pr-revisions.json").active(BRANCH) else [])
                        ]})
    monkeypatch.setattr(developer.github, "candidate_lifecycle",
                        lambda branch: CandidateLifecycle("passed", False, "https://github.com/repo-owner/localpilot/pull/220", "open", True))
    monkeypatch.setattr(developer.github, "owner_revision_requests",
                        lambda branch: [OwnerRevision(250, 220, BRANCH, "Add a reproducible and independently checked evaluation.")])
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda branch: workspace)
    monkeypatch.setattr(developer.github, "reviewer_modified_test_paths", lambda workspace: set())
    monkeypatch.setattr(developer.github, "_run",
                        lambda args, **kwargs: CommandResult(
                            True,
                            "" if "status" in args else "abc123" if "rev-parse" in args
                            else f"abc123 refs/heads/{BRANCH}", "", 0))
    monkeypatch.setattr(developer, "_select_developer_model",
                        lambda: SimpleNamespace(model="test-developer", reason=""))
    monkeypatch.setattr(developer, "_candidate_tools", lambda *args, **kwargs: object())

    observed = []
    def finish(**kwargs):
        observed.append(kwargs)
        assert kwargs["branch"] == BRANCH
        assert kwargs["workspace"] == workspace
        assert kwargs["task"]["acceptance"][-1].startswith(
            "Add a reproducible and independently checked evaluation."
        )
        developer.memory.finish_cycle(
            kwargs["cycle_id"], status="candidate_pending_validation", summary="Revised",
            reusable_lesson="Owner request implemented", checks_passed=True,
            pushed=True, validation_state="awaiting_pr",
        )
        return EvolutionResult("candidate_pending_validation", BRANCH, workspace, "Revised", True)
    monkeypatch.setattr(developer, "_continue_candidate", finish)

    assert developer._revise_owner_reviewed_candidate(force=False).status == "candidate_pending_validation"
    assert len(observed) == 1
    assert OwnerRevisionLedger(developer.data_dir / "owner-pr-revisions.json").seen(BRANCH) == {250}
    assert developer._revise_owner_reviewed_candidate(force=False) is None
    assert len(observed) == 1


def test_owner_feedback_refuses_stale_pr_head(monkeypatch, tmp_path) -> None:
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    workspace = tmp_path / "candidate"; workspace.mkdir()
    cycle = developer.memory.start_cycle(
        task_id="task", branch=BRANCH, everyday_model="nestra",
        developer_model="developer", workspace=workspace, is_worktree=True,
    )
    developer.memory.finish_cycle(
        cycle, status="candidate_pending_validation", summary="",
        reusable_lesson="", checks_passed=True, pushed=True, validation_state="passed",
    )
    monkeypatch.setattr(developer.github, "candidate_lifecycle",
                        lambda branch: CandidateLifecycle("passed", False, "url", "open", True))
    monkeypatch.setattr(developer.github, "owner_revision_requests",
                        lambda branch: [OwnerRevision(777, 220, BRANCH, "Fix the benchmark")])
    monkeypatch.setattr(developer, "_load_task_by_id", lambda key: {"id": key, "title": "task"})
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda branch: workspace)
    monkeypatch.setattr(developer.github, "_run",
                        lambda args, **kwargs: CommandResult(
                            True, "" if "status" in args else "local-sha" if "rev-parse" in args
                            else f"remote-sha refs/heads/{BRANCH}", "", 0))
    result = developer._revise_owner_reviewed_candidate(force=False)
    assert result.status == "deferred"
    assert not OwnerRevisionLedger(developer.data_dir / "owner-pr-revisions.json").seen(BRANCH)
