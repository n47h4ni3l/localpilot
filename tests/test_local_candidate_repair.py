import json
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.github_integration import CommandResult
from localpilot.learning import LearningMemory
from localpilot.selfdev import CandidateTools, SelfDeveloper


def _start_local_cycle(memory: LearningMemory, workspace: Path) -> int:
    return memory.start_cycle(
        task_id="repair-task",
        branch="localpilot/candidate-repair-task",
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )


def test_unpushed_candidate_is_pending_and_recoverable(tmp_path: Path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    cycle = _start_local_cycle(memory, workspace)
    memory.finish_cycle(
        cycle,
        status="candidate_needs_work",
        summary="static_checks=failed",
        reusable_lesson="repair the same candidate",
        checks_passed=False,
        pushed=False,
    )

    assert memory.pending_task_ids() == {"repair-task"}
    candidates = memory.local_candidates()
    assert len(candidates) == 1
    assert candidates[0].cycle_id == cycle
    assert candidates[0].workspace == str(workspace.resolve())
    assert candidates[0].local_repair_attempts == 0


def test_repair_attempt_is_counted_before_model_work(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    cycle = _start_local_cycle(memory, tmp_path / "candidate")

    assert memory.record_local_repair_attempt(
        cycle,
        check_result="static_checks=failed\ntests/test_candidate_changes.py: SyntaxError",
    ) == 1
    assert memory.record_local_repair_attempt(
        cycle,
        check_result="static_checks=failed\ntests/test_candidate_changes.py: SyntaxError",
    ) == 2
    assert memory.local_candidates()[0].local_repair_attempts == 2


def test_static_failure_feedback_repairs_same_workspace(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    broken = workspace / "tests" / "test_candidate_changes.py"
    broken.parent.mkdir(parents=True)
    broken.write_text("this is invalid python !!!\n", encoding="utf-8")

    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.max_local_repair_attempts = 2
    developer = SelfDeveloper(config, project)
    cycle = _start_local_cycle(developer.memory, workspace)
    tools = CandidateTools(
        workspace,
        existing_changed_paths={"tests/test_candidate_changes.py"},
    )
    seen_prompts: list[str] = []

    def fake_stage(**kwargs):
        seen_prompts.append(kwargs["messages"][0]["content"])
        tools.write_project_file(
            "tests/test_candidate_changes.py",
            "def test_candidate_changes():\n    assert True\n",
        )
        return json.dumps(
            {
                "summary": "Repaired generated test syntax.",
                "reusable_lesson": "Compile generated tests before push.",
            }
        )

    monkeypatch.setattr(developer, "_tool_stage", fake_stage)
    monkeypatch.setattr(developer, "_check_resources", lambda *_args, **_kwargs: None)
    initial = tools.run_candidate_static_checks()

    result = developer._repair_static_failures(
        chat=lambda **_kwargs: None,
        model="developer",
        tools=tools,
        task={"id": "repair-task", "title": "Repair task", "acceptance": []},
        branch="localpilot/candidate-repair-task",
        cycle_id=cycle,
        check_result=initial,
        attempts_used=0,
        force=True,
    )

    assert result.passed is True
    assert result.attempts_used == 1
    assert "SyntaxError" in seen_prompts[0]
    assert "tests/test_candidate_changes.py" in seen_prompts[0]
    assert "this is invalid python" in seen_prompts[0]
    assert broken.read_text(encoding="utf-8").startswith("def test_candidate_changes")
    assert developer.memory.local_candidates()[0].local_repair_attempts == 1


def test_static_repair_stops_at_durable_attempt_limit(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "broken.py").write_text("invalid !!!\n", encoding="utf-8")

    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.max_local_repair_attempts = 2
    developer = SelfDeveloper(config, project)
    cycle = _start_local_cycle(developer.memory, workspace)
    tools = CandidateTools(workspace, existing_changed_paths={"broken.py"})
    stages: list[int] = []

    def stalled_stage(**_kwargs):
        stages.append(1)
        return "No edit made."

    monkeypatch.setattr(developer, "_tool_stage", stalled_stage)
    monkeypatch.setattr(developer, "_check_resources", lambda *_args, **_kwargs: None)

    result = developer._repair_static_failures(
        chat=lambda **_kwargs: {"message": {"content": "not a change plan"}},
        model="developer",
        tools=tools,
        task={"id": "repair-task", "title": "Repair task", "acceptance": []},
        branch="localpilot/candidate-repair-task",
        cycle_id=cycle,
        check_result=tools.run_candidate_static_checks(),
        attempts_used=0,
        force=True,
    )

    assert result.passed is False
    assert result.attempts_used == 2
    assert len(stages) == 2
    assert developer.memory.local_candidates()[0].local_repair_attempts == 2


def test_recovery_reuses_worktree_and_branch_before_push(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "selfdev-backlog.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "repair-task",
                        "title": "Repair task",
                        "status": "todo",
                        "acceptance": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "candidate"
    changed = workspace / "localpilot" / "change.py"
    changed.parent.mkdir(parents=True)
    changed.write_text("VALUE = 1\n", encoding="utf-8")

    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    cycle = _start_local_cycle(developer.memory, workspace)
    developer.memory.finish_cycle(
        cycle,
        status="candidate_needs_work",
        summary="stale local candidate",
        reusable_lesson="resume it",
        checks_passed=False,
        pushed=False,
    )
    calls: list[tuple[str, Path, str]] = []

    monkeypatch.setattr(
        developer.github,
        "worktree_for_branch",
        lambda branch: workspace if branch == "localpilot/candidate-repair-task" else None,
    )
    monkeypatch.setattr(
        developer.github,
        "candidate_changed_paths",
        lambda candidate_workspace: ["localpilot/change.py"],
    )
    monkeypatch.setattr(
        developer.github,
        "branch_has_candidate_commit",
        lambda _workspace: False,
    )
    monkeypatch.setattr(
        developer.github,
        "reviewer_modified_test_paths",
        lambda _workspace, refresh=False: set(),
    )

    def commit(candidate_workspace, message, paths):
        calls.append(("commit", candidate_workspace, message))
        assert paths == ["localpilot/change.py"]
        return CommandResult(True, "committed", "", 0)

    def push(candidate_workspace, branch):
        calls.append(("push", candidate_workspace, branch))
        return CommandResult(True, "pushed", "", 0)

    monkeypatch.setattr(developer.github, "commit_paths", commit)
    monkeypatch.setattr(developer.github, "push_branch", push)

    result = developer._repair_local_candidate(force=True)

    assert result is not None
    assert result.status == "candidate_pending_validation"
    assert result.branch == "localpilot/candidate-repair-task"
    assert result.workspace == workspace
    assert calls == [
        ("commit", workspace, "candidate: Repair task"),
        ("push", workspace, "localpilot/candidate-repair-task"),
    ]
    assert developer.memory.pending_candidates()[0].cycle_id == cycle


def test_recovered_changes_still_count_toward_file_limit(tmp_path: Path):
    (tmp_path / "existing.py").write_text("VALUE = 1\n", encoding="utf-8")
    tools = CandidateTools(
        tmp_path,
        max_files=2,
        existing_changed_paths={"existing.py"},
    )

    tools.write_project_file("new.py", "VALUE = 2\n")

    try:
        tools.write_project_file("too_many.py", "VALUE = 3\n")
    except RuntimeError as exc:
        assert "file-write limit" in str(exc)
    else:
        raise AssertionError("Recovered candidate bypassed the file limit")


def test_recovered_candidate_cannot_seed_protected_or_excess_changes(tmp_path: Path):
    protected = tmp_path / "tests" / "test_contract.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("def test_contract(): pass\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="Reviewer-controlled"):
        CandidateTools(
            tmp_path,
            protected_paths={"tests/test_contract.py"},
            existing_changed_paths={"tests/test_contract.py"},
        )

    (tmp_path / "one.py").write_text("ONE = 1\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("TWO = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file-write limit"):
        CandidateTools(
            tmp_path,
            max_files=1,
            existing_changed_paths={"one.py", "two.py"},
        )


def test_exhausted_failed_candidate_is_never_committed_or_pushed(
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
                        "id": "repair-task",
                        "title": "Repair task",
                        "status": "todo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "broken.py").write_text("invalid !!!\n", encoding="utf-8")
    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.max_local_repair_attempts = 1
    developer = SelfDeveloper(config, project)
    cycle = _start_local_cycle(developer.memory, workspace)
    developer.memory.record_local_repair_attempt(
        cycle,
        check_result="static_checks=failed\nbroken.py: SyntaxError",
    )

    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda _branch: workspace)
    monkeypatch.setattr(developer.github, "candidate_changed_paths", lambda _workspace: ["broken.py"])
    monkeypatch.setattr(developer.github, "branch_has_candidate_commit", lambda _workspace: False)
    monkeypatch.setattr(
        developer.github,
        "reviewer_modified_test_paths",
        lambda _workspace, refresh=False: set(),
    )
    monkeypatch.setattr(
        developer.github,
        "commit_paths",
        lambda *_args, **_kwargs: pytest.fail("static-failing candidate was committed"),
    )
    monkeypatch.setattr(
        developer.github,
        "push_branch",
        lambda *_args, **_kwargs: pytest.fail("static-failing candidate was pushed"),
    )

    result = developer._repair_local_candidate(force=True)

    assert result is not None
    assert result.status == "candidate_needs_work"
    assert result.tests_passed is False
    assert "attempt limit reached" in result.summary
