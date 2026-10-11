from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import pytest

from localpilot.selfdev import CandidateTools
from localpilot.selfdev import SelfDeveloper
from localpilot.config import Config
from localpilot.github_integration import CommandResult
from localpilot.selfdev_response_parsing import StaticRepairResult


def test_oversized_candidate_can_be_replaced_but_cannot_pass_unchanged(tmp_path: Path):
    path = tmp_path / "broken.py"
    content = "# oversized historical edit\n" * 40000
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds 1 MB"):
        CandidateTools(tmp_path, existing_changed_paths=["broken.py"])

    tools = CandidateTools(
        tmp_path, existing_changed_paths=["broken.py"], allow_oversized_repair=True
    )
    assert tools.oversized_repair_paths == {"broken.py"}
    assert tools.read_project_file("broken.py", 1000) == content[:1000]
    assert path.read_text(encoding="utf-8") == content
    assert tools.run_candidate_static_checks().startswith("static_checks=failed")
    with pytest.raises(ValueError, match="1 MB safety limit"):
        tools.write_project_file("broken.py", content)
    assert path.read_text(encoding="utf-8") == content
    tools.write_project_file("broken.py", "VALUE = 1\n")
    assert tools.run_candidate_static_checks().startswith("static_checks=passed")


def test_bounded_read_does_not_materialize_whole_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "broken.py"
    path.write_text("# line\n" * 160000, encoding="utf-8")
    tools = CandidateTools(
        tmp_path, existing_changed_paths=["broken.py"], allow_oversized_repair=True
    )
    def unbounded_read(*args, **kwargs):
        raise AssertionError("Read attempted to load the entire oversized file")
    monkeypatch.setattr(Path, "read_text", unbounded_read)
    assert len(tools.read_project_file("broken.py", 1200)) == 1200


@pytest.mark.parametrize("relative", ["secret.exe", "../outside.py"])
def test_repair_does_not_admit_disallowed_paths(tmp_path: Path, relative: str):
    path = tmp_path / relative
    path.write_text("x" * 1000001, encoding="utf-8")
    with pytest.raises((ValueError, PermissionError)):
        CandidateTools(tmp_path, existing_changed_paths=[relative], allow_oversized_repair=True)


def test_reviewer_protection_still_blocks_oversized_repair(tmp_path: Path):
    (tmp_path / "reviewer.py").write_text("# protected\n" * 100000, encoding="utf-8")
    with pytest.raises(PermissionError, match="Reviewer-controlled"):
        CandidateTools(
            tmp_path, protected_paths=["reviewer.py"],
            existing_changed_paths=["reviewer.py"], allow_oversized_repair=True,
        )


@pytest.mark.parametrize("replace_oversized", [True, False])
def test_failed_ci_cycle_can_repair_size_fault_before_delivery(tmp_path: Path, monkeypatch, replace_oversized: bool):
    project = tmp_path / "project"
    workspace = tmp_path / "candidate"
    project.mkdir()
    workspace.mkdir()
    branch = "localpilot/candidate-oversized-20261011-000000"
    for args in (["init", "-b", branch], ["config", "user.name", "Test"], ["config", "user.email", "test@example.invalid"]):
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)
    path = workspace / "broken.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "broken.py"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "candidate"], cwd=workspace, check=True, capture_output=True)
    path.write_text("# oversized historical edit\n" * 40000, encoding="utf-8")
    task = {"id": "size-repair", "title": "Repair size failure", "acceptance": [], "evolution_class": "repair"}
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    cycle = developer.memory.start_cycle(task_id=task["id"], branch=branch, everyday_model="test", developer_model="test")
    developer.memory.update_candidate_workspace(cycle, workspace)
    developer.memory.finish_cycle(cycle, status="candidate_pending_validation", summary="CI failed", reusable_lesson="repair", checks_passed=False, pushed=True)
    developer.memory.update_candidate_review(cycle, validation_state="failed", merged=False, pull_request_url=None)
    monkeypatch.setattr(developer, "_load_task_by_id", lambda identifier: task)
    monkeypatch.setattr(developer, "_select_developer_model", lambda: SimpleNamespace(model="test"))
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda value: workspace)
    monkeypatch.setattr(developer.github, "reviewer_modified_test_paths", lambda value: set())
    monkeypatch.setattr(developer.github, "candidate_changed_paths", lambda value: ["broken.py"])
    monkeypatch.setattr(developer.github, "failed_workflow_log", lambda value: "SyntaxError in broken.py")
    deliveries = []
    monkeypatch.setattr(developer.github, "commit_paths", lambda *args: deliveries.append("commit") or CommandResult(True, "committed", "", 0))
    monkeypatch.setattr(developer.github, "push_branch", lambda *args: deliveries.append("push") or CommandResult(True, "pushed", "", 0))
    def repair(**kwargs):
        tools = developer._active_checkpoint["tools"]
        assert tools.oversized_repair_paths == {"broken.py"}
        assert len(tools.read_project_file("broken.py", 1000)) == 1000
        if replace_oversized:
            tools.write_project_file("broken.py", "VALUE = 2\n")
        return json.dumps({"summary": "Bounded repair", "reusable_lesson": "Validate before delivery"})
    monkeypatch.setattr(developer, "_tool_stage", repair)
    monkeypatch.setattr(developer, "_repair_static_failures", lambda **kwargs: StaticRepairResult(kwargs["check_result"], False, "No valid repair", 1))
    result = developer._repair_failed_candidate(force=True)
    assert result is not None
    if replace_oversized:
        assert result.status == "candidate_pending_validation"
        assert deliveries == ["commit", "push"]
        assert path.read_text(encoding="utf-8") == "VALUE = 2\n"
        assert developer.memory.pending_candidates()[0].cycle_id == cycle
    else:
        assert result.tests_passed is False
        assert deliveries == []
        assert path.stat().st_size > 1_000_000
