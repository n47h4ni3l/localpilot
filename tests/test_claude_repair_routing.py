from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.selfdev import CandidateTools


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_claude_repair_scope_uses_candidate_diff_and_excludes_reviewer_tests(tmp_path: Path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    test_path = workspace / "tests" / "test_contract.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_contract():\n    assert True\n", encoding="utf-8")
    _git(workspace, "init", "--quiet", "-b", "main")
    _git(workspace, "config", "user.name", "Repair Routing Test")
    _git(workspace, "config", "user.email", "repair-routing@invalid.local")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "--quiet", "-m", "base")
    _git(workspace, "checkout", "--quiet", "-b", "localpilot/candidate-routing-test")
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    test_path.write_text("def test_contract():\n    assert VALUE == 2\n", encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "--quiet", "-m", "candidate: routing test")

    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    tools = CandidateTools(
        workspace,
        protected_paths={"tests/test_contract.py"},
    )

    assert developer._claude_repair_scope(workspace, tools) == ("module.py",)


@pytest.mark.parametrize("stage", ["ci_repair", "local_static_repair"])
def test_configured_claude_backend_owns_autonomous_repair_edits(
    tmp_path: Path,
    monkeypatch,
    stage: str,
):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.implementation_backend = "claude_code"
    developer = SelfDeveloper(config, project)
    tools = CandidateTools(workspace)
    task = {
        "id": "repair-routing",
        "title": "Repair routing",
        "acceptance": ["candidate is repaired"],
        "evaluation": {
            "metric": "tests",
            "baseline": "failing",
            "success_criterion": "passing",
            "measurement_method": "CI",
        },
    }
    developer._active_checkpoint = {
        "cycle_id": 7,
        "task": task,
        "branch": "localpilot/candidate-routing-test",
        "workspace": workspace,
        "tools": tools,
    }

    monkeypatch.setattr(
        developer,
        "_claude_repair_scope",
        lambda _workspace, _tools: ("module.py",),
    )
    monkeypatch.setattr(developer, "_evolution_context", lambda _task: "bounded contract")
    monkeypatch.setattr(developer.memory, "reusable_lessons", lambda _limit: [])
    captured = {}

    def claude_impl(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "summary": "Claude repaired the candidate.",
                "reusable_lesson": "Keep repair edits in Claude Code.",
                "evaluation_evidence": {
                    "metric": "tests",
                    "baseline_evidence": "failing",
                    "candidate_evidence": "passing",
                    "result": "pending_ci",
                    "measurement_artifact": "CI",
                },
            }
        )

    monkeypatch.setattr(developer, "_run_claude_code_implementation", claude_impl)

    result = developer._tool_stage(
        chat=lambda **_kwargs: None,
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": "CI exploded in module.py; repair the concrete failure.",
            }
        ],
        functions=[],
        rounds=14,
        force=True,
        branch="localpilot/candidate-routing-test",
        stage=stage,
    )

    assert "Claude repaired" in result
    assert captured["grounding_plan"] == {
        "referenced_paths": ["module.py"],
        "new_runtime_paths": [],
    }
    assert "CI exploded in module.py" in captured["research"]
    assert captured["tools"] is tools
    assert captured["cycle_id"] == 7


def test_claude_repair_fails_closed_when_candidate_has_no_safe_owned_paths(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()

    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.implementation_backend = "claude_code"
    developer = SelfDeveloper(config, project)
    tools = CandidateTools(workspace)
    developer._active_checkpoint = {
        "cycle_id": 8,
        "task": {
            "id": "repair-routing",
            "title": "Repair routing",
            "acceptance": [],
            "evaluation": {
                "metric": "tests",
                "baseline": "failing",
                "success_criterion": "passing",
                "measurement_method": "CI",
            },
        },
        "branch": "localpilot/candidate-routing-test",
        "workspace": workspace,
        "tools": tools,
    }
    monkeypatch.setattr(developer, "_claude_repair_scope", lambda *_args: ())

    with pytest.raises(RuntimeError, match="no safe candidate-owned paths"):
        developer._tool_stage(
            chat=lambda **_kwargs: None,
            model="gpt-oss:20b",
            messages=[{"role": "system", "content": "repair"}],
            functions=[],
            rounds=14,
            force=True,
            branch="localpilot/candidate-routing-test",
            stage="ci_repair",
        )
