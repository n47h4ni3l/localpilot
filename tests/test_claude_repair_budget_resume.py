from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.implementation_backend import (
    ImplementationPreflight,
    ImplementationResult,
    ImplementationStatus,
)
from localpilot.selfdev import CandidateTools, SelfDeveloper as BaseSelfDeveloper
from localpilot.selfdev_results import CyclePaused


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Durable Repair Test")
    _git(root, "config", "user.email", "durable-repair@invalid.local")
    _git(root, "add", "module.py")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root


def _task() -> dict:
    return {
        "id": "durable-repair-fixture",
        "title": "durable repair fixture",
        "acceptance": ["review eventually passes"],
        "hypothesis": "durable repair progress survives a bounded invocation",
        "evaluation": {
            "metric": "review",
            "baseline": "rejected",
            "success_criterion": "approved",
            "measurement_method": "independent review",
        },
    }


class RepairingBackend:
    name = "claude_code"

    def __init__(self) -> None:
        self.calls = 0
        self.repair_passes: list[int] = []
        self.feedback: list[str] = []

    def preflight(self):
        return ImplementationPreflight(
            True,
            self.name,
            "gpt-oss:20b",
            "fake-test-double",
            version="test",
            context_tokens=65536,
        )

    def run(self, request, *, repair_pass=0):
        self.calls += 1
        self.repair_passes.append(repair_pass)
        self.feedback.append(request.review_feedback)
        (request.workspace / "module.py").write_text(
            f"VALUE = {self.calls + repair_pass + 1}\n",
            encoding="utf-8",
        )
        return ImplementationResult(
            ImplementationStatus.COMPLETED,
            self.name,
            "gpt-oss:20b",
            "candidate changed",
            changed_paths=("module.py",),
            diff_digest=f"{self.calls + repair_pass:064x}"[-64:],
            tests=(
                {
                    "command": "python -m pytest",
                    "passed": True,
                    "exit_code": 0,
                    "output_digest": "b" * 64,
                },
            ),
            session_id="same-session",
            repair_pass=repair_pass,
        )


def _run(developer: SelfDeveloper, workspace: Path) -> str:
    developer._active_claude_repair_stage = "ci_repair"
    return developer._run_claude_code_implementation(
        chat=lambda **_kwargs: None,
        developer_model="gpt-oss:20b",
        task=_task(),
        branch="localpilot/candidate-durable-repair",
        workspace=workspace,
        tools=CandidateTools(workspace),
        cycle_id=41,
        research="CI failed and needs a bounded repair.",
        grounding_plan={
            "referenced_paths": ["module.py"],
            "new_runtime_paths": [],
        },
        grounding_evidence=["module.py is candidate-owned"],
        evolution_context="bounded",
        lessons=[],
        force=True,
    )


def test_repair_pass_and_feedback_resume_after_cycle_budget_pause(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    workspace = _candidate(tmp_path)
    config = Config()
    config.agent.data_dir = "data"

    first_backend = RepairingBackend()
    first = SelfDeveloper(config, project, implementation_backend=first_backend)
    admissions: list[int] = []

    def bounded_admission(*, force: bool, branch: str, stage: str) -> None:
        admissions.append(len(admissions) + 1)
        if len(admissions) == 2:
            raise CyclePaused(
                "evolution budget exhausted: cycle wall-clock budget exhausted after 900s"
            )
        first._claude_unit_admitted = True

    monkeypatch.setattr(first, "_admit_claude_unit", bounded_admission)
    monkeypatch.setattr(
        first,
        "_developer_chat",
        lambda *args, **kwargs: {
            "content": json.dumps(
                {
                    "approved": False,
                    "feedback": ["Fix the remaining CI contract violation."],
                    "summary": "one more repair is required",
                }
            )
        },
    )

    with pytest.raises(CyclePaused, match="wall-clock budget exhausted"):
        _run(first, workspace)

    assert first_backend.repair_passes == [0]
    progress = first._load_claude_repair_progress(
        branch="localpilot/candidate-durable-repair",
        cycle_id=41,
        stage="ci_repair",
    )
    assert progress is not None
    assert progress["repair_pass"] == 1
    assert "remaining CI contract" in progress["review_feedback"]

    second_backend = RepairingBackend()
    resumed = SelfDeveloper(config, project, implementation_backend=second_backend)

    def admit_resume(*, force: bool, branch: str, stage: str) -> None:
        resumed._claude_unit_admitted = True

    monkeypatch.setattr(resumed, "_admit_claude_unit", admit_resume)
    monkeypatch.setattr(
        resumed,
        "_developer_chat",
        lambda *args, **kwargs: {
            "content": json.dumps(
                {"approved": True, "feedback": [], "summary": "accept"}
            )
        },
    )

    result = _run(resumed, workspace)

    assert second_backend.repair_passes == [1]
    assert "remaining CI contract" in second_backend.feedback[0]
    assert "LocalPilot review approved" in result
    assert resumed._load_claude_repair_progress(
        branch="localpilot/candidate-durable-repair",
        cycle_id=41,
        stage="ci_repair",
    ) is None


def test_admitted_claude_unit_defers_wall_clock_budget_but_restores_it(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, project)
    sentinel = object()
    developer._budget = sentinel
    developer._claude_unit_admitted = True
    observed = []

    def base_check(self, force, branch, *, during_inference=False):
        observed.append(self._budget)

    monkeypatch.setattr(BaseSelfDeveloper, "_check_resources", base_check)

    developer._check_resources(True, "candidate/test", during_inference=True)

    assert observed == [None]
    assert developer._budget is sentinel
