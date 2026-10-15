from __future__ import annotations

import json
import subprocess
from pathlib import Path

from localpilot.config import Config
from localpilot.implementation_backend import (
    ImplementationPreflight,
    ImplementationResult,
    ImplementationStatus,
)
from localpilot.selfdev import CandidateTools, SelfDeveloper


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    for command in (
        ("init", "-q"),
        ("config", "user.name", "Review Scope Test"),
        ("config", "user.email", "review-scope@invalid.local"),
        ("add", "."),
        ("commit", "-qm", "baseline"),
    ):
        proc = subprocess.run(
            ["git", "-C", str(root), *command],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
    return root


class FakeBackend:
    name = "claude_code"

    def __init__(self, *, change: bool):
        self.change = change
        self.calls = 0
        self.prompts: list[str] = []

    def preflight(self):
        return ImplementationPreflight(
            True, self.name, "nestra-test", "fake",
            version="test", context_tokens=65536,
        )

    def run(self, request, *, repair_pass=0):
        self.calls += 1
        self.prompts.append(request.prompt)
        if self.change:
            (request.workspace / "module.py").write_text(
                f"VALUE = {self.calls + 1}\n", encoding="utf-8"
            )
        return ImplementationResult(
            ImplementationStatus.COMPLETED,
            self.name, "nestra-test", "syntax repaired",
            changed_paths=("module.py",) if self.change else (),
            diff_digest="a" * 64 if self.change else "",
            tests=({
                "command": "python -m pytest -q",
                "passed": True, "exit_code": 0,
                "output_digest": "b" * 64,
            },),
            repair_pass=repair_pass,
        )


def _run(tmp_path, monkeypatch, *, change: bool, approve: bool):
    workspace = _candidate(tmp_path)
    config = Config()
    config.agent.data_dir = "data"
    backend = FakeBackend(change=change)
    developer = SelfDeveloper(config, workspace, implementation_backend=backend)
    developer._active_claude_repair_stage = "local_static_repair"
    reviews: list[str] = []

    def reviewer(*args, **kwargs):
        reviews.append(kwargs["messages"][0]["content"])
        return {"content": json.dumps({
            "approved": approve,
            "feedback": ["missing immediate fix"] if not approve else [],
            "summary": "repair accepted" if approve else "repair needs work",
        })}
    monkeypatch.setattr(developer, "_developer_chat", reviewer)
    task = {
        "id": "long-term-baseline", "title": "Measure hallucinations",
        "acceptance": ["Finish a complete factuality benchmark with ±1% precision."],
        "evaluation": {
            "metric": "factual errors", "baseline": "unknown",
            "measurement_method": "benchmark",
        },
    }
    result = None
    error = None
    try:
        result = developer._run_claude_code_implementation(
            chat=lambda **kwargs: None, developer_model="nestra-test",
            task=task, branch="candidate/test", workspace=workspace,
            tools=CandidateTools(workspace), cycle_id=1,
            research="Fix the syntax error in module.py.",
            grounding_plan={
                "referenced_paths": ["module.py"], "new_runtime_paths": []
            },
            grounding_evidence=["module.py"],
            evolution_context="Overall benchmark remains unfinished.",
            lessons=[], force=True,
        )
    except RuntimeError as exc:
        error = str(exc)
    return backend, reviews, result, error


def test_syntax_repair_reviews_only_immediate_contract(tmp_path, monkeypatch):
    backend, reviews, outcome, error = _run(
        tmp_path, monkeypatch, change=True, approve=True
    )
    assert error is None
    assert "review approved" in outcome
    assert backend.calls == 1
    assert "Immediate acceptance" in backend.prompts[0]
    assert "Fix the concrete recorded static/CI error" in backend.prompts[0]
    assert "not a pass/fail criterion" in backend.prompts[0]
    assert "Immediate acceptance" in reviews[0]
    assert "do not demand completion" in reviews[0]
    assert "±1%" not in reviews[0]


def test_no_diff_progress_stops_after_one_review_repair(tmp_path, monkeypatch):
    backend, reviews, outcome, error = _run(
        tmp_path, monkeypatch, change=False, approve=False
    )
    assert outcome is None
    assert "review_rejected" in error
    assert backend.calls == 2
    assert len(reviews) == 1
