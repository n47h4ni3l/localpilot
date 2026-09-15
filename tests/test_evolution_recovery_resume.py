from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from localpilot.checkpoint import EvolutionCheckpoint
from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.selfdev import SelfDeveloper as BaseSelfDeveloper
from localpilot.selfdev_results import EvolutionResult


def _task() -> dict:
    return {
        "id": "resume-existing-candidate",
        "title": "Resume an existing candidate",
        "acceptance": ["Retained candidate work is revalidated without repeating implementation."],
        "evolution_class": "improve_cognition",
        "capability_target": "Reliable candidate recovery",
        "mission_alignment": "Preserve useful autonomous work across bounded invocations.",
        "current_frontier": "Recovery can retain an implementation diff.",
        "why_high_leverage": "Avoids repeating expensive completed stages.",
        "capability_unlocked": "Stage-correct candidate recovery.",
        "next_frontier": "Long-running autonomous development.",
        "question": "Can an existing diff resume directly at validation?",
        "observed_limitation": "Recovery previously re-entered grounding.",
        "evidence": ["A retained candidate already contains implementation changes."],
        "alternatives": ["Repeat grounding", "Revalidate the existing diff"],
        "hypothesis": "Revalidation preserves progress and avoids irrelevant grounding failures.",
        "evaluation": {
            "metric": "completed recovery resumes",
            "baseline": "repeated pre-write stages",
            "success_criterion": "resume begins at validation or repair",
            "measurement_method": "checkpoint stage regression test",
        },
        "expected_complexity": "low",
    }


def _developer(tmp_path: Path) -> SelfDeveloper:
    config = Config()
    config.agent.data_dir = "data"
    return SelfDeveloper(config, tmp_path)


def _checkpoint(tmp_path: Path, *, static_status: str | None = None) -> EvolutionCheckpoint:
    return EvolutionCheckpoint.create(
        cycle_id=56,
        task=_task(),
        branch="localpilot/candidate-resume-existing-candidate-20260915-000000",
        workspace=tmp_path / "candidate",
        milestone="recovery",
        files_changed=["localpilot/example.py"],
        git_head="a" * 40,
        git_state_digest="b" * 64,
        static_check_status=static_status,
        static_check_failures=("example failure",) if static_status == "failed" else (),
        next_action="Revalidate the candidate and resolve the recorded failure before delivery.",
    )


def test_recovery_with_existing_diff_resumes_at_validation_not_grounding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    checkpoint = _checkpoint(tmp_path)
    captured = {}

    def fake_resume(_self, validated, *, force):
        captured["checkpoint"] = validated[0]
        captured["force"] = force
        return EvolutionResult("candidate_needs_work", validated[0].branch, workspace, "captured", False)

    monkeypatch.setattr(BaseSelfDeveloper, "_resume_checkpoint_candidate", fake_resume)
    result = developer._resume_checkpoint_candidate(
        (checkpoint, SimpleNamespace(), _task(), workspace),
        force=True,
    )

    resumed = captured["checkpoint"]
    persisted = developer.checkpoints.load()
    assert result.status == "candidate_needs_work"
    assert captured["force"] is True
    assert resumed.milestone == "static_checks"
    assert "do not repeat completed research, grounding, or implementation" in resumed.next_action
    assert persisted is not None
    assert persisted.milestone == "static_checks"


def test_recovery_with_known_static_failure_resumes_directly_at_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    checkpoint = _checkpoint(tmp_path, static_status="failed")
    captured = {}

    def fake_resume(_self, validated, *, force):
        captured["checkpoint"] = validated[0]
        return EvolutionResult("candidate_needs_work", validated[0].branch, workspace, "captured", False)

    monkeypatch.setattr(BaseSelfDeveloper, "_resume_checkpoint_candidate", fake_resume)
    developer._resume_checkpoint_candidate(
        (checkpoint, SimpleNamespace(), _task(), workspace),
        force=True,
    )

    resumed = captured["checkpoint"]
    assert resumed.milestone == "local_static_repair"
    assert resumed.static_check_status == "failed"
    assert "do not repeat completed research, grounding, or implementation" in resumed.next_action
