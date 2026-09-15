from __future__ import annotations

import json
from pathlib import Path

from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.selfdev import SelfDeveloper as BaseSelfDeveloper
from localpilot.selfdev_response_parsing import StaticRepairResult, _json_object


def _developer(tmp_path: Path) -> SelfDeveloper:
    config = Config()
    config.agent.data_dir = "data"
    return SelfDeveloper(config, tmp_path)


def _task() -> dict:
    return {
        "id": "long-context",
        "title": "Expand context handling",
        "acceptance": ["Preserve context across long interactions."],
        "evolution_class": "improve_cognition",
        "capability_target": "Long-form interaction",
        "mission_alignment": "Improve reliable long conversations.",
        "current_frontier": "Context can truncate during long work.",
        "why_high_leverage": "Long tasks depend on retained context.",
        "capability_unlocked": "Longer coherent interactions.",
        "next_frontier": "Adaptive context retention.",
        "question": "Can long interactions retain useful context?",
        "observed_limitation": "Long conversations can lose relevant context.",
        "evidence": ["A long-context benchmark exists."],
        "alternatives": ["Keep current behavior", "Improve context handling"],
        "hypothesis": "The candidate reduces long-context failures.",
        "evaluation": {
            "metric": "hallucination rate",
            "baseline": "current long-context benchmark result",
            "success_criterion": "hallucination rate <= 10%",
            "measurement_method": "Automated hallucination detection on benchmark dataset",
        },
        "expected_complexity": "medium",
        "source": "capability_discovery",
    }


def test_successful_static_repair_preserves_pending_evaluation_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)

    monkeypatch.setattr(
        BaseSelfDeveloper,
        "_repair_static_failures",
        lambda _self, *args, **kwargs: StaticRepairResult(
            "static_checks=passed",
            True,
            json.dumps(
                {
                    "summary": "Removed unsafe page navigation.",
                    "reusable_lesson": "Keep repaired frontend behavior inside the host boundary.",
                }
            ),
            1,
        ),
    )

    result = developer._repair_static_failures(
        task=_task(),
        cycle_id=7,
        branch="localpilot/candidate-long-context-test",
    )
    payload = _json_object(result.final_text)
    evidence = payload["evaluation_evidence"]

    assert result.passed is True
    assert evidence["result"] == "pending_ci"
    assert evidence["metric"] == "hallucination rate"
    assert evidence["baseline_evidence"] == "current long-context benchmark result"
    assert evidence["measurement_artifact"] == (
        "Automated hallucination detection on benchmark dataset"
    )
    assert "Static repair completed" in evidence["candidate_evidence"]


def test_static_repair_does_not_overwrite_explicit_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)
    original = {
        "summary": "Repair finished.",
        "reusable_lesson": "Keep measured evidence authoritative.",
        "evaluation_evidence": {
            "metric": "hallucination rate",
            "baseline_evidence": "10%",
            "candidate_evidence": "14%",
            "result": "regressed",
            "measurement_artifact": "benchmark.json",
        },
    }
    monkeypatch.setattr(
        BaseSelfDeveloper,
        "_repair_static_failures",
        lambda _self, *args, **kwargs: StaticRepairResult(
            "static_checks=passed",
            True,
            json.dumps(original),
            1,
        ),
    )

    result = developer._repair_static_failures(task=_task(), cycle_id=8, branch="candidate")
    payload = _json_object(result.final_text)

    assert payload["evaluation_evidence"]["result"] == "regressed"
    assert payload["evaluation_evidence"]["candidate_evidence"] == "14%"


def test_failed_static_repair_does_not_invent_pending_ci_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    developer = _developer(tmp_path)
    monkeypatch.setattr(
        BaseSelfDeveloper,
        "_repair_static_failures",
        lambda _self, *args, **kwargs: StaticRepairResult(
            "static_checks=failed",
            False,
            json.dumps({"summary": "Repair did not pass."}),
            1,
        ),
    )

    result = developer._repair_static_failures(task=_task(), cycle_id=9, branch="candidate")
    payload = _json_object(result.final_text)

    assert result.passed is False
    assert "evaluation_evidence" not in payload
