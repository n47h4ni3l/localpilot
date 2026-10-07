#!/usr/bin/env python3
"""Inspect Eval v1 hard failures from their original per-task evidence.

Aggregate scores cannot distinguish model defects from LocalPilot runner/tool
or review failures. This report NEVER infers model regression from a count.
The raw answer and scoring evidence remain on the local machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Expected a JSON report object")
    return data


def _scored_rows(report: dict) -> dict[str, dict]:
    rows = report.get("scores", [])
    if not isinstance(rows, list):
        raise RuntimeError("A scored Eval v1 summary with per-task rows is required")
    return {str(row["task_id"]): row for row in rows}


def build_triage(candidate_run: dict, candidate_scores: dict, baseline_run: dict | None = None, baseline_scores: dict | None = None) -> dict:
    responses = {str(row["task_id"]): row for row in candidate_run.get("tasks", [])}
    grades = _scored_rows(candidate_scores)
    if not responses or not grades or set(grades) != set(responses):
        raise RuntimeError("Raw report and per-task scorecard do not match in complete task coverage")
    baseline_responses = {str(row["task_id"]): row for row in (baseline_run or {}).get("tasks", [])}
    baseline_grades = _scored_rows(baseline_scores) if baseline_scores is not None else {}
    failures = []
    for task_id, grade in sorted(grades.items()):
        if not grade.get("hard_failure"):
            continue
        raw = responses[task_id]
        exception = raw.get("error")
        failures.append({
            "task_id": task_id,
            "category": raw.get("task_type"),
            "candidate_score": grade.get("score"),
            "candidate_rationale": grade.get("rationale"),
            "candidate_response": raw.get("response"),
            "candidate_runner_exception": exception,
            "baseline_score": baseline_grades.get(task_id, {}).get("score"),
            "baseline_hard_failure": baseline_grades.get(task_id, {}).get("hard_failure"),
            "baseline_response": baseline_responses.get(task_id, {}).get("response"),
            "baseline_runner_exception": baseline_responses.get(task_id, {}).get("error"),
            "triage_status": "unresolved_requires_raw_evidence_review",
            "investigate_first": "runner / environment" if exception else "model output, tool trace and reviewer rationale",
        })
    return {
        "schema_version": 1,
        "candidate_model": candidate_run.get("model"), "candidate_repository": candidate_run.get("repository"),
        "baseline_model": (baseline_run or {}).get("model"),
        "baseline_repository": (baseline_run or {}).get("repository"),
        "hard_failure_count": len(failures),
        "reported_hard_failure_count": candidate_scores.get("hard_failure_count"),
        "different_repository_revisions": (
            candidate_run.get("repository", {}).get("head") != baseline_run.get("repository", {}).get("head")
        ) if baseline_run else None,
        "failures": failures,
        "assessment": "Counts alone cannot attribute cause. A task with a runtime exception suggests investigating the harness, but no model or infrastructure diagnosis is established without traces and controlled reruns.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.baseline_run) != bool(args.baseline_scores):
        parser.error("Provide both baseline report and baseline scored summary, or neither")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing triage report")
    triage = build_triage(
        _read(args.candidate_run), _read(args.candidate_scores),
        _read(args.baseline_run) if args.baseline_run else None,
        _read(args.baseline_scores) if args.baseline_scores else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(triage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {triage['hard_failure_count']} unresolved per-task hard failures: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
