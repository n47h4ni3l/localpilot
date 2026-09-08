#!/usr/bin/env python3
"""Compare baseline and candidate model evidence using evaluation promotion gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import write_json

CRITICAL_CATEGORIES = frozenset({"repository_reasoning", "debugging", "tool_use", "epistemics"})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _number(value: Any, label: str, *, counter: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{label} must be a finite number")
    if value < 0 or (not counter and value > 4) or (counter and value != int(value)):
        raise RuntimeError(f"{label} must be {'a nonnegative integer' if counter else 'between 0 and 4'}")
    return int(value) if counter else float(value)


def _optional_count(value: Any, label: str) -> int | None:
    return None if value is None else int(_number(value, label, counter=True))


def _digest(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _metadata(summary: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    run = summary.get("run", {})
    if not isinstance(run, dict):
        raise RuntimeError("run must be an object")
    for key in ("scale_max", "score_scale_max"):
        if key in scores and _number(scores[key], key) != 4:
            raise RuntimeError("Only the 0–4 benchmark scale is supported")
    return {
        "task_count": _optional_count(summary.get("task_count", run.get("task_count")), "task_count"),
        "expected_task_count": _optional_count(summary.get("expected_task_count"), "expected_task_count"),
        **{key: _digest(summary.get(key), key) for key in ("task_ids_sha256", "expected_task_ids_sha256", "suite_sha256")},
        "model": summary.get("model"),
    }


def _validate_counters(result: dict[str, Any], names: Sequence[str]) -> None:
    count = result["task_count"]
    if count is not None:
        if count == 0:
            raise RuntimeError("A scored benchmark must contain at least one task")
        for name in names:
            if result[name] is not None and result[name] > count:
                raise RuntimeError(f"{name} exceeds task_count")


def normalize_eval(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("eval_name") != "LocalPilot Eval v1":
        raise RuntimeError("Expected a LocalPilot Eval v1 summary")
    scores = summary.get("scores") if isinstance(summary.get("scores"), dict) else summary
    critical = scores.get("critical_category_mean", scores.get("critical_overall_mean"))
    categories = scores.get("category_means", {})
    counts = scores.get("category_counts", {})
    if not isinstance(categories, dict) or not isinstance(counts, dict):
        raise RuntimeError("Eval category_means and category_counts must be objects")
    result = {
        **_metadata(summary, scores),
        "overall": _number(scores.get("overall_mean"), "Eval overall mean"),
        "critical": _number(critical, "Eval critical mean"),
        "hard_failures": _number(scores.get("hard_failure_count"), "Eval hard failures", counter=True),
        "categories": {str(key): _number(value, f"category {key}") for key, value in categories.items()},
        "category_counts": {str(key): _number(value, f"category count {key}", counter=True) for key, value in counts.items()},
    }
    if counts and (set(counts) != set(categories) or any(value == 0 for value in counts.values())):
        raise RuntimeError("Category counts must cover exactly the scored categories with positive counts")
    if counts and result["task_count"] is not None and sum(counts.values()) != result["task_count"]:
        raise RuntimeError("Category counts do not sum to task_count")
    _validate_counters(result, ["hard_failures"])
    return result


def normalize_execution(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("benchmark_name") != "LocalPilot Evolution Execution v1":
        raise RuntimeError("Expected a LocalPilot Evolution Execution v1 summary")
    scores = summary.get("scores") if isinstance(summary.get("scores"), dict) else summary
    result = {
        **_metadata(summary, scores),
        "overall": _number(scores.get("overall_mean"), "execution overall mean"),
        "perfect_tasks": _number(scores.get("perfect_task_count"), "execution perfect tasks", counter=True),
        "hard_failures": _number(scores.get("hard_failure_count"), "execution hard failures", counter=True),
        "scope_violations": _optional_count(scores.get("scope_violation_count"), "execution scope violations"),
    }
    _validate_counters(result, ["hard_failures", "perfect_tasks", "scope_violations"])
    if result["task_count"] is not None and result["hard_failures"] + result["perfect_tasks"] > result["task_count"]:
        raise RuntimeError("Hard failures and perfect tasks exceed task_count")
    return result


def _coverage(before: dict[str, Any], after: dict[str, Any]) -> bool | None:
    required = ("task_count", "expected_task_count", "task_ids_sha256", "expected_task_ids_sha256", "suite_sha256")
    if any(item[key] is None for item in (before, after) for key in required):
        return None
    return (
        before["task_count"] == before["expected_task_count"] == after["task_count"] == after["expected_task_count"]
        and before["task_ids_sha256"] == before["expected_task_ids_sha256"] == after["task_ids_sha256"] == after["expected_task_ids_sha256"]
        and before["suite_sha256"] == after["suite_sha256"]
    )


def _same_model(eval_result: dict[str, Any], execution_result: dict[str, Any]) -> bool | None:
    models = [eval_result["model"], execution_result["model"]]
    if any(not isinstance(model, dict) or not model.get("name") or not model.get("digest") for model in models):
        return None
    return all(models[0][key] == models[1][key] for key in ("name", "digest"))


def compare(
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    baseline_execution: dict[str, Any],
    candidate_execution: dict[str, Any],
    *,
    critical_tolerance: float = 0.0,
) -> dict[str, Any]:
    critical_tolerance = float(_number(critical_tolerance, "critical regression tolerance"))
    before_eval = normalize_eval(baseline_eval)
    after_eval = normalize_eval(candidate_eval)
    before_exec = normalize_execution(baseline_execution)
    after_exec = normalize_execution(candidate_execution)

    category_names = sorted(set(before_eval["categories"]) | set(after_eval["categories"]))
    category_deltas = {
        name: {
            "baseline": before_eval["categories"].get(name),
            "candidate": after_eval["categories"].get(name),
            "critical": name in CRITICAL_CATEGORIES,
            "delta": (
                round(after_eval["categories"][name] - before_eval["categories"][name], 4)
                if name in before_eval["categories"] and name in after_eval["categories"]
                else None
            ),
        }
        for name in category_names
    }
    categories_complete = (
        set(before_eval["categories"]) == set(after_eval["categories"])
        and CRITICAL_CATEGORIES.issubset(before_eval["categories"])
    )
    category_regression = all(
        after_eval["categories"][name] >= before_eval["categories"][name] - critical_tolerance
        for name in CRITICAL_CATEGORIES
        if name in before_eval["categories"] and name in after_eval["categories"]
    )
    category_gate = False if not category_regression else (True if categories_complete else None)
    category_coverage = (
        before_eval["category_counts"] == after_eval["category_counts"]
        if before_eval["category_counts"] and after_eval["category_counts"] else None
    )
    metric_gates = {
        "eval_overall_not_lower": after_eval["overall"] >= before_eval["overall"],
        "no_material_critical_regression": after_eval["critical"] >= before_eval["critical"] - critical_tolerance,
        "no_material_critical_category_regression": category_gate,
        "eval_hard_failures_not_higher": after_eval["hard_failures"] <= before_eval["hard_failures"],
        "execution_at_least_baseline": after_exec["overall"] >= before_exec["overall"],
        "execution_hard_failures_not_higher": after_exec["hard_failures"] <= before_exec["hard_failures"],
        "no_scope_violations": after_exec["scope_violations"] == 0 if after_exec["scope_violations"] is not None else None,
    }
    gates = {
        **metric_gates,
        "eval_complete_matching_coverage": _coverage(before_eval, after_eval),
        "execution_complete_matching_coverage": _coverage(before_exec, after_exec),
        "matching_category_coverage": category_coverage,
        "baseline_model_identity_matches_both_evaluations": _same_model(before_eval, before_exec),
        "candidate_model_identity_matches_both_evaluations": _same_model(after_eval, after_exec),
        "baseline_scope_evidence_available": True if before_exec["scope_violations"] is not None else None,
    }
    missing = [name for name, passed in gates.items() if passed is None]
    failed = [name for name, passed in gates.items() if passed is False]
    return {
        "schema_version": 1,
        "artifact_type": "model_promotion_comparison",
        "promotion_recommended": not missing and not failed,
        "human_promotion_required": True,
        "comparison_status": "blocked" if failed else ("incomplete_evidence" if missing else "eligible_for_human_review"),
        "observed_metric_gates_passed": all(value is not False for value in metric_gates.values()),
        "missing_evidence": missing,
        "failed_gates": failed,
        "preference": {"lower_hard_failures_preferred": True, "eval_hard_failures_reduced": after_eval["hard_failures"] < before_eval["hard_failures"]},
        "promotion_basis": "held-out Eval v1 plus Evolution Execution v1; training loss is not a promotion signal",
        "critical_regression_tolerance": critical_tolerance,
        "gates": gates,
        "eval_v1": {
            "baseline": before_eval,
            "candidate": after_eval,
            "delta": {
                "overall": round(after_eval["overall"] - before_eval["overall"], 4),
                "critical": round(after_eval["critical"] - before_eval["critical"], 4),
                "hard_failures": after_eval["hard_failures"] - before_eval["hard_failures"],
            },
            "category_deltas": category_deltas,
        },
        "evolution_execution": {
            "baseline": before_exec,
            "candidate": after_exec,
            "delta": {
                "overall": round(after_exec["overall"] - before_exec["overall"], 4),
                "perfect_tasks": after_exec["perfect_tasks"] - before_exec["perfect_tasks"],
                "hard_failures": after_exec["hard_failures"] - before_exec["hard_failures"],
                "scope_violations": after_exec["scope_violations"] - before_exec["scope_violations"]
                if after_exec["scope_violations"] is not None and before_exec["scope_violations"] is not None else None,
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply LocalPilot model promotion gates.")
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--baseline-execution", type=Path, required=True)
    parser.add_argument("--candidate-execution", type=Path, required=True)
    parser.add_argument("--critical-tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare(
            _load(args.baseline_eval), _load(args.candidate_eval),
            _load(args.baseline_execution), _load(args.candidate_execution),
            critical_tolerance=args.critical_tolerance,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"promotion_recommended": False, "error": str(exc)}))
        return 2
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["promotion_recommended"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
