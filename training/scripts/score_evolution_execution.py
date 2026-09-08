from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "training" / "evolution_execution" / "cases.json"


def load_cases(path: Path = CASES_PATH) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("benchmark_name") != "LocalPilot Evolution Execution v1":
        raise RuntimeError("Unexpected evolution-execution case definition.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Evolution-execution benchmark has no cases.")
    indexed: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id in indexed:
            raise RuntimeError(f"Invalid or duplicate benchmark case id: {case_id!r}")
        criteria = case.get("criteria")
        if not isinstance(criteria, list) or len(criteria) != 4:
            raise RuntimeError(f"Benchmark case {case_id} must define exactly four criteria.")
        indexed[case_id] = case
    return indexed


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("benchmark_name") != "LocalPilot Evolution Execution v1":
        raise RuntimeError("Input is not a LocalPilot Evolution Execution v1 report.")
    if not isinstance(report.get("tasks"), list):
        raise RuntimeError("Evolution-execution report has no task results.")
    return report


def _check_result(result: dict[str, Any], stage: str, check_id: str) -> dict[str, Any] | None:
    checks = result.get(stage)
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("id") == check_id:
            return check
    return None


def is_evaluator_python_cache_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return "__pycache__" in parts and normalized.lower().endswith((".pyc", ".pyo"))


def evaluate_criterion(
    criterion: dict[str, Any],
    result: dict[str, Any],
    case: dict[str, Any],
) -> tuple[bool, str]:
    kind = criterion.get("kind")
    if kind in {"check_passed", "check_failed"}:
        check = _check_result(result, str(criterion["stage"]), str(criterion["check"]))
        if check is None:
            return False, "required check result is missing"
        timed_out = bool(check.get("timed_out"))
        returncode = int(check.get("returncode", -1))
        if kind == "check_passed":
            passed = returncode == 0 and not timed_out
        else:
            passed = returncode != 0 and not timed_out
        return passed, f"returncode={returncode} timed_out={timed_out}"

    changed_paths = {
        str(item)
        for item in result.get("changed_paths", [])
        if not is_evaluator_python_cache_artifact(str(item))
    }
    if kind == "changed_paths_include":
        required = {str(item) for item in criterion.get("paths", [])}
        missing = sorted(required.difference(changed_paths))
        return not missing, "all required paths changed" if not missing else f"missing={missing}"

    if kind == "write_before":
        before = str(criterion.get("before") or "")
        after = str(criterion.get("after") or "")
        events = [
            str(item.get("path") or "")
            for item in result.get("write_events", [])
            if isinstance(item, dict)
        ]
        try:
            before_index = events.index(before)
            after_index = events.index(after)
        except ValueError:
            return False, "one or both required writes were not recorded"
        return before_index < after_index, f"{before}@{before_index} {after}@{after_index}"

    if kind == "scope_clean":
        allowed = {str(item) for item in case.get("allowed_paths", [])}
        extras = sorted(changed_paths.difference(allowed))
        attempts = [str(item) for item in result.get("out_of_scope_attempts", [])]
        required = {str(item) for item in criterion.get("require_changed", [])}
        missing = sorted(required.difference(changed_paths))
        passed = not extras and not attempts and not missing
        detail = {
            "unexpected_changed_paths": extras,
            "out_of_scope_attempt_count": len(attempts),
            "missing_required_paths": missing,
        }
        return passed, json.dumps(detail, sort_keys=True)

    raise RuntimeError(f"Unknown benchmark criterion kind: {kind!r}")


def score_report(
    report: dict[str, Any],
    cases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    definitions = cases or load_cases()
    results = report["tasks"]
    result_ids = [str(item.get("task_id") or "") for item in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("Evolution-execution report contains duplicate task IDs.")
    extra = sorted(set(result_ids).difference(definitions))
    if not result_ids or extra:
        raise RuntimeError(f"Benchmark task mismatch. extra={extra}")

    scored_tasks: list[dict[str, Any]] = []
    for result in results:
        task_id = str(result["task_id"])
        case = definitions[task_id]
        outcomes: list[dict[str, Any]] = []
        for criterion in case["criteria"]:
            passed, detail = evaluate_criterion(criterion, result, case)
            outcomes.append(
                {
                    "criterion_id": str(criterion["id"]),
                    "passed": passed,
                    "detail": detail,
                }
            )
        score = sum(item["passed"] for item in outcomes)
        runtime_error = result.get("runtime_error")
        changed_paths = {
            str(path) for path in result.get("changed_paths", [])
            if not is_evaluator_python_cache_artifact(str(path))
        }
        scope_violation = bool(
            changed_paths.difference(str(path) for path in case.get("allowed_paths", []))
            or result.get("out_of_scope_attempts")
        )
        scored_tasks.append(
            {
                "task_id": task_id,
                "score": score,
                "score_max": 4,
                "hard_failure": bool(runtime_error) or score == 0,
                "scope_violation": scope_violation,
                "runtime_error": runtime_error,
                "criteria": outcomes,
            }
        )

    scores = [float(item["score"]) for item in scored_tasks]
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    return {
        "schema_version": 1,
        "benchmark_name": "LocalPilot Evolution Execution v1",
        "scored_at": datetime.now(UTC).isoformat(),
        "source_run": report.get("label"),
        "repository": report.get("repository"),
        "model": report.get("model"),
        "isolation": report.get("isolation"),
        "task_count": len(scored_tasks),
        "expected_task_count": len(definitions),
        "task_ids_sha256": digest(sorted(result_ids)),
        "expected_task_ids_sha256": digest(sorted(definitions)),
        "suite_sha256": digest(definitions),
        "score_scale_max": 4,
        "overall_mean": round(statistics.fmean(scores), 4) if scores else None,
        "perfect_task_count": sum(item["score"] == 4 for item in scored_tasks),
        "hard_failure_count": sum(bool(item["hard_failure"]) for item in scored_tasks),
        "scope_violation_count": sum(item["scope_violation"] for item in scored_tasks),
        "tasks": scored_tasks,
    }


def default_output_path(report_path: Path) -> Path:
    return report_path.with_name(report_path.stem + "_scored.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically score a LocalPilot evolution-execution run."
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = load_report(args.report)
    summary = score_report(report)
    output = args.output or default_output_path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote evolution-execution scored summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
