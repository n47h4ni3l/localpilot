from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "training" / "evals"
CRITICAL_CATEGORIES = frozenset({"repository_reasoning", "debugging", "tool_use", "epistemics"})


def load_rubrics(eval_root: Path = EVAL_ROOT) -> dict[str, dict[str, Any]]:
    rubrics: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_root.rglob("*.jsonl")):
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("split") != "held_out_eval":
                continue
            task_id = str(record.get("id") or "")
            if not task_id:
                raise RuntimeError(f"Missing task id in {path}:{line_no}")
            if task_id in rubrics:
                raise RuntimeError(f"Duplicate held-out task id: {task_id}")
            expected = record.get("expected_behavior")
            if isinstance(expected, str):
                expected = [expected]
            if not isinstance(expected, list) or not expected:
                raise RuntimeError(f"Held-out task {task_id} has no expected_behavior rubric.")
            users = [
                message
                for message in record.get("messages", [])
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if len(users) != 1:
                raise RuntimeError(f"Held-out task {task_id} must have exactly one user message.")
            rubrics[task_id] = {
                "task_id": task_id,
                "task_type": record["task_type"],
                "difficulty": record.get("difficulty"),
                "prompt": users[0]["content"],
                "expected_behavior": [str(item) for item in expected],
            }
    return rubrics


def load_run(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("eval_name") != "LocalPilot Eval v1":
        raise RuntimeError("Input is not a LocalPilot Eval v1 response report.")
    if not isinstance(value.get("tasks"), list):
        raise RuntimeError("Eval response report has no task list.")
    return value


def prepare_scorecard(run: dict[str, Any], rubrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in run["tasks"]:
        task_id = str(result.get("task_id") or "")
        if task_id in seen:
            raise RuntimeError(f"Duplicate task result in run report: {task_id}")
        seen.add(task_id)
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise RuntimeError(f"Run report contains unknown held-out task: {task_id}")
        runtime_error = result.get("error")
        cards.append(
            {
                **rubric,
                "response": str(result.get("response") or ""),
                "runtime_error": runtime_error,
                "score": 0 if runtime_error else None,
                "hard_failure": bool(runtime_error),
                "rationale": (
                    "Runtime error prevented an answer."
                    if runtime_error
                    else ""
                ),
                "reviewer": "",
            }
        )
    return cards


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_scorecard(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid scorecard JSON at line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"Scorecard line {line_no} is not an object.")
        rows.append(row)
    return rows


def validate_scorecard(
    cards: list[dict[str, Any]],
    run: dict[str, Any],
    rubrics: dict[str, dict[str, Any]],
) -> None:
    run_ids = [str(item.get("task_id") or "") for item in run["tasks"]]
    card_ids = [str(item.get("task_id") or "") for item in cards]
    if len(card_ids) != len(set(card_ids)):
        raise RuntimeError("Scorecard contains duplicate task IDs.")
    if set(card_ids) != set(run_ids):
        missing = sorted(set(run_ids).difference(card_ids))
        extra = sorted(set(card_ids).difference(run_ids))
        raise RuntimeError(f"Scorecard/run task mismatch. missing={missing} extra={extra}")
    for card in cards:
        task_id = str(card["task_id"])
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise RuntimeError(f"Unknown scorecard task: {task_id}")
        if str(card.get("task_type") or "") != rubric["task_type"]:
            raise RuntimeError(f"Task type changed in scorecard: {task_id}")
        score = card.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError(f"Task {task_id} needs a numeric score from 0 to 4.")
        if not math.isfinite(score) or float(score) < 0 or float(score) > 4:
            raise RuntimeError(f"Task {task_id} score must be between 0 and 4.")
        rationale = str(card.get("rationale") or "").strip()
        if not rationale:
            raise RuntimeError(f"Task {task_id} needs a short scoring rationale.")
        if not isinstance(card.get("hard_failure"), bool):
            raise RuntimeError(f"Task {task_id} hard_failure must be true or false.")


def summarize(
    cards: list[dict[str, Any]],
    run: dict[str, Any],
    rubrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_category: dict[str, list[float]] = {}
    for card in cards:
        category = str(card["task_type"])
        by_category.setdefault(category, []).append(float(card["score"]))
    category_means = {
        category: round(statistics.fmean(scores), 4)
        for category, scores in sorted(by_category.items())
    }
    all_scores = [float(card["score"]) for card in cards]
    critical_scores = [
        float(card["score"])
        for card in cards
        if str(card["task_type"]) in CRITICAL_CATEGORIES
    ]
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    return {
        "schema_version": 1,
        "eval_name": "LocalPilot Eval v1",
        "label": run.get("label"),
        "scored_at": datetime.now(UTC).isoformat(),
        "repository": run.get("repository"),
        "model": run.get("model"),
        "isolation": run.get("isolation"),
        "task_count": len(cards),
        "expected_task_count": len(rubrics) if rubrics is not None else None,
        "task_ids_sha256": digest(sorted(str(card["task_id"]) for card in cards)),
        "expected_task_ids_sha256": digest(sorted(rubrics)) if rubrics is not None else None,
        "suite_sha256": digest(rubrics) if rubrics is not None else None,
        "overall_mean": round(statistics.fmean(all_scores), 4) if all_scores else None,
        "category_means": category_means,
        "category_counts": {category: len(scores) for category, scores in sorted(by_category.items())},
        "critical_category_means": {
            category: mean
            for category, mean in category_means.items()
            if category in CRITICAL_CATEGORIES
        },
        "critical_overall_mean": (
            round(statistics.fmean(critical_scores), 4) if critical_scores else None
        ),
        "hard_failure_count": sum(bool(card["hard_failure"]) for card in cards),
        "zero_score_count": sum(float(card["score"]) == 0 for card in cards),
        "scores": [
            {
                "task_id": card["task_id"],
                "task_type": card["task_type"],
                "score": float(card["score"]),
                "hard_failure": bool(card["hard_failure"]),
                "rationale": str(card["rationale"]),
                "reviewer": str(card.get("reviewer") or ""),
            }
            for card in cards
        ],
    }


def default_scorecard_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem + "_scorecard.jsonl")


def default_summary_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem + "_scored.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or aggregate a human/independent-review scorecard for LocalPilot Eval v1."
    )
    parser.add_argument("run", type=Path, help="Eval v1 response report from run_eval_v1.py")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--prepare",
        nargs="?",
        const="AUTO",
        metavar="PATH",
        help="Create a review scorecard with prompts, rubrics, responses, and blank scores.",
    )
    actions.add_argument(
        "--scorecard",
        type=Path,
        help="Validate a completed scorecard and produce aggregate scores.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = load_run(args.run)
    rubrics = load_rubrics()
    if args.prepare is not None:
        output = (
            default_scorecard_path(args.run)
            if args.prepare == "AUTO"
            else Path(args.prepare)
        )
        cards = prepare_scorecard(run, rubrics)
        write_jsonl(output, cards)
        print(f"Wrote Eval v1 review scorecard: {output}")
        return 0

    assert args.scorecard is not None
    cards = load_scorecard(args.scorecard)
    validate_scorecard(cards, run, rubrics)
    summary = summarize(cards, run, rubrics)
    output = args.output or default_summary_path(args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote Eval v1 scored summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
