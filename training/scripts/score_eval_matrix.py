#!/usr/bin/env python3
"""Blind review and paired-task uncertainty intervals for three-arm Nestra tests.

Model responses are scored only after inference, never by the evaluated model.
Bootstrap resamples TASKS, not the three correlated repetitions of each task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from training.scripts import run_eval_matrix as runner

ORIGINS = frozenset({"none", "model", "tool", "harness", "environment", "uncertain"})
NONMODEL = frozenset({"tool", "harness", "environment", "uncertain"})
BOOTSTRAPS = 5000
SCORING_SCALE = [
    "4 = fully satisfies the applicable expected behaviour with no material unsupported claim.",
    "3 = substantially correct; only minor omission or imprecision that does not change the outcome.",
    "2 = partially correct; contains useful reasoning but a material omission or error remains.",
    "1 = mostly incorrect or unusable; misses a core expected behaviour.",
    "0 = no usable answer, directly contradicts a core requirement, or materially fabricates evidence.",
    "hard_failure = true only for a fundamental core-task failure; a score below 4 is not automatically hard.",
]


def load_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in runner.load_tasks()}
    selected = [tasks[task_id] for task_id in report["task_ids"]]
    if report.get("suite") != "Nestra Paired Evaluation v2" or report.get("task_digest") != runner.digest(selected):
        raise RuntimeError("Report is not a valid immutable paired-eval task selection")
    if report.get("repeats") != 3 or len(report["cells"]) != report["planned_cells"] or report["planned_cells"] != 9 * len(selected):
        raise RuntimeError("Report is incomplete: finish all three repeats of all three arms first")
    keys = [(c["task_id"], c["repeat"], c["arm"]) for c in report["cells"]]
    expected = {(t["id"], repetition, arm) for t in selected for repetition in (1, 2, 3) for arm in runner.ARMS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Duplicate, missing or unexpected report cells")
    if not report.get("completed_at"):
        raise RuntimeError("Report completion marker is missing")
    return report, tasks


def review_id(report: dict[str, Any], cell: dict[str, Any]) -> str:
    data = f"{report['run_id']}|{cell['task_id']}|{cell['repeat']}|{cell['arm']}"
    return "case-" + hashlib.sha256(data.encode("utf-8")).hexdigest()[:20]


def prepare(report: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for cell in report["cells"]:
        task = tasks[cell["task_id"]]
        prompt, _ = runner.original._task_prompt(task)
        cards.append({
            "review_id": review_id(report, cell),
            "task_id": cell["task_id"], "task_type": task["task_type"],
            "difficulty": task.get("difficulty"),
            "prompt": prompt, "expected_behavior": task["expected_behavior"],
            "scoring_scale": SCORING_SCALE,
            "response": cell["response"],
            "execution_error": cell["error"],
            "score": None, "hard_failure": None, "failure_origin": None,
            "rationale": "", "reviewer": "",
        })
    # Arm identity and repetition are deliberately absent from independent review.
    return sorted(cards, key=lambda card: hashlib.sha256((report["run_id"] + card["review_id"] + "shuffle").encode()).hexdigest())


def load_cards(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_cards(cards: list[dict[str, Any]], expected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    originals = {row["review_id"]: row for row in expected}
    ids = [c.get("review_id") for c in cards]
    if len(ids) != len(set(ids)) or set(ids) != set(originals):
        raise RuntimeError("Review card IDs are missing, duplicated or extra")
    immutable_fields = ("task_id", "task_type", "difficulty", "prompt", "expected_behavior", "scoring_scale", "response", "execution_error")
    for row in cards:
        original = originals[row["review_id"]]
        for field in immutable_fields:
            if row.get(field) != original.get(field):
                raise RuntimeError(f"Protected review evidence changed: {row['review_id']} {field}")
        value = row.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 <= value <= 4):
            raise RuntimeError(f"Score missing or invalid: {row['review_id']}")
        if not isinstance(row.get("hard_failure"), bool):
            raise RuntimeError(f"Hard failure missing: {row['review_id']}")
        if row.get("failure_origin") not in ORIGINS:
            raise RuntimeError(f"Failure origin missing/invalid: {row['review_id']}")
        if row["hard_failure"] and row["failure_origin"] == "none":
            raise RuntimeError(f"Hard failure requires an attributable or uncertain origin: {row['review_id']}")
        if row.get("execution_error") and row["failure_origin"] in {"none", "model"}:
            raise RuntimeError(f"Execution error cannot be silently classed as model performance: {row['review_id']}")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise RuntimeError(f"Independent scoring rationale required: {row['review_id']}")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise RuntimeError(f"Reviewer identifier required: {row['review_id']}")
    return {card["review_id"]: card for card in cards}


def _ci(task_deltas: list[float], seed: int) -> dict[str, Any]:
    if not task_deltas:
        return {"mean_delta": None, "ci95": None, "tasks": 0, "direction": "insufficient"}
    mean = statistics.fmean(task_deltas)
    rng = random.Random(seed)
    count = len(task_deltas)
    draws = sorted(statistics.fmean(rng.choices(task_deltas, k=count)) for _ in range(BOOTSTRAPS))
    low, high = draws[int(0.025 * BOOTSTRAPS)], draws[int(0.975 * BOOTSTRAPS)]
    direction = "positive" if low > 0 else ("negative" if high < 0 else "inconclusive")
    return {"mean_delta": round(mean, 4), "ci95": [round(low, 4), round(high, 4)],
            "tasks": count, "direction": direction}


def summarize(report: dict[str, Any], reviewed: dict[str, dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model_failures: Counter[str] = Counter()
    failure_origins: dict[str, Counter[str]] = defaultdict(Counter)
    excluded: dict[str, set[str]] = defaultdict(set)
    excluded_arms: dict[str, set[str]] = defaultdict(set)
    repetitions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cell in report["cells"]:
        card = reviewed[review_id(report, cell)]
        task_id, arm, repetition = cell["task_id"], cell["arm"], cell["repeat"]
        repetitions[(task_id, arm, repetition)] = card
        failure_origins[arm][card["failure_origin"]] += 1
        if card["failure_origin"] in NONMODEL:
            excluded[task_id].add(f"{arm} repeat {repetition}: {card['failure_origin']}")
            excluded_arms[task_id].add(arm)
        else:
            scores[task_id][arm].append(float(card["score"]))
            if card["hard_failure"]:
                model_failures[arm] += 1
    def evaluable_for(required_arms: tuple[str, ...]) -> list[str]:
        return [
            task_id for task_id in report["task_ids"]
            if all(
                arm not in excluded_arms[task_id] and len(scores[task_id][arm]) == 3
                for arm in required_arms
            )
        ]

    primary_evaluable = evaluable_for(("base_localpilot", "nestra_localpilot"))
    secondary_evaluable = evaluable_for(("base_direct", "base_localpilot"))
    all_three_evaluable = evaluable_for(runner.ARMS)

    def contrast(candidate: str, reference: str, included: list[str], seed: int) -> dict[str, Any]:
        deltas = [statistics.fmean(scores[tid][candidate]) - statistics.fmean(scores[tid][reference]) for tid in included]
        result = _ci(deltas, seed)
        result["candidate_mean"] = round(statistics.fmean(
            [statistics.fmean(scores[tid][candidate]) for tid in included]), 4) if included else None
        result["reference_mean"] = round(statistics.fmean(
            [statistics.fmean(scores[tid][reference]) for tid in included]), 4) if included else None
        result["task_wins"] = sum(delta > 0 for delta in deltas)
        result["task_losses"] = sum(delta < 0 for delta in deltas)
        result["task_ties"] = sum(delta == 0 for delta in deltas)
        return result

    by_category = {}
    for category in sorted({tasks[tid]["task_type"] for tid in report["task_ids"]}):
        included = [tid for tid in primary_evaluable if tasks[tid]["task_type"] == category]
        by_category[category] = contrast("nestra_localpilot", "base_localpilot", included, 4200 + len(by_category))
    return {
        "schema_version": 1, "suite": report["suite"], "run_id": report["run_id"],
        "task_digest": report["task_digest"], "repository": report["repository"],
        "models": report["models"], "inference": report["inference"],
        "planned_tasks": len(report["task_ids"]), "triplicate_inference_calls": len(report["cells"]),
        "evaluable_primary_tasks": len(primary_evaluable),
        "evaluable_secondary_tasks": len(secondary_evaluable),
        "evaluable_all_three_tasks": len(all_three_evaluable),
        "excluded_tasks": {tid: sorted(why) for tid, why in sorted(excluded.items())},
        "origin_counts": {arm: dict(failure_origins[arm]) for arm in runner.ARMS},
        "model_hard_failures": dict(model_failures),
        "primary_weight_effect": contrast("nestra_localpilot", "base_localpilot", primary_evaluable, 10207),
        "secondary_scaffold_effect": contrast("base_localpilot", "base_direct", secondary_evaluable, 10208),
        "by_category_weight_effect": by_category,
        "interpretation": "Paired task-level percentile bootstrap with 5000 resamples. CI does not prove causality; independent rubric audit and failure triage required. Each contrast excludes only tasks with non-model/uncertain failures in an arm used by that contrast, so the optional direct arm cannot invalidate the primary weight comparison.",
        "automated_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", type=Path, metavar="SCORECARD_JSONL")
    modes.add_argument("--scorecard", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, tasks = load_report(args.report)
    prepared = prepare(report, tasks)
    if args.prepare:
        if args.prepare.exists():
            raise RuntimeError("Refusing to overwrite an existing review scorecard")
        args.prepare.parent.mkdir(parents=True, exist_ok=True)
        args.prepare.write_text("".join(json.dumps(card, ensure_ascii=False) + "\n" for card in prepared), encoding="utf-8")
        print(f"Prepared {len(prepared)} blinded cases: {args.prepare}")
        return 0
    scored = validate_cards(load_cards(args.scorecard), prepared)
    summary = summarize(report, scored, tasks)
    if args.output:
        if args.output.exists():
            raise RuntimeError("Refusing to overwrite an existing score summary")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
