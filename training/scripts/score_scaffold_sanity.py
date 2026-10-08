#!/usr/bin/env python3
"""Blind rubric scoring for the exploratory same-weights Nestra sanity test.

A paired eight-task pilot is descriptive; do not infer statistically
significant model improvement or regressions from this sample.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.scripts import run_scaffold_sanity as sanity
from training.scripts import score_eval_matrix as rubric


def load_run(path: Path) -> tuple[dict, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    tasks = sanity.selected_tasks()
    expected = {(task["id"], repeat, arm) for task in tasks for repeat in (1, 2, 3) for arm in sanity.ARMS}
    keys = [(c.get("task_id"), c.get("repeat"), c.get("arm")) for c in report.get("cells", [])]
    if (
        report.get("suite") != sanity.SUITE
        or report.get("task_ids") != [t["id"] for t in tasks]
        or report.get("task_digest") != sanity.matrix.digest(tasks)
        or report.get("repeats") != sanity.REPEATS
        or report.get("arms") != list(sanity.ARMS)
        or report.get("planned_cells") != len(expected)
        or report.get("completed_at") is None
        or len(keys) != len(set(keys))
        or set(keys) != expected
    ):
        raise RuntimeError("Sanity run is incomplete, changed or not a matched 48-cell report")
    return report, {t["id"]: t for t in tasks}


def summarize(report: dict, review: dict, tasks: dict) -> dict:
    grades = defaultdict(lambda: defaultdict(list))
    failures = Counter()
    origins = defaultdict(Counter)
    for cell in report["cells"]:
        card = review[rubric.review_id(report, cell)]
        arm = cell["arm"]
        task_id = cell["task_id"]
        grades[task_id][arm].append(card["score"])
        if card["hard_failure"]:
            failures[arm] += 1
        origins[arm][card["failure_origin"]] += 1
    task_deltas = []
    by_category = defaultdict(list)
    for task_id in report["task_ids"]:
        direct = grades[task_id]["nestra_direct"]
        scaffold = grades[task_id]["nestra_localpilot"]
        if len(direct) != 3 or len(scaffold) != 3:
            raise RuntimeError("Expected three scored runs in each condition")
        diff = statistics.fmean(scaffold) - statistics.fmean(direct)
        task_deltas.append(diff)
        by_category[tasks[task_id]["task_type"]].append(diff)
    return {
        "suite": report["suite"], "run_id": report["run_id"],
        "model": report["model"], "repository": report["repository"],
        "task_count": len(task_deltas), "graded_responses": len(report["cells"]),
        "direct_mean": round(statistics.fmean([
            score for task in report["task_ids"] for score in grades[task]["nestra_direct"]
        ]), 4),
        "scaffold_mean": round(statistics.fmean([
            score for task in report["task_ids"] for score in grades[task]["nestra_localpilot"]
        ]), 4),
        "scaffold_minus_direct_paired_delta": round(statistics.fmean(task_deltas), 4),
        "scaffold_task_wins": sum(v > 0 for v in task_deltas),
        "scaffold_task_losses": sum(v < 0 for v in task_deltas),
        "task_ties": sum(v == 0 for v in task_deltas),
        "category_deltas": {k: round(statistics.fmean(v), 4) for k, v in sorted(by_category.items())},
        "hard_failures": dict(failures), "failure_origins": {a: dict(origins[a]) for a in sanity.ARMS},
        "automatic_signals": sanity.automatic_notes(report)["by_arm"],
        "caveats": [
            "This eight-task sample is a sanity check, not a powered causal estimate.",
            "Identical model weights, different tool access and orchestration.",
            "Scaffold scores measure what the user receives; no non-model scores were silently excluded.",
            "Investigate execution/harness faults and audit intermediate drafts before attributing cause.",
        ],
        "automated_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path, metavar="BLIND_SCORECARDS")
    group.add_argument("--scorecard", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, tasks = load_run(args.report)
    reference_cards = rubric.prepare(report, tasks)
    if args.prepare:
        if args.prepare.exists():
            raise RuntimeError("Refusing to overwrite a blind review file")
        args.prepare.parent.mkdir(parents=True, exist_ok=True)
        args.prepare.write_text("".join(json.dumps(card, ensure_ascii=False) + "\n"
                                         for card in reference_cards), encoding="utf-8")
        print(f"Prepared {len(reference_cards)} arm-blind review cards: {args.prepare}")
        return 0
    scored = rubric.validate_cards(rubric.load_cards(args.scorecard), reference_cards)
    result = summarize(report, scored, tasks)
    if args.output:
        if args.output.exists():
            raise RuntimeError("Refusing to overwrite existing scored report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
