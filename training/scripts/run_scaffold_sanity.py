#!/usr/bin/env python3
"""Small, non-promotional P1 direct-vs-LocalPilot sanity experiment.

One frozen Nestra weight digest; eight selected held-out cases; three independent
repetitions per condition. Results are separate from the 684-cell P1 evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.scripts import run_eval_matrix as matrix

ARMS = ("nestra_direct", "nestra_localpilot")
TASK_IDS = (
    "lp-paired-repo-001",
    "lp-paired-debug-003",
    "lp-paired-debug-008",
    "lp-paired-tool-001",
    "lp-paired-tool-003",
    "lp-paired-evolution-002",
    "lp-paired-epistemics-002",
    "lp-paired-general-001",
)
REPEATS = 3
SUITE = "Nestra Scaffold Sanity v1"


def selected_tasks() -> list[dict]:
    all_tasks = {task["id"]: task for task in matrix.load_tasks()}
    if len(set(TASK_IDS)) != len(TASK_IDS) or set(TASK_IDS) - set(all_tasks):
        raise RuntimeError("Sanity task selection must be unique and exist in held-out evaluation")
    return [all_tasks[task_id] for task_id in TASK_IDS]


def plan(tasks: list[dict]) -> list[tuple[str, int, str]]:
    # Alternate which condition runs first, across tasks and repetitions.
    return [
        (task["id"], rep, arm)
        for task in sorted(tasks, key=lambda t: matrix.digest(t["id"]))
        for rep in range(REPEATS)
        for arm in (ARMS if (int(matrix.digest(task["id"])[:2], 16) + rep) % 2 == 0 else ARMS[::-1])
    ]


def identity(model_name: str) -> dict:
    model = matrix.original.ollama_model_identity(model_name)
    if not model.get("digest"):
        raise RuntimeError(f"No immutable Ollama digest for {model_name}")
    return model


def setup_report(tasks: list[dict], state: dict, config: object) -> dict:
    return {
        "schema_version": 1, "suite": SUITE,
        "run_id": uuid.uuid4().hex, "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None, "repository": state, "model": identity(config.model.name),
        "inference": {
            "think": config.model.think,
            "temperature": config.model.temperature,
            "context_tokens": config.model.context_tokens,
        },
        "task_ids": [task["id"] for task in tasks], "task_digest": matrix.digest(tasks),
        "arms": list(ARMS), "repeats": REPEATS, "planned_cells": len(tasks) * REPEATS * len(ARMS),
        "isolation": {
            "fresh_agent_each_cell": True, "library_disabled": True,
            "external_web_disabled": True, "training_tree_removed": True,
            "scaffold_tools": sorted(matrix.original.ALLOWED_EVAL_TOOLS),
            "direct_tools": [], "source_snapshot": "git archive HEAD",
        },
        "note": "Secondary, exploratory same-weights system-quality comparison; not formal model-weight evaluation.",
        "cells": [],
    }


def validate_resume(report: dict, tasks: list[dict], state: dict, config: object) -> None:
    if report.get("suite") != SUITE or report.get("arms") != list(ARMS) or report.get("repeats") != REPEATS:
        raise RuntimeError("Not a compatible scaffold-sanity report")
    if report.get("task_digest") != matrix.digest(tasks) or report.get("repository", {}).get("head") != state["head"]:
        raise RuntimeError("Task set or repository changed; cannot resume")
    if report.get("model", {}).get("digest") != identity(config.model.name)["digest"]:
        raise RuntimeError("Model weights changed; cannot resume")
    expected_settings = {"think": config.model.think, "temperature": config.model.temperature,
                         "context_tokens": config.model.context_tokens}
    if report.get("inference") != expected_settings or report.get("planned_cells") != len(plan(tasks)):
        raise RuntimeError("Inference settings or workload changed; cannot resume")
    cells = report.get("cells", [])
    keys = [(c["task_id"], c["repeat"] - 1, c["arm"]) for c in cells]
    if len(set(keys)) != len(keys) or not set(keys).issubset(set(plan(tasks))):
        raise RuntimeError("Duplicated/unknown result cells; cannot resume")


def automatic_notes(report: dict) -> dict:
    by_arm = {}
    for arm in ARMS:
        cells = [c for c in report["cells"] if c["arm"] == arm]
        by_arm[arm] = {
            "completed": len(cells),
            "execution_errors": sum(bool(c.get("error")) for c in cells),
            "localpilot_withheld_markers": sum(
                "[LocalPilot" in str(c.get("response", "")) for c in cells
            ),
            "total_duration_seconds": round(sum(c.get("duration_seconds", 0) for c in cells), 2),
        }
    return {
        "completed_cells": len(report["cells"]), "planned_cells": report["planned_cells"],
        "by_arm": by_arm,
        "warning": "Automated signals do not grade truth, usefulness or safety. Blind rubric review is required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nestra:20b-p1")
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "reports" / "nestra_scaffold_sanity.json")
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    tasks = selected_tasks()
    if args.plan_only:
        print(json.dumps({"model": args.model, "tasks": len(tasks),
                          "arms": list(ARMS), "repeats": REPEATS,
                          "model_calls": len(plan(tasks)), "task_ids": list(TASK_IDS)}, indent=2))
        return 0

    state = matrix.original.repository_state(ROOT)
    matrix.require_evaluator_checkout(state, None)
    config = matrix.original.isolated_config(ROOT, model=args.model)
    output = args.output
    if args.resume:
        if not output.is_file():
            raise RuntimeError("No prior report to resume")
        report = json.loads(output.read_text(encoding="utf-8"))
        validate_resume(report, tasks, state, config)
    else:
        if output.exists():
            raise RuntimeError("Refusing to overwrite existing sanity report")
        report = setup_report(tasks, state, config)
        matrix.original.write_report(output, report)
    completed = {(c["task_id"], c["repeat"] - 1, c["arm"]) for c in report["cells"]}
    remaining = [key for key in plan(tasks) if key not in completed]
    if args.max_new_cells is not None:
        remaining = remaining[:args.max_new_cells]
    lookup = {t["id"]: t for t in tasks}
    errors = 0
    with tempfile.TemporaryDirectory(prefix="nestra-scaffold-sanity-") as temp:
        snapshot = Path(temp) / "repo"
        matrix.original.build_isolated_snapshot(ROOT, snapshot)
        for index, (task_id, repetition, arm) in enumerate(remaining, 1):
            print(f"[{index}/{len(remaining)}] {task_id} rep={repetition+1} {arm}", flush=True)
            internal_arm = "base_direct" if arm == "nestra_direct" else "base_localpilot"
            cell = matrix._run_cell(lookup[task_id], repetition, internal_arm, config, config, snapshot)
            cell["arm"] = arm
            gc.collect()
            report["cells"].append(cell)
            matrix.original.write_report(output, report)
            if cell["error"]:
                errors += 1
            if errors >= 3:
                print("Three runtime faults: stop and investigate before consuming more GPU time.", flush=True)
                break
    if len(report["cells"]) == report["planned_cells"]:
        report["completed_at"] = datetime.now(UTC).isoformat()
    matrix.original.write_report(output, report)
    print(json.dumps(automatic_notes(report), indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
