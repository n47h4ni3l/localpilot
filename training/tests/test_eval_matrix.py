from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from training.scripts import run_eval_matrix as matrix
from training.scripts import score_eval_matrix as score


def _cfg(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name=name, think="high", temperature=0.1, context_tokens=8192),
        agent=types.SimpleNamespace(data_dir="example"),
    )


class TriplicateRunnerTests(unittest.TestCase):
    def test_original_eval_v1_is_untouched_and_new_suite_has_76_distinct_tasks(self) -> None:
        self.assertEqual(len(matrix.original.load_eval_tasks()), 25)
        self.assertEqual(len(matrix.original.load_eval_tasks(matrix.EXTRA_ROOT)), 51)
        tasks = matrix.load_tasks()
        self.assertEqual(len(tasks), 76)
        self.assertEqual(len({t["id"] for t in tasks}), 76)
        self.assertEqual({t["task_type"] for t in tasks}, {
            "repository_reasoning", "debugging", "tool_use", "research",
            "evolution", "epistemics", "generalization",
        })
        self.assertTrue(all(
            "assistant" not in [m.get("role") for m in t["messages"]]
            and t["split"] == "held_out_eval"
            and t["expected_behavior"]
            for t in tasks
        ))

    def test_exactly_684_matched_cells_and_balanced_slot_order(self) -> None:
        tasks = matrix.load_tasks()
        order = matrix.plan(tasks)
        self.assertEqual(len(order), 684)
        self.assertEqual(len(set(order)), 684)
        for task in tasks:
            values = [item for item in order if item[0] == task["id"]]
            self.assertEqual(len(values), 9)
            for repeat in range(3):
                self.assertEqual({arm for tid, r, arm in values if r == repeat}, set(matrix.ARMS))
                self.assertEqual(
                    [arm for tid, r, arm in values if r == repeat], list(matrix.ROTATIONS[repeat])
                )
        for slot in range(3):
            self.assertEqual(
                {matrix.ROTATIONS[repeat][slot] for repeat in range(3)}, set(matrix.ARMS)
            )

    def test_response_text_handles_ollama_mapping_and_object(self) -> None:
        self.assertEqual(matrix._response_text({"message": {"content": "hello"}}), "hello")
        self.assertEqual(matrix._response_text(types.SimpleNamespace(
            message=types.SimpleNamespace(content="world"))), "world")
        with self.assertRaisesRegex(RuntimeError, "no message"):
            matrix._response_text({"unexpected": "value"})

    def test_model_exception_is_preserved_as_runtime_error_not_model_score(self) -> None:
        task = matrix.load_tasks()[0]
        with patch.object(matrix, "_direct", side_effect=RuntimeError("simulated inference failure")):
            result = matrix._run_cell(task, 0, "base_direct", _cfg("base"), _cfg("candidate"), Path("."))
        self.assertEqual(result["arm"], "base_direct")
        self.assertEqual(result["repeat"], 1)
        self.assertEqual(result["error"]["type"], "RuntimeError")
        self.assertEqual(result["response"], "")

    def test_distinct_digest_identity_and_matched_config_enforced(self) -> None:
        def identity(name: str) -> dict:
            return {"name": name, "digest": name, "size_bytes": 1234}
        tasks = matrix.load_tasks()[:2]
        with patch.object(matrix.original, "ollama_model_identity", side_effect=identity):
            report = matrix._initial_report(tasks, {"head": "head"}, _cfg("base"), _cfg("p1"))
            self.assertEqual(report["planned_cells"], 18)
            self.assertEqual(report["task_digest"], matrix.digest(tasks))
            with self.assertRaisesRegex(RuntimeError, "same model digest"):
                matrix._initial_report(tasks, {"head": "head"}, _cfg("base"), _cfg("base"))
            altered = _cfg("p1")
            altered.model.temperature = 0.9
            with self.assertRaisesRegex(RuntimeError, "identical reasoning"):
                matrix._initial_report(tasks, {"head": "head"}, _cfg("base"), altered)

    def test_resume_fails_closed_on_task_changes_revisions_or_model_changes(self) -> None:
        tasks = matrix.load_tasks()[:2]
        def identity(name: str) -> dict:
            return {"name": name, "digest": name, "size_bytes": 1234}
        with patch.object(matrix.original, "ollama_model_identity", side_effect=identity):
            original = matrix._initial_report(tasks, {"head": "sha"}, _cfg("base"), _cfg("p1"))
            matrix.validate_resume(original, tasks, {"head": "sha"}, _cfg("base"), _cfg("p1"))
            with self.assertRaisesRegex(RuntimeError, "task or repository changes"):
                matrix.validate_resume(original, tasks, {"head": "other"}, _cfg("base"), _cfg("p1"))
            with self.assertRaisesRegex(RuntimeError, "task or repository changes"):
                matrix.validate_resume(original, tasks[:1], {"head": "sha"}, _cfg("base"), _cfg("p1"))
            with self.assertRaisesRegex(RuntimeError, "different model digests"):
                matrix.validate_resume(original, tasks, {"head": "sha"}, _cfg("base"), _cfg("p2"))
            invalid = copy.deepcopy(original)
            invalid["cells"] = [
                {"task_id": tasks[0]["id"], "repeat": 1, "arm": "base_direct"},
                {"task_id": tasks[0]["id"], "repeat": 1, "arm": "base_direct"},
            ]
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                matrix.validate_resume(invalid, tasks, {"head": "sha"}, _cfg("base"), _cfg("p1"))


def _synthetic_run(tasks: list[dict], *, baseline: float = 1.0, p1: float = 3.0) -> dict:
    cells = []
    for task in tasks:
        for repetition in (1, 2, 3):
            for arm in matrix.ARMS:
                value = baseline if arm == "base_localpilot" else (p1 if arm == "nestra_localpilot" else 0.0)
                cells.append({
                    "task_id": task["id"], "task_type": task["task_type"],
                    "repeat": repetition, "arm": arm,
                    "response": f"Response {arm} score target {value}", "error": None,
                    "duration_seconds": 1.0,
                })
    return {"run_id": "opaque-secret", "suite": "Nestra Paired Evaluation v2",
            "task_digest": matrix.digest(tasks), "task_ids": [task["id"] for task in tasks],
            "repeats": 3, "planned_cells": len(cells), "completed_at": "done",
            "models": {"base": {"digest": "a"}, "candidate": {"digest": "b"}},
            "repository": {"head": "fixed"}, "inference": {}, "cells": cells}


class TriplicateScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = matrix.load_tasks()[:3]
        self.report = _synthetic_run(self.tasks)
        self.lookup = {t["id"]: t for t in self.tasks}
        self.prepared = score.prepare(self.report, self.lookup)

    def _graded(self) -> list[dict]:
        graded = copy.deepcopy(self.prepared)
        for row in graded:
            row["score"] = 3.0 if "score target 3" in row["response"] else (
                1.0 if "score target 1" in row["response"] else 0.0)
            row["hard_failure"] = False
            row["failure_origin"] = "none"
            row["rationale"] = "Compared against supplied three-point rubric."
            row["reviewer"] = "independent-human"
        return graded

    def test_review_is_blind_to_arm_and_repeat_and_complete(self) -> None:
        self.assertEqual(len(self.prepared), 27)
        self.assertEqual(len({c["review_id"] for c in self.prepared}), 27)
        self.assertTrue(all("arm" not in c and "repeat" not in c for c in self.prepared))
        self.assertTrue(all(c["score"] is None for c in self.prepared))
        self.assertTrue(all(c["scoring_scale"] == score.SCORING_SCALE for c in self.prepared))

    def test_reject_modified_evidence_or_invalid_scores(self) -> None:
        cards = self._graded()
        self.assertEqual(len(score.validate_cards(cards, self.prepared)), 27)
        corrupted = copy.deepcopy(cards)
        corrupted[0]["response"] = "modified by reviewer"
        with self.assertRaisesRegex(RuntimeError, "Protected review evidence changed"):
            score.validate_cards(corrupted, self.prepared)
        corrupted = copy.deepcopy(cards)
        corrupted[0]["score"] = 5
        with self.assertRaisesRegex(RuntimeError, "Score missing or invalid"):
            score.validate_cards(corrupted, self.prepared)
        corrupted = copy.deepcopy(cards)
        corrupted[0]["failure_origin"] = "fantasy"
        with self.assertRaisesRegex(RuntimeError, "Failure origin missing/invalid"):
            score.validate_cards(corrupted, self.prepared)
        corrupted = copy.deepcopy(cards)
        corrupted[0]["hard_failure"] = True
        with self.assertRaisesRegex(RuntimeError, "Hard failure requires"):
            score.validate_cards(corrupted, self.prepared)

    def test_scored_weight_effect_is_paired_and_ci_not_a_raw_average(self) -> None:
        graded = score.validate_cards(self._graded(), self.prepared)
        summary = score.summarize(self.report, graded, self.lookup)
        effect = summary["primary_weight_effect"]
        self.assertEqual(effect["mean_delta"], 2.0)
        self.assertEqual(effect["ci95"], [2.0, 2.0])
        self.assertEqual(effect["direction"], "positive")
        self.assertEqual(effect["tasks"], 3)
        self.assertEqual(summary["evaluable_primary_tasks"], 3)
        self.assertEqual(summary["evaluable_secondary_tasks"], 3)
        self.assertEqual(summary["evaluable_all_three_tasks"], 3)
        self.assertFalse(summary["automated_promotion"])

    def test_infrastructure_failure_is_not_counted_as_model_regression(self) -> None:
        cells = copy.deepcopy(self.report)
        target = next(cell for cell in cells["cells"] if cell["arm"] == "base_localpilot")
        target["error"] = {"type": "RuntimeError", "message": "broken broker"}
        cards = score.prepare(cells, self.lookup)
        for row in cards:
            row["score"], row["hard_failure"], row["failure_origin"] = 2, False, "none"
            row["rationale"], row["reviewer"] = "Reviewed", "external"
        faulty = next(row for row in cards if row["execution_error"])
        with self.assertRaisesRegex(RuntimeError, "cannot be silently classed as model performance"):
            score.validate_cards(cards, cards)
        faulty["failure_origin"] = "harness"
        summary = score.summarize(cells, score.validate_cards(cards, score.prepare(cells, self.lookup)), self.lookup)
        self.assertEqual(summary["evaluable_primary_tasks"], 2)
        self.assertEqual(summary["primary_weight_effect"]["tasks"], 2)
        self.assertTrue(summary["excluded_tasks"])
        self.assertEqual(summary["model_hard_failures"], {})

    def test_optional_direct_failure_does_not_exclude_primary_weight_effect(self) -> None:
        cells = copy.deepcopy(self.report)
        target = next(cell for cell in cells["cells"] if cell["arm"] == "base_direct")
        target["error"] = {"type": "RuntimeError", "message": "direct arm unavailable"}
        cards = score.prepare(cells, self.lookup)
        for row in cards:
            row["score"], row["hard_failure"], row["failure_origin"] = 2, False, "none"
            row["rationale"], row["reviewer"] = "Reviewed", "external"
        faulty = next(row for row in cards if row["execution_error"])
        faulty["failure_origin"] = "environment"
        summary = score.summarize(
            cells,
            score.validate_cards(cards, score.prepare(cells, self.lookup)),
            self.lookup,
        )
        self.assertEqual(summary["evaluable_primary_tasks"], 3)
        self.assertEqual(summary["primary_weight_effect"]["tasks"], 3)
        self.assertEqual(summary["evaluable_secondary_tasks"], 2)
        self.assertEqual(summary["secondary_scaffold_effect"]["tasks"], 2)
        self.assertEqual(summary["evaluable_all_three_tasks"], 2)

    def test_report_cannot_be_scored_when_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            with patch.object(matrix, "load_tasks", return_value=self.tasks):
                report, _ = score.load_report(path)
                self.assertEqual(report["planned_cells"], 27)
                self.report["cells"].pop()
                path.write_text(json.dumps(self.report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    score.load_report(path)


if __name__ == "__main__":
    unittest.main()
