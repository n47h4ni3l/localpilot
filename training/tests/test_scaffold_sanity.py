from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.scripts import run_scaffold_sanity as sanity
from training.scripts import score_scaffold_sanity as scorer
from training.scripts import score_eval_matrix as blind


class ScaffoldSanityTests(unittest.TestCase):
    def test_sanity_plan_uses_same_p1_weights_in_two_arms(self):
        tasks = sanity.selected_tasks()
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({t["id"] for t in tasks}), 8)
        keys = sanity.plan(tasks)
        self.assertEqual(len(keys), 48)
        self.assertEqual(len(set(keys)), 48)
        for task in tasks:
            per_task = [cell for cell in keys if cell[0] == task["id"]]
            self.assertEqual(len(per_task), 6)
            for rep in range(3):
                self.assertEqual({arm for _, r, arm in per_task if r == rep}, set(sanity.ARMS))
            first = [arm for _, rep, arm in per_task if rep == 0][0]
            second = [arm for _, rep, arm in per_task if rep == 1][0]
            self.assertNotEqual(first, second)

    def test_resume_rejects_changed_weights_or_revision(self):
        cfg = sanity.matrix.original.isolated_config(sanity.ROOT, model="nestra:20b-p1")
        tasks = sanity.selected_tasks()
        with patch.object(sanity.matrix.original, "ollama_model_identity",
                          side_effect=lambda name: {"name": name, "digest": name}):
            report = sanity.setup_report(tasks, {"head": "sha"}, cfg)
            sanity.validate_resume(report, tasks, {"head": "sha"}, cfg)
            with self.assertRaisesRegex(RuntimeError, "repository changed"):
                sanity.validate_resume(report, tasks, {"head": "different"}, cfg)
            altered = copy.deepcopy(cfg)
            altered.model.name = "another-model"
            with self.assertRaisesRegex(RuntimeError, "Model weights changed"):
                sanity.validate_resume(report, tasks, {"head": "sha"}, altered)
            report["cells"] = [{"task_id": tasks[0]["id"], "repeat": 1, "arm": "nestra_direct"}] * 2
            with self.assertRaisesRegex(RuntimeError, "Duplicated"):
                sanity.validate_resume(report, tasks, {"head": "sha"}, cfg)

    def test_scorer_rejects_incomplete_runs(self):
        cfg = sanity.matrix.original.isolated_config(sanity.ROOT, model="nestra:20b-p1")
        tasks = sanity.selected_tasks()
        with patch.object(sanity.matrix.original, "ollama_model_identity",
                          return_value={"digest": "same-weights"}):
            report = sanity.setup_report(tasks, {"head": "sha"}, cfg)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "run.json"
            output.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                scorer.load_run(output)

    def test_blind_score_summary_keeps_harness_issues_visible(self):
        tasks = sanity.selected_tasks()
        cfg = sanity.matrix.original.isolated_config(sanity.ROOT, model="nestra:20b-p1")
        with patch.object(sanity.matrix.original, "ollama_model_identity",
                          return_value={"digest": "p1"}):
            report = sanity.setup_report(tasks, {"head": "sha"}, cfg)
        for task_id, rep, arm in sanity.plan(tasks):
            report["cells"].append({
                "task_id": task_id, "task_type": next(t["task_type"] for t in tasks if t["id"] == task_id),
                "repeat": rep + 1, "arm": arm,
                "response": "Answer with evidence" if arm == "nestra_localpilot" else "Uncertain",
                "error": None, "duration_seconds": 1,
                "evidence": {},
            })
        report["completed_at"] = "complete"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            checked, lookup = scorer.load_run(path)
        cards = blind.prepare(checked, lookup)
        self.assertTrue(all("arm" not in c and "repeat" not in c for c in cards))
        for c in cards:
            c["score"] = 3 if "Answer with evidence" in c["response"] else 1
            c["hard_failure"] = False
            c["failure_origin"] = "none"
            c["rationale"] = "Independent rubric check."
            c["reviewer"] = "example-independent"
        validated = blind.validate_cards(cards, blind.prepare(checked, lookup))
        results = scorer.summarize(checked, validated, lookup)
        self.assertEqual(results["graded_responses"], 48)
        self.assertEqual(results["scaffold_minus_direct_paired_delta"], 2)
        self.assertEqual(results["scaffold_task_wins"], 8)
        self.assertFalse(results["automated_promotion"])


if __name__ == "__main__":
    unittest.main()
