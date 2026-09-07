from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_eval_v1.py"
SPEC = importlib.util.spec_from_file_location("localpilot_score_eval_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


class EvalV1ScoringTests(unittest.TestCase):
    @staticmethod
    def run_report() -> dict:
        return {
            "eval_name": "LocalPilot Eval v1",
            "label": "baseline",
            "repository": {"head": "abc"},
            "model": {"name": "gpt-oss:20b"},
            "isolation": {"training_tree_removed": True},
            "tasks": [
                {
                    "task_id": "task-a",
                    "task_type": "debugging",
                    "response": "Inspect current evidence first.",
                    "error": None,
                },
                {
                    "task_id": "task-b",
                    "task_type": "generalization",
                    "response": "",
                    "error": {"type": "RuntimeError", "message": "failed"},
                },
            ],
        }

    @staticmethod
    def rubrics() -> dict:
        return {
            "task-a": {
                "task_id": "task-a",
                "task_type": "debugging",
                "difficulty": "medium",
                "prompt": "What next?",
                "expected_behavior": ["Inspect evidence."],
            },
            "task-b": {
                "task_id": "task-b",
                "task_type": "generalization",
                "difficulty": "hard",
                "prompt": "Transfer the principle.",
                "expected_behavior": ["Transfer rather than memorize."],
            },
        }

    def test_prepare_scorecard_keeps_rubric_separate_from_run(self) -> None:
        cards = scorer.prepare_scorecard(self.run_report(), self.rubrics())
        self.assertEqual(len(cards), 2)
        self.assertIsNone(cards[0]["score"])
        self.assertFalse(cards[0]["hard_failure"])
        self.assertEqual(cards[1]["score"], 0)
        self.assertTrue(cards[1]["hard_failure"])
        self.assertIn("expected_behavior", cards[0])

    def test_validate_scorecard_requires_scores_and_rationales(self) -> None:
        cards = scorer.prepare_scorecard(self.run_report(), self.rubrics())
        with self.assertRaisesRegex(RuntimeError, "numeric score"):
            scorer.validate_scorecard(cards, self.run_report(), self.rubrics())

        cards[0]["score"] = 3
        cards[0]["rationale"] = "Grounded and actionable with one material gap."
        cards[0]["reviewer"] = "test"
        scorer.validate_scorecard(cards, self.run_report(), self.rubrics())

    def test_summary_reports_overall_category_and_hard_failures(self) -> None:
        cards = scorer.prepare_scorecard(self.run_report(), self.rubrics())
        cards[0]["score"] = 4
        cards[0]["rationale"] = "Fully grounded."
        cards[0]["reviewer"] = "test"
        summary = scorer.summarize(cards, self.run_report())
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["overall_mean"], 2.0)
        self.assertEqual(summary["category_means"]["debugging"], 4.0)
        self.assertEqual(summary["category_means"]["generalization"], 0.0)
        self.assertEqual(summary["hard_failure_count"], 1)
        self.assertEqual(summary["critical_category_means"]["debugging"], 4.0)


if __name__ == "__main__":
    unittest.main()
