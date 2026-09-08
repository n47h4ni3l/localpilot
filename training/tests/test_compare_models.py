from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from training.scripts.compare_models import CRITICAL_CATEGORIES, compare
from training.scripts.score_eval_v1 import summarize as summarize_eval
from training.scripts.score_evolution_execution import score_report as summarize_execution


def metadata(task_count: int, suite: str) -> dict:
    digest = hashlib.sha256(suite.encode()).hexdigest()
    return {
        "model": {"name": "test-model", "digest": "test-model-immutable-digest"},
        "task_count": task_count, "expected_task_count": task_count,
        "task_ids_sha256": digest, "expected_task_ids_sha256": digest,
        "suite_sha256": digest,
    }


def eval_summary(overall: float, critical: float, hard_failures: int, categories: dict[str, float]) -> dict:
    return {
        **metadata(4, "eval-test-suite"),
        "eval_name": "LocalPilot Eval v1",
        "scores": {
            "overall_mean": overall,
            "critical_category_mean": critical,
            "hard_failure_count": hard_failures,
            "category_means": dict(categories),
            "category_counts": {name: 1 for name in categories} if len(categories) == 4 else {},
            "training_loss": -999,
        },
    }


def execution_summary(overall: float, perfect: int, hard_failures: int, scope: int = 0) -> dict:
    return {
        **metadata(4, "execution-test-suite"),
        "benchmark_name": "LocalPilot Evolution Execution v1",
        "scores": {
            "overall_mean": overall,
            "perfect_task_count": perfect,
            "hard_failure_count": hard_failures,
            "scope_violation_count": scope,
        },
    }


class CompareModelsTests(unittest.TestCase):
    def evidence(self) -> list[dict]:
        categories = {name: 2.0 for name in CRITICAL_CATEGORIES}
        return [eval_summary(2, 2, 1, categories), eval_summary(2, 2, 0, categories),
                execution_summary(3.75, 3, 0), execution_summary(4, 4, 0)]

    def test_promotes_only_from_eval_and_execution_evidence(self) -> None:
        evidence = self.evidence()
        evidence[0]["scores"]["category_means"]["debugging"] = 1.25
        result = compare(*evidence)
        self.assertTrue(result["promotion_recommended"])
        self.assertEqual(result["eval_v1"]["category_deltas"]["debugging"]["delta"], 0.75)
        self.assertIn("training loss is not a promotion signal", result["promotion_basis"])
        self.assertTrue(result["human_promotion_required"])
        self.assertTrue(result["preference"]["eval_hard_failures_reduced"])

    def test_critical_regression_blocks_promotion(self) -> None:
        result = compare(
            eval_summary(1.88, 1.6875, 1, {}),
            eval_summary(2.0, 1.6, 0, {}),
            execution_summary(3.75, 3, 0),
            execution_summary(4.0, 4, 0),
        )
        self.assertFalse(result["promotion_recommended"])
        self.assertFalse(result["gates"]["no_material_critical_regression"])

    def test_execution_regression_or_scope_violation_blocks_promotion(self) -> None:
        result = compare(
            eval_summary(1.88, 1.6875, 1, {}),
            eval_summary(2.0, 1.8, 0, {}),
            execution_summary(3.75, 3, 0),
            execution_summary(3.5, 3, 0, scope=1),
        )
        self.assertFalse(result["promotion_recommended"])
        self.assertFalse(result["gates"]["execution_at_least_baseline"])
        self.assertFalse(result["gates"]["no_scope_violations"])

    def test_critical_category_regression_cannot_hide_in_improved_average(self) -> None:
        evidence = self.evidence()
        evidence[1]["scores"]["critical_category_mean"] = 3.0
        evidence[1]["scores"]["category_means"]["debugging"] = 1.0
        result = compare(*evidence)
        self.assertTrue(result["gates"]["no_material_critical_regression"])
        self.assertFalse(result["gates"]["no_material_critical_category_regression"])
        self.assertFalse(result["promotion_recommended"])

    def test_explicit_tolerance_applies_to_each_critical_category(self) -> None:
        evidence = self.evidence()
        evidence[1]["scores"]["critical_category_mean"] = 1.95
        evidence[1]["scores"]["category_means"]["debugging"] = 1.95
        self.assertTrue(compare(*evidence, critical_tolerance=0.1)["promotion_recommended"])

    def test_missing_scope_is_unknown_and_cannot_promote(self) -> None:
        evidence = self.evidence()
        del evidence[3]["scores"]["scope_violation_count"]
        result = compare(*evidence)
        self.assertIsNone(result["gates"]["no_scope_violations"])
        self.assertIsNone(result["evolution_execution"]["delta"]["scope_violations"])
        self.assertFalse(result["promotion_recommended"])

    def test_missing_category_evidence_cannot_promote(self) -> None:
        evidence = self.evidence()
        del evidence[1]["scores"]["category_means"]
        del evidence[1]["scores"]["category_counts"]
        result = compare(*evidence)
        self.assertFalse(result["promotion_recommended"])
        self.assertIsNone(result["eval_v1"]["category_deltas"]["debugging"]["delta"])

    def test_missing_coverage_metadata_cannot_promote(self) -> None:
        for index in range(4):
            for field in ("task_count", "expected_task_count", "task_ids_sha256", "expected_task_ids_sha256", "suite_sha256"):
                with self.subTest(index=index, field=field):
                    evidence = self.evidence()
                    del evidence[index][field]
                    result = compare(*evidence)
                    self.assertFalse(result["promotion_recommended"])
                    self.assertEqual(result["comparison_status"], "incomplete_evidence")

    def test_partial_different_tasks_or_different_suite_cannot_promote(self) -> None:
        for field, value in (("expected_task_count", 5), ("task_ids_sha256", "b" * 64), ("suite_sha256", "b" * 64)):
            with self.subTest(field=field):
                evidence = self.evidence()
                evidence[3][field] = value
                result = compare(*evidence)
                self.assertFalse(result["promotion_recommended"])
                self.assertFalse(result["gates"]["execution_complete_matching_coverage"])

    def test_model_digest_mismatch_or_omission_cannot_promote(self) -> None:
        for model in ({"name": "test-model"}, {"name": "test-model", "digest": "other-model"}, None):
            evidence = self.evidence()
            evidence[3]["model"] = model
            self.assertFalse(compare(*evidence)["promotion_recommended"])

    def test_invalid_numbers_fail_closed(self) -> None:
        for field, values in {
            "overall_mean": [float("nan"), float("inf"), -1, 4.1, True, "2"],
            "critical_category_mean": [float("nan"), float("-inf"), -1, 5],
            "hard_failure_count": [float("nan"), 0.5, -1, True, 5],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    evidence = self.evidence()
                    evidence[1]["scores"][field] = value
                    with self.assertRaises(RuntimeError):
                        compare(*evidence)
        for value in (float("nan"), float("inf"), -0.01, 4.1):
            with self.assertRaises(RuntimeError):
                compare(*self.evidence(), critical_tolerance=value)
        for value in (float("nan"), -1, 0.5, 5, True):
            evidence = self.evidence()
            evidence[3]["scores"]["scope_violation_count"] = value
            with self.assertRaises(RuntimeError):
                compare(*evidence)

    def test_invalid_category_score_or_coverage_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence[1]["scores"]["category_means"]["debugging"] = float("nan")
        with self.assertRaises(RuntimeError):
            compare(*evidence)
        evidence = self.evidence()
        evidence[1]["scores"]["category_counts"]["debugging"] = 2
        with self.assertRaises(RuntimeError):
            compare(*evidence)

    def test_accepts_top_level_scorer_format_without_exposing_scorecard_rows(self) -> None:
        evidence = self.evidence()
        for summary in evidence:
            summary.update(summary.pop("scores"))
            summary["scores"] = [{"rationale": "private-review-text"}]
        result = compare(*evidence)
        self.assertTrue(result["promotion_recommended"])
        self.assertNotIn("private-review-text", json.dumps(result))

    def test_frozen_aggregate_comparison_has_deltas_but_does_not_invent_evidence(self) -> None:
        baselines = Path(__file__).resolve().parents[1] / "baselines"
        filenames = ("eval_v1_gpt_oss_20b_6670fa9.json", "eval_v1_post_cc_gpt_oss_20b_f619abb.json",
                     "evolution_execution_pre_cc_gpt_oss_20b_c213be3.json", "evolution_execution_post_cc_gpt_oss_20b_f619abb.json")
        result = compare(*(json.loads((baselines / name).read_text(encoding="utf-8")) for name in filenames))
        self.assertEqual(result["eval_v1"]["delta"]["overall"], 0.4)
        self.assertEqual(result["evolution_execution"]["delta"]["overall"], 0.75)
        self.assertTrue(result["observed_metric_gates_passed"])
        self.assertFalse(result["promotion_recommended"])
        self.assertEqual(result["comparison_status"], "incomplete_evidence")


class ScorerMetadataTests(unittest.TestCase):
    """Use independent tiny fixtures; never load the held-out suite in these tests."""

    def test_eval_metadata_proves_partial_run_is_incomplete(self) -> None:
        cards = [{"task_id": "a", "task_type": "debugging", "score": 4, "hard_failure": False, "rationale": "test"}]
        summary = summarize_eval(cards, {"model": {}}, {"a": {"id": "a"}, "b": {"id": "b"}})
        self.assertEqual(summary["expected_task_count"], 2)
        self.assertEqual(summary["category_counts"], {"debugging": 1})
        self.assertNotEqual(summary["task_ids_sha256"], summary["expected_task_ids_sha256"])

    def test_execution_metadata_counts_actual_scope_violation_not_missing_required_edit(self) -> None:
        cases = {
            "a": {"allowed_paths": ["allowed.py"], "criteria": [
                {"id": "scope", "kind": "scope_clean", "require_changed": ["allowed.py"]},
            ]},
            "b": {"allowed_paths": [], "criteria": []},
        }
        result = {"task_id": "a", "changed_paths": [], "out_of_scope_attempts": []}
        report = {"tasks": [result]}
        summary = summarize_execution(report, cases)
        self.assertEqual(summary["scope_violation_count"], 0)
        self.assertEqual(summary["expected_task_count"], 2)
        self.assertNotEqual(summary["task_ids_sha256"], summary["expected_task_ids_sha256"])
        result["changed_paths"] = ["unexpected.py"]
        self.assertEqual(summarize_execution(report, cases)["scope_violation_count"], 1)
        result["changed_paths"] = []
        result["out_of_scope_attempts"] = ["blocked attempt"]
        self.assertEqual(summarize_execution(report, cases)["scope_violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
