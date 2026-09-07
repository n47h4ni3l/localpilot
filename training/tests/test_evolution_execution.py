from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_evolution_execution.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("localpilot_run_evolution_execution", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)

SCORER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_evolution_execution.py"
SCORER_SPEC = importlib.util.spec_from_file_location("localpilot_score_evolution_execution", SCORER_PATH)
assert SCORER_SPEC is not None and SCORER_SPEC.loader is not None
scorer = importlib.util.module_from_spec(SCORER_SPEC)
SCORER_SPEC.loader.exec_module(scorer)


class EvolutionExecutionDefinitionTests(unittest.TestCase):
    def test_frozen_eval_baseline_contains_aggregate_metadata_only(self) -> None:
        path = runner.ROOT / "training" / "baselines" / "eval_v1_gpt_oss_20b_6670fa9.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rendered_keys = {
            key
            for value in artifact.values()
            if isinstance(value, dict)
            for key in value
        }
        self.assertTrue(artifact["content_policy"]["aggregate_metadata_only"])
        self.assertFalse({"prompt", "rubric", "response", "scorecard_rows"}.intersection(rendered_keys))
        self.assertNotIn("tasks", artifact)

    def test_suite_has_four_small_cases_and_four_criteria_each(self) -> None:
        document = runner.load_case_document()
        self.assertEqual(document["score_scale_max"], 4)
        self.assertEqual(len(document["cases"]), 4)
        self.assertEqual(len({case["id"] for case in document["cases"]}), 4)
        self.assertTrue(all(len(case["criteria"]) == 4 for case in document["cases"]))
        self.assertTrue(all("target_patch" not in case for case in document["cases"]))

    def test_model_messages_exclude_hidden_checks_and_scoring_criteria(self) -> None:
        case = runner.load_case_document()["cases"][0]
        rendered = json.dumps(runner.implementation_messages(case), sort_keys=True)
        hidden_source = (
            runner.BENCHMARK_ROOT
            / case["hidden_fixture"]
            / "test_acceptance.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(case["hidden_fixture"], rendered)
        self.assertNotIn("old_import_is_same_function_object", rendered)
        self.assertNotIn(hidden_source, rendered)
        for criterion in case["criteria"]:
            self.assertNotIn(criterion["id"], rendered)

    def test_hidden_acceptance_is_removed_after_evaluator_check(self) -> None:
        case = runner.load_case_document()["cases"][0]
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "fixture"
            runner.create_fixture_repository(case, workspace)
            results = runner.run_checks(case, workspace)
            self.assertEqual({item["id"] for item in results}, {"project_tests", "hidden_acceptance"})
            self.assertFalse((workspace / runner.ACCEPTANCE_DIR_NAME).exists())

    def test_evaluator_checks_do_not_pollute_changed_paths_with_bytecode(self) -> None:
        case = runner.load_case_document()["cases"][0]
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "fixture"
            runner.create_fixture_repository(case, workspace)
            runner.run_checks(case, workspace)
            self.assertEqual(runner.changed_paths(workspace), [])
            self.assertEqual(list(workspace.rglob("*.py[co]")), [])

    def test_diagnosis_fixture_reproduces_its_committed_failure(self) -> None:
        case = next(
            item
            for item in runner.load_case_document()["cases"]
            if item["id"] == "evoexec-diagnose-repair"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "fixture"
            runner.create_fixture_repository(case, workspace)
            checks = {item["id"]: item for item in runner.run_checks(case, workspace)}
            self.assertNotEqual(checks["project_tests"]["returncode"], 0)


class RecordingCandidateToolsTests(unittest.TestCase):
    def test_records_successful_write_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tools = runner.RecordingCandidateTools(
                workspace,
                allowed_paths=["tests/test_item.py", "item.py"],
                readable_paths=["tests/test_item.py", "item.py"],
                max_files=2,
            )
            tools.write_project_file("tests/test_item.py", "VALUE = 1\n")
            tools.write_project_file("item.py", "VALUE = 1\n")
            self.assertEqual(
                [event["path"] for event in tools.write_events],
                ["tests/test_item.py", "item.py"],
            )

    def test_rejects_and_records_out_of_scope_write_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tools = runner.RecordingCandidateTools(
                workspace,
                allowed_paths=["item.py"],
                readable_paths=["item.py"],
                max_files=1,
            )
            with self.assertRaises(PermissionError):
                tools.write_project_file("README.md", "not allowed\n")
            self.assertEqual(len(runner.out_of_scope_attempts(tools)), 1)
            self.assertFalse((workspace / "README.md").exists())


class EvolutionExecutionScorerTests(unittest.TestCase):
    def _perfect_report(self) -> dict:
        cases = scorer.load_cases()
        tasks = []
        for case in cases.values():
            baseline_project_code = 1 if case["id"] == "evoexec-diagnose-repair" else 0
            events = [{"path": path} for path in case["allowed_paths"]]
            if case["id"] == "evoexec-regression-first":
                events = [
                    {"path": "tests/test_pricing.py"},
                    {"path": "pricing.py"},
                ]
            tasks.append(
                {
                    "task_id": case["id"],
                    "baseline_checks": [
                        {"id": "project_tests", "returncode": baseline_project_code, "timed_out": False},
                        {"id": "hidden_acceptance", "returncode": 1, "timed_out": False},
                    ],
                    "initial_checks": [],
                    "final_checks": [
                        {"id": "project_tests", "returncode": 0, "timed_out": False},
                        {"id": "hidden_acceptance", "returncode": 0, "timed_out": False},
                    ],
                    "changed_paths": case["allowed_paths"],
                    "write_events": events,
                    "out_of_scope_attempts": [],
                    "runtime_error": None,
                }
            )
        return {
            "benchmark_name": "LocalPilot Evolution Execution v1",
            "label": "test",
            "repository": {"head": "abc"},
            "model": {"name": "test-model"},
            "isolation": {"disposable_fixture_repository_per_task": True},
            "tasks": tasks,
        }

    def test_perfect_report_scores_four_for_every_case(self) -> None:
        summary = scorer.score_report(self._perfect_report())
        self.assertEqual(summary["overall_mean"], 4.0)
        self.assertEqual(summary["perfect_task_count"], 4)
        self.assertEqual(summary["hard_failure_count"], 0)

    def test_evaluator_bytecode_is_ignored_by_scope_scoring(self) -> None:
        report = self._perfect_report()
        report["tasks"][0]["changed_paths"].extend(
            [
                "acme/__pycache__/legacy.cpython-312.pyc",
                "tests/__pycache__/test_legacy.cpython-312.pyo",
            ]
        )
        summary = scorer.score_report(report)
        first = next(
            item for item in summary["tasks"]
            if item["task_id"] == report["tasks"][0]["task_id"]
        )
        scope = next(
            item for item in first["criteria"]
            if item["criterion_id"] == "scope_preserved"
        )
        self.assertEqual(first["score"], 4)
        self.assertTrue(scope["passed"])

    def test_unexpected_source_file_loses_scope_criterion(self) -> None:
        report = self._perfect_report()
        report["tasks"][0]["changed_paths"].append("acme/unexpected.py")
        summary = scorer.score_report(report)
        first = next(
            item for item in summary["tasks"]
            if item["task_id"] == report["tasks"][0]["task_id"]
        )
        scope = next(
            item for item in first["criteria"]
            if item["criterion_id"] == "scope_preserved"
        )
        self.assertEqual(first["score"], 3)
        self.assertFalse(scope["passed"])
        self.assertIn("acme/unexpected.py", scope["detail"])

    def test_scope_attempt_loses_scope_criterion_even_with_evaluator_bytecode(self) -> None:
        report = self._perfect_report()
        report["tasks"][0]["changed_paths"].append(
            "acme/__pycache__/legacy.cpython-312.pyc"
        )
        report["tasks"][0]["out_of_scope_attempts"] = ["README.md: rejected"]
        summary = scorer.score_report(report)
        first = next(
            item for item in summary["tasks"]
            if item["task_id"] == report["tasks"][0]["task_id"]
        )
        self.assertEqual(first["score"], 3)
        self.assertFalse(
            next(
                item for item in first["criteria"]
                if item["criterion_id"] == "scope_preserved"
            )["passed"]
        )

    def test_timeout_does_not_count_as_reproduced_failure(self) -> None:
        report = self._perfect_report()
        diagnosis = next(
            item for item in report["tasks"] if item["task_id"] == "evoexec-diagnose-repair"
        )
        diagnosis["baseline_checks"][0]["timed_out"] = True
        summary = scorer.score_report(report)
        scored = next(
            item for item in summary["tasks"] if item["task_id"] == "evoexec-diagnose-repair"
        )
        self.assertEqual(scored["score"], 3)

    def test_single_task_smoke_report_can_be_scored(self) -> None:
        report = self._perfect_report()
        report["tasks"] = report["tasks"][:1]
        summary = scorer.score_report(report)
        self.assertEqual(summary["task_count"], 1)
        self.assertEqual(summary["overall_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
