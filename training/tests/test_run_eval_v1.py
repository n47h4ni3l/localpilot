from __future__ import annotations

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval_v1.py"
SPEC = importlib.util.spec_from_file_location("localpilot_run_eval_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class EvalV1RunnerTests(unittest.TestCase):
    def test_seed_suite_loads_25_held_out_tasks_without_answers(self) -> None:
        tasks = runner.load_eval_tasks()
        self.assertEqual(len(tasks), 25)
        self.assertEqual(len({task["id"] for task in tasks}), 25)
        self.assertTrue(all(task["split"] == "held_out_eval" for task in tasks))
        self.assertTrue(
            all(
                all(message.get("role") != "assistant" for message in task["messages"])
                for task in tasks
            )
        )

    def test_baseline_checkout_rejects_non_main(self) -> None:
        state = {
            "branch": "feature/test",
            "head": "abc",
            "origin_main": "abc",
            "clean": True,
            "status": "",
        }
        with self.assertRaisesRegex(RuntimeError, "must run from main"):
            runner.require_baseline_checkout(
                state,
                allow_non_main=False,
                allow_dirty=False,
                skip_upstream_check=False,
            )

    def test_baseline_checkout_rejects_dirty_tree(self) -> None:
        state = {
            "branch": "main",
            "head": "abc",
            "origin_main": "abc",
            "clean": False,
            "status": " M file.py",
        }
        with self.assertRaisesRegex(RuntimeError, "clean working tree"):
            runner.require_baseline_checkout(
                state,
                allow_non_main=False,
                allow_dirty=False,
                skip_upstream_check=False,
            )

    def test_baseline_checkout_rejects_stale_main(self) -> None:
        state = {
            "branch": "main",
            "head": "abc",
            "origin_main": "def",
            "clean": True,
            "status": "",
        }
        with self.assertRaisesRegex(RuntimeError, "does not match origin/main"):
            runner.require_baseline_checkout(
                state,
                allow_non_main=False,
                allow_dirty=False,
                skip_upstream_check=False,
            )

    def test_snapshot_contains_source_but_removes_entire_training_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval-snapshot-") as temp_dir:
            destination = Path(temp_dir) / "repo"
            runner.build_isolated_snapshot(runner.ROOT, destination)
            self.assertTrue((destination / "localpilot" / "agent.py").is_file())
            self.assertFalse((destination / "training").exists())

    def test_tool_surface_is_reduced_to_repository_readers(self) -> None:
        tools = {name: object() for name in runner.ALLOWED_EVAL_TOOLS}
        tools["search_public_web"] = object()
        tools["open_windows_app"] = object()
        agent = types.SimpleNamespace(tools=tools)
        runner.restrict_eval_tools(agent)
        self.assertEqual(set(agent.tools), set(runner.ALLOWED_EVAL_TOOLS))

    def test_run_report_task_prompt_does_not_include_expected_behavior(self) -> None:
        task = runner.load_eval_tasks()[0]
        prompt, systems = runner._task_prompt(task)
        self.assertEqual(systems, [])
        self.assertEqual(prompt, task["messages"][0]["content"])
        self.assertNotIn("expected_behavior", prompt)


if __name__ == "__main__":
    unittest.main()
