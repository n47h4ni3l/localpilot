from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PhaseOneArtifactTests(unittest.TestCase):
    def test_frozen_aggregate_metadata_has_expected_scores_and_no_protected_content(self) -> None:
        expected = {
            "eval_v1_post_cc_gpt_oss_20b_f619abb.json": ("eval_name", 1.88, 1),
            "evolution_execution_pre_cc_gpt_oss_20b_c213be3.json": ("benchmark_name", 3.0, 2),
            "evolution_execution_post_cc_gpt_oss_20b_f619abb.json": ("benchmark_name", 3.75, 0),
        }
        forbidden_keys = {"prompt", "rubric", "answer", "response", "scorecard", "target_patch", "hidden_acceptance"}
        for filename, (_, overall, failures) in expected.items():
            value = json.loads((ROOT / "training" / "baselines" / filename).read_text(encoding="utf-8"))
            self.assertTrue(value["content_policy"]["aggregate_metadata_only"])
            self.assertEqual(value["scores"]["overall_mean"], overall)
            self.assertEqual(value["scores"]["hard_failure_count"], failures)
            serialized_keys = {str(key).casefold() for item in _walk(value) if isinstance(item, dict) for key in item}
            self.assertTrue(forbidden_keys.isdisjoint(serialized_keys))


def _walk(value: object):
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


if __name__ == "__main__":
    unittest.main()
