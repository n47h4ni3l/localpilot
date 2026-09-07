from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"
SPEC = importlib.util.spec_from_file_location("localpilot_validate_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class ValidateDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix=".validator-", dir=Path(__file__).parent)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, records: list[dict]) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    @staticmethod
    def train_record(record_id: str = "lp-train-001", prompt: str = "Inspect this repository failure.") -> dict:
        return {
            "id": record_id,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Inspect current evidence before concluding."},
            ],
            "task_type": "debugging",
            "source": "localpilot_verified_history",
            "license": "project_owned",
            "quality_tier": "A",
            "split": "train",
            "verification_status": "ci_verified",
            "provenance": {
                "repository": "n47h4ni3l/localpilot",
                "reference": "github:pr/82",
            },
        }

    @staticmethod
    def eval_record(record_id: str = "lp-eval-001", prompt: str = "A search returned no symbol. What next?") -> dict:
        return {
            "id": record_id,
            "messages": [{"role": "user", "content": prompt}],
            "task_type": "repository_reasoning",
            "source": "localpilot_eval_v1_curated",
            "license": "project_owned",
            "quality_tier": "A",
            "split": "held_out_eval",
            "verification_status": "source_verified",
            "provenance": {
                "repository": "n47h4ni3l/localpilot",
                "reference": "localpilot/selfdev.py",
            },
            "expected_behavior": [
                "Treat an empty search as insufficient evidence of absence.",
                "Use another repository observation before concluding.",
            ],
        }

    def test_valid_train_and_eval_records_pass(self) -> None:
        path = self.write("valid.jsonl", [self.train_record(), self.eval_record()])
        report = validator.validate_files([path])
        self.assertTrue(report.valid, report.errors)

    def test_quality_tier_d_cannot_enter_train(self) -> None:
        record = self.train_record()
        record["quality_tier"] = "D"
        record["verification_status"] = "unverified"
        path = self.write("tier-d.jsonl", [record])
        report = validator.validate_files([path])
        self.assertFalse(report.valid)
        self.assertTrue(any("quality tier D" in error for error in report.errors))

    def test_held_out_eval_must_not_contain_answer(self) -> None:
        record = self.eval_record()
        record["messages"].append({"role": "assistant", "content": "Hidden answer"})
        path = self.write("leaky-answer.jsonl", [record])
        report = validator.validate_files([path])
        self.assertFalse(report.valid)
        self.assertTrue(any("must not contain assistant answers" in error for error in report.errors))

    def test_duplicate_ids_are_rejected(self) -> None:
        first = self.train_record("lp-dup-001", "Prompt one")
        second = self.train_record("lp-dup-001", "Prompt two")
        path = self.write("duplicate-id.jsonl", [first, second])
        report = validator.validate_files([path])
        self.assertTrue(any("duplicate id" in error for error in report.errors))

    def test_normalized_message_duplicates_are_rejected(self) -> None:
        first = self.train_record("lp-norm-001", "Inspect   current evidence")
        second = self.train_record("lp-norm-002", "  inspect current EVIDENCE  ")
        path = self.write("normalized.jsonl", [first, second])
        report = validator.validate_files([path])
        self.assertTrue(any("normalized-message duplicate" in error for error in report.errors))

    def test_train_to_held_out_prompt_leakage_is_rejected(self) -> None:
        prompt = "Decide whether a missing symbol proves the feature is absent."
        train = self.train_record("lp-leak-train", prompt)
        held_out = self.eval_record("lp-leak-eval", "  decide whether a missing symbol PROVES the feature is absent. ")
        path = self.write("split-leakage.jsonl", [train, held_out])
        report = validator.validate_files([path])
        self.assertTrue(any("held-out prompt leakage" in error for error in report.errors))

    def test_missing_provenance_reference_is_rejected(self) -> None:
        record = self.eval_record()
        record["provenance"] = {"repository": "n47h4ni3l/localpilot"}
        path = self.write("missing-reference.jsonl", [record])
        report = validator.validate_files([path])
        self.assertTrue(any("provenance.reference" in error for error in report.errors))

    def test_directory_input_recurses_jsonl_files(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        self.write("root.jsonl", [self.train_record("lp-dir-001", "Root prompt")])
        (nested / "nested.jsonl").write_text(
            json.dumps(self.eval_record("lp-dir-002", "Nested prompt")) + "\n",
            encoding="utf-8",
        )
        report = validator.validate_files([self.root])
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.files_read, 2)

    def test_main_returns_nonzero_for_invalid_dataset(self) -> None:
        record = self.eval_record()
        del record["license"]
        path = self.write("invalid.jsonl", [record])
        self.assertEqual(validator.main([str(path)]), validator.EXIT_INVALID)


if __name__ == "__main__":
    unittest.main()
