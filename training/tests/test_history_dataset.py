from __future__ import annotations

import tempfile
import copy
import unittest
from pathlib import Path
from unittest import mock

from training.scripts import build_localpilot_history_dataset as builder


class HistoryDatasetTests(unittest.TestCase):
    def test_leakage_failure_writes_only_rejection_report_and_preserves_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus.jsonl"
            report = Path(temporary) / "stats.json"
            output.write_text("existing corpus", encoding="utf-8")
            stats = builder.corpus_statistics([], duplicates_rejected=0, leakage_rejected=1)
            with (
                mock.patch.object(builder, "assert_output_path_allowed"),
                mock.patch.object(builder, "build_dataset", return_value=([], stats)),
                self.assertRaisesRegex(RuntimeError, "no corpus was written"),
            ):
                builder.main(["--output", str(output), "--report", str(report)])
            self.assertEqual(output.read_text(encoding="utf-8"), "existing corpus")
            self.assertIn('"held_out_leakage": 1', report.read_text(encoding="utf-8"))

    def test_tracked_corpus_is_verified_reproducible_and_leak_free(self) -> None:
        rows, stats = builder.build_dataset([builder.DEFAULT_SOURCE], builder.DEFAULT_MANIFEST)
        self.assertEqual(len(rows), 24)
        self.assertEqual(stats["counts"]["quality_tier"], {"A": 24})
        self.assertEqual(stats["counts"]["split"], {"train": 20, "validation": 4})
        self.assertEqual(set(stats["counts"]["task_type"].values()), {3})
        self.assertEqual(stats["provenance"]["completeness_percent"], 100.0)
        self.assertEqual(stats["rejections"], {"duplicates": 0, "held_out_leakage": 0})
        self.assertEqual(rows, builder.load_jsonl(builder.DEFAULT_OUTPUT))
        self.assertTrue(all(row["verification_status"] == "source_verified" for row in rows))
        self.assertTrue(all(row["provenance"]["source_evidence"]["files"] for row in rows))

    def test_forbidden_eval_source_is_rejected_before_read(self) -> None:
        forbidden = builder.ROOT / "training" / "evals" / "debugging" / "eval_v1_seed.jsonl"
        with mock.patch.object(builder, "load_jsonl") as reader:
            with self.assertRaisesRegex(RuntimeError, "Forbidden training source path"):
                builder.build_dataset([forbidden], builder.DEFAULT_MANIFEST)
        reader.assert_not_called()

    def test_outside_repository_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                builder.assert_source_path_allowed(Path(temporary) / "candidate.jsonl")

    def test_all_sources_are_checked_before_any_candidate_is_read(self) -> None:
        forbidden = builder.ROOT / "training" / "evolution_execution" / "acceptance" / "secret.jsonl"
        with mock.patch.object(builder, "load_jsonl") as reader:
            with self.assertRaisesRegex(RuntimeError, "Forbidden"):
                builder.build_dataset([builder.DEFAULT_SOURCE, forbidden], builder.DEFAULT_MANIFEST)
        reader.assert_not_called()

    def test_scorecards_target_patches_and_prefix_lookalikes_are_refused(self) -> None:
        for relative in ("docs/scorecard.jsonl", "localpilot/target_patch.jsonl", "docs/expected_patch.py", "README.md-untrusted.jsonl"):
            with self.subTest(path=relative), self.assertRaises(RuntimeError):
                builder.assert_source_path_allowed(builder.ROOT / relative)

    def test_symlink_source_is_refused_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "training" / "sources"
            sources.mkdir(parents=True)
            target = sources / "target.jsonl"
            target.write_text("{}\n", encoding="utf-8")
            alias = sources / "alias.jsonl"
            try:
                alias.symlink_to(target)
            except OSError:
                self.skipTest("Symlink creation unavailable on this Windows account")
            with self.assertRaisesRegex(RuntimeError, "symlinks"):
                builder.assert_source_path_allowed(alias, root)

    def test_unverified_or_malformed_source_does_not_reach_git(self) -> None:
        for field, value in (("verification_status", "unverified"), ("messages", []), ("task_type", "")):
            row = copy.deepcopy(builder.load_jsonl(builder.DEFAULT_SOURCE)[0])
            row[field] = value
            with (
                self.subTest(field=field),
                mock.patch.object(builder, "load_jsonl", return_value=[row]),
                mock.patch.object(builder, "validate_provenance") as verifier,
                self.assertRaisesRegex(RuntimeError, "schema"),
            ):
                builder.build_dataset([builder.DEFAULT_SOURCE], builder.DEFAULT_MANIFEST)
            verifier.assert_not_called()

    def test_forbidden_provenance_is_rejected_before_git_reads(self) -> None:
        row = copy.deepcopy(builder.load_jsonl(builder.DEFAULT_SOURCE)[0])
        row["provenance"]["tests"] = ["training/evolution_execution/acceptance/private.py"]
        with mock.patch.object(builder, "_git") as git:
            with self.assertRaisesRegex(RuntimeError, "Forbidden"):
                builder.validate_provenance(row)
        git.assert_not_called()

    def test_duplicate_prompts_with_changed_answers_cannot_cross_splits(self) -> None:
        first = copy.deepcopy(builder.load_jsonl(builder.DEFAULT_SOURCE)[0])
        second = copy.deepcopy(first)
        second["id"] += "-duplicate"
        second["split"] = "validation"
        second["messages"][-1]["content"] = "A different response to the same prompt."
        with (
            mock.patch.object(builder, "load_jsonl", return_value=[second, first]),
            mock.patch.object(builder, "validate_provenance", return_value={"files": [], "tests": []}),
        ):
            rows, stats = builder.build_dataset([builder.DEFAULT_SOURCE], builder.DEFAULT_MANIFEST)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], first["id"])
        self.assertEqual(stats["rejections"]["duplicates"], 1)

    def test_output_cannot_overwrite_source_or_held_out_files(self) -> None:
        for relative in ("training/sources/source.jsonl", "training/evals/secret.jsonl", "training/datasets/../../README.md"):
            with self.subTest(path=relative), self.assertRaises(RuntimeError):
                builder.assert_output_path_allowed(builder.ROOT / relative, "datasets", ".jsonl")

    def test_tier_d_never_enters_train_or_validation(self) -> None:
        row = {
            "id": "tier-d",
            "quality_tier": "D",
            "split": "train",
            "source": "localpilot_verified_history",
            "license": "project_owned",
        }
        with mock.patch.object(builder, "load_jsonl", return_value=[row]):
            with self.assertRaisesRegex(RuntimeError, "Tier D"):
                builder.build_dataset([builder.DEFAULT_SOURCE], builder.DEFAULT_MANIFEST)


if __name__ == "__main__":
    unittest.main()
