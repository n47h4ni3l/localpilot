"""Integrity and cumulative-lineage checks for frozen Nestra P1.2 curriculum."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
from validate_dataset import validate_files
from train_adapter import expand_training_examples, load_config, _validate_config

CORPUS = ROOT / "training/datasets/nestra_p12_verified_1000.jsonl"
MANIFEST = ROOT / "training/manifests/nestra_p12_verified_1000_manifest.json"
CONFIG = ROOT / "training/configs/qlora_p12_verified_1000.yaml"
EXTERNAL = ROOT / "training/datasets/external_corpus_v1.jsonl"
PROJECT = ROOT / "training/datasets/corpus_v1_seed.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestP12Curriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(CORPUS)
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.config = load_config(CONFIG)

    def test_frozen_bytes_and_counts(self):
        self.assertEqual(
            hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            self.manifest["corpus_sha256"],
        )
        self.assertEqual(len(self.rows), 591)
        self.assertEqual(self.manifest["records"], 591)
        self.assertEqual(self.manifest["artifact_type"], "external_corpus_manifest")
        self.assertEqual(Counter(row["split"] for row in self.rows), {"train": 534, "validation": 57})
        self.assertEqual(self.manifest["split_counts"], {"train": 534, "validation": 57})
        self.assertEqual(
            {split: len(expand_training_examples(self.rows, split)) for split in ("train", "validation")},
            {"train": 900, "validation": 100},
        )
        self.assertEqual(
            self.manifest["training_example_counts"],
            {"train": 900, "validation": 100, "total": 1000},
        )

    def test_schema_and_no_duplicates(self):
        report = validate_files([CORPUS])
        self.assertTrue(report.valid, "\n".join(report.errors[:20]))
        self.assertEqual({row["split"] for row in self.rows}, {"train", "validation"})
        self.assertEqual(len({row["id"] for row in self.rows}), len(self.rows))
        self.assertTrue(all(row["verification_status"] in {"source_verified", "ci_verified", "human_verified"} for row in self.rows))
        self.assertTrue(all(row["quality_tier"] in {"A", "B"} for row in self.rows))

    def test_no_unverified_or_transformed_source_rows(self):
        sources = {}
        for path in (EXTERNAL, PROJECT):
            for row in read_jsonl(path):
                self.assertNotIn(row["id"], sources)
                sources[row["id"]] = row
        self.assertEqual(
            self.manifest["source_snapshots"]["external_source_git_blob_sha"],
            "23fe83a13a130335ab869577e544a581837087bd",
        )
        for row in self.rows:
            self.assertIn(row["id"], sources, row["id"])
            self.assertEqual(row, sources[row["id"]], row["id"])
        self.assertEqual(sum(row["quality_tier"] == "A" for row in self.rows), 24)

    def test_accepted_p1_lineage_and_proven_training_settings(self):
        _validate_config(self.config)
        self.assertEqual(self.config["lineage"], {
            "package": 2,
            "parent_manifest": "training/lineage/package-1.json",
        })
        self.assertEqual(self.config["status"], "proposed")
        self.assertEqual(self.config["backend"]["compile_mode"], "native")
        self.assertEqual(self.config["training"]["optimizer"], "adamw_8bit")
        self.assertEqual(self.config["training"]["micro_batch_size"], 1)
        self.assertEqual(self.config["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(self.config["training"]["epochs"], 3)
        self.assertEqual(self.config["training"]["estimated_optimizer_steps"], 675)
        self.assertEqual(self.config["training"]["eval_strategy"], "no")
        self.assertNotIn("recovery", self.config)
        self.assertEqual(self.config["data"]["corpus_path"], "training/datasets/nestra_p12_verified_1000.jsonl")
        self.assertEqual(self.config["data"]["corpus_manifest"], "training/manifests/nestra_p12_verified_1000_manifest.json")
        self.assertNotIn("stop_after_step", self.config.get("diagnostics", {}))
        parent = json.loads((ROOT / "training/lineage/package-1.json").read_text(encoding="utf-8"))
        self.assertEqual(parent["accepted_as_lineage_head"], True)
        self.assertEqual(parent["deployment"]["model"], "nestra:20b-p1")
        self.assertEqual(parent["deployment"]["digest"], "870386dd9cbfe625ab110af128ffbf6216565e5c9d2d603a070e8f6228f203d9")


if __name__ == "__main__":
    unittest.main()
