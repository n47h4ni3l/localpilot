"""Independent consistency checks for the selected P1.2 candidate corpus.

These tests do not constitute semantic quality approval or a target-PC dry-run.
They compare actual expanded training targets against claimed counts and
prevent a misleadingly small/large run or mid-run checkpoints.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))

from build_p12_corpus import build
from train_adapter import _validate_config, expand_training_examples, load_config

CONFIG = ROOT / "training/configs/qlora_p12_curated_1000.yaml"
LINEAGE = ROOT / "training/lineage/package-1.json"


class P12IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        content, cls.manifest = build()
        cls.rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        cls.config = load_config(CONFIG)
        cls.selection = json.loads((ROOT / "training/manifests/p12_selection.json").read_text(encoding="utf-8"))

    def test_selected_records_and_provenance(self):
        self.assertEqual(len(self.rows), 1000)
        self.assertEqual(len({r["id"] for r in self.rows}), 1000)
        self.assertEqual(Counter(r["split"] for r in self.rows),
                         {"train": 900, "validation": 100})
        self.assertEqual(Counter(r["source"] for r in self.rows), {
            "HuggingFaceTB/smoltalk": 400,
            "HuggingFaceTB/everyday-conversations-llama3.1-2k": 100,
            "Team-ACE/ToolACE": 150,
            "bigcode/self-oss-instruct-sc2-exec-filter-50k": 250,
            "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2": 100,
        })
        curated = [r for r in self.rows if r["task_type"] == "self_development"]
        self.assertEqual(len(curated), 50)
        for row in curated:
            self.assertEqual(row["verification_status"], "synthetic_validated")
            self.assertNotEqual(row["verification_status"], "human_verified")
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertTrue(row["messages"][-1]["content"].strip())
            evidence = row["metadata"]["self_improvement_evidence"]
            self.assertEqual(evidence["failed_execution"]["exit_code"], 1)
            self.assertEqual(evidence["passed_execution"]["exit_code"], 0)
            self.assertEqual(evidence["changed_expressions"], 1)
            self.assertTrue(evidence["tests_preserved"])
            self.assertTrue(evidence["passed_execution"]["all_assertions_executed"])
        self.assertTrue(all(r["metadata"]["fresh_source"] for r in self.rows))
        self.assertTrue(all(r["metadata"]["semantic_review"] == "pending_independent_review" for r in self.rows))
        self.assertEqual(len({r["metadata"]["family_id"] for r in self.rows}), 1000)

    def test_real_expansion_matches_manifest_and_step_count(self):
        expanded_train = expand_training_examples(self.rows, "train")
        expanded_validation = expand_training_examples(self.rows, "validation")
        self.assertEqual(len(expanded_train), self.selection["training_example_counts"]["train"])
        self.assertEqual(len(expanded_validation), self.selection["training_example_counts"]["validation"])
        self.assertEqual(
            self.manifest["training_example_counts"],
            {"train": len(expanded_train),
             "validation": len(expanded_validation),
             "total": len(expanded_train) + len(expanded_validation)},
        )
        training = self.config["training"]
        self.assertEqual(training["max_steps"], None)
        steps = math.ceil(len(expanded_train) / training["effective_batch_size"]) * training["epochs"]
        self.assertEqual(steps, self.selection["estimated_optimizer_steps"])
        self.assertEqual(training["estimated_optimizer_steps"], steps)
        self.assertEqual(training["checkpoint_steps"], steps)
        self.assertEqual(training["first_checkpoint_step"], steps)
        self.assertEqual(training["validation_steps"], steps)
        self.assertEqual(training["eval_strategy"], "no")
        self.assertEqual(training["save_total_limit"], 1)

    def test_p1_lineage_and_unchanged_training_recipe(self):
        _validate_config(self.config)
        accepted = json.loads(LINEAGE.read_text(encoding="utf-8"))
        self.assertTrue(accepted["accepted_as_lineage_head"])
        self.assertEqual(accepted["deployment"]["model"], "nestra:20b-p1")
        self.assertEqual(self.config["lineage"]["package"], 2)
        self.assertEqual(self.config["lineage"]["parent_manifest"],
                         "training/lineage/package-1.json")
        self.assertEqual(self.manifest["parent_manifest"],
                         "training/lineage/package-1.json")
        self.assertEqual(self.config["adapter"]["rank"], 8)
        self.assertEqual(self.config["adapter"]["alpha"], 16)
        self.assertEqual(self.config["training"]["learning_rate"], 0.0001)
        self.assertEqual(self.config["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(self.config["training"]["micro_batch_size"], 1)
        self.assertEqual(self.config["training"]["precision"], "bf16")
        self.assertEqual(self.config["backend"]["compile_mode"], "native")
        self.assertEqual(self.config["output"]["save_adapter_only"], True)
        self.assertEqual(self.config["status"], "proposed")


if __name__ == "__main__":
    unittest.main()
