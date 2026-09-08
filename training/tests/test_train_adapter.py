from __future__ import annotations

import copy
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from training.scripts import train_adapter as runner
from training.scripts.build_eval_manifest import build_manifest
from training.scripts.training_common import sha256_json, write_json, write_jsonl


def record(identifier: str, prompt: str, split: str) -> dict:
    row = {
        "id": identifier, "messages": [{"role": "user", "content": prompt}],
        "task_type": "debugging", "source": "localpilot_verified_history",
        "license": "project_owned", "quality_tier": "A", "split": split,
        "verification_status": "source_verified", "provenance": {"reference": "test:evidence"},
    }
    if split == "held_out_eval":
        row["expected_behavior"] = ["State independent test evidence and uncertainty."]
    else:
        row["messages"].append({"role": "assistant", "content": "A verified response."})
    return row


class TrainAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = runner.load_config(runner.DEFAULT_CONFIG)
        self.config["backend"]["python_version"] = list(sys.version_info[:2])
        self.config["model"]["cache_directory"] = str(self.root / "cache")
        self.dataset = self.root / self.config["data"]["corpus_path"]
        self.rows = [record("example-train", "Diagnose the parsing fault.", "train"), record("example-validation", "Validate the independent correction.", "validation")]
        write_jsonl(self.dataset, self.rows)
        self.eval_root = self.root / "fixture-evals"
        write_jsonl(self.eval_root / "test.jsonl", [record("eval-private", "Explain the secret canary policy.", "held_out_eval")])
        self.manifest = build_manifest(self.eval_root)
        self.manifest_path = self.root / self.config["data"]["eval_manifest"]
        write_json(self.manifest_path, self.manifest)
        self.config["data"]["eval_manifest_records_sha256"] = self.manifest["records_sha256"]
        self.config_path = self.root / "training/configs/qlora_v1.yaml"
        self.save_config()
        self.snapshot = self.root / "cache/hub/snapshots" / self.config["model"]["revision"]
        self.snapshot.mkdir(parents=True)
        header = json.dumps({"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
        (self.snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
        loaded_config = types.SimpleNamespace(model_type="gpt_oss", quantization_config={"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True})
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: list(range(20)))
        self.tokenizer_loader = mock.Mock(return_value=tokenizer)
        self.snapshot_loader = mock.Mock(return_value=str(self.snapshot))
        self.lora_loader = mock.Mock()
        self.modules = {name: types.SimpleNamespace(__version__=self.config["backend"]["package_versions"].get(name, "test")) for name in runner.REQUIRED_IMPORTS}
        self.modules["transformers"].AutoConfig = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=loaded_config))
        self.modules["transformers"].AutoTokenizer = types.SimpleNamespace(from_pretrained=self.tokenizer_loader)
        self.modules["huggingface_hub"].snapshot_download = self.snapshot_loader
        self.modules["peft"].LoraConfig = self.lora_loader
        self.importer = mock.Mock(side_effect=self.modules.__getitem__)
        for target, replacement in (
            ("ROOT", self.root), ("TRAINING_OUTPUT_ROOT", self.root / "training/outputs"),
            ("_is_wsl", lambda: True), ("_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"}),
            ("_system_ram_gib", lambda: 28.0), ("_free_storage_gib", lambda path: 100.0),
            ("_gpu_preflight", lambda *args: {"architecture": "gfx1201", "hip": "7.14.1"}),
        ):
            patcher = mock.patch.object(runner, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_config(self) -> None:
        write_json(self.config_path, self.config)

    def dry_run(self, **kwargs) -> dict:
        return runner.dry_run(self.config_path, importer=self.importer, **kwargs)

    def test_mocked_target_passes_without_training_or_implicit_download(self) -> None:
        report_path = self.root / "training/reports/custom.json"
        result = self.dry_run(report_path=report_path)
        self.assertTrue(result["passed"], [item for item in result["checks"] if not item["passed"]])
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["dataset_sha256"], sha256_json(self.rows))
        self.assertIn(str(report_path), result["resolved_training_command"])
        self.assertEqual(self.importer.call_args_list[0].args, ("unsloth",))
        self.assertTrue(self.snapshot_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(self.tokenizer_loader.call_args.kwargs["local_files_only"])
        self.lora_loader.assert_called_once()
        self.assertEqual(result["model_evidence"]["token_counts"]["total"], 40)

    def test_missing_or_truncated_weights_fail_even_with_tokenizer(self) -> None:
        (self.snapshot / "model.safetensors").write_bytes(b"truncated")
        result = self.dry_run()
        self.assertFalse(result["passed"])
        self.assertFalse(next(item for item in result["checks"] if item["name"] == "model_tokenizer_and_adapter")["passed"])

    def test_shard_index_cannot_escape_snapshot(self) -> None:
        write_json(self.snapshot / "model.safetensors.index.json", {"weight_map": {"weight": "../outside.safetensors"}})
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            runner._weight_inventory(self.snapshot)

    def test_held_out_prompt_copied_into_assistant_blocks_before_import(self) -> None:
        self.rows[0]["messages"][-1]["content"] = "Preface. EXPLAIN the secret canary policy. Epilogue."
        write_jsonl(self.dataset, self.rows)
        result = self.dry_run()
        self.assertFalse(result["passed"])
        self.importer.assert_not_called()

    def test_invalid_schema_and_held_out_source_path_are_refused_before_model_import(self) -> None:
        for mutation in ("schema", "path"):
            with self.subTest(mutation=mutation):
                if mutation == "schema":
                    self.rows[0]["license"] = ""
                    write_jsonl(self.dataset, self.rows)
                else:
                    self.config["data"]["corpus_path"] = "training/evals/private.jsonl"
                    self.save_config()
                self.assertFalse(self.dry_run()["passed"])
        self.importer.assert_not_called()

    def test_modified_or_empty_manifest_blocks_and_does_not_crash_report(self) -> None:
        for manifest in ({}, {**self.manifest, "records": []}):
            write_json(self.manifest_path, manifest)
            self.assertFalse(self.dry_run()["passed"])
        self.importer.assert_not_called()

    def test_missing_validation_split_is_refused(self) -> None:
        write_jsonl(self.dataset, self.rows[:1])
        self.assertFalse(self.dry_run()["passed"])

    def test_long_tokenized_example_is_refused_instead_of_truncated(self) -> None:
        self.tokenizer_loader.return_value.apply_chat_template = lambda *args, **kwargs: list(range(1025))
        self.assertFalse(self.dry_run()["passed"])

    def test_changed_backend_version_or_insufficient_ram_blocks(self) -> None:
        self.modules["bitsandbytes"].__version__ = "0.49.0"
        self.assertFalse(self.dry_run()["passed"])
        with mock.patch.object(runner, "_system_ram_gib", return_value=16.0):
            result = self.dry_run()
        self.assertFalse(next(item for item in result["checks"] if item["name"] == "system_ram")["passed"])

    def test_config_rejects_nonfinite_rates_boolean_rank_and_mismatched_batch(self) -> None:
        for section, key, value in (("training", "learning_rate", float("nan")), ("adapter", "rank", True), ("training", "effective_batch_size", 99)):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(RuntimeError):
                runner._validate_config(config)

    def test_training_requires_approval_and_rejects_changed_spec(self) -> None:
        result = self.dry_run()
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            runner._verify_training_gate(self.config, result, self.config_path)
        self.config["status"] = "approved_after_dry_run"
        runner._verify_training_gate(self.config, result, self.config_path)
        self.config["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(RuntimeError, "settings"):
            runner._verify_training_gate(self.config, result, self.config_path)

    def test_changed_dataset_manifest_and_environment_invalidate_training_evidence(self) -> None:
        result = self.dry_run()
        self.config["status"] = "approved_after_dry_run"
        altered = copy.deepcopy(result)
        altered["environment"]["machine"] = "another-machine"
        with self.assertRaisesRegex(RuntimeError, "environment"):
            runner._verify_training_gate(self.config, altered, self.config_path)
        altered = copy.deepcopy(result)
        altered["eval_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            runner._verify_training_gate(self.config, altered, self.config_path)
        self.rows[0]["messages"][-1]["content"] += " Changed."
        write_jsonl(self.dataset, self.rows)
        with self.assertRaisesRegex(RuntimeError, "dataset"):
            runner._verify_training_gate(self.config, result, self.config_path)

    def test_output_and_report_cannot_overwrite_repository_data(self) -> None:
        root = runner.TRAINING_OUTPUT_ROOT
        for path in (root, self.dataset):
            self.assertFalse(runner._safe_output(path)[0])
        root.mkdir(parents=True)
        target = root / "already-exists"
        target.write_text("important", encoding="utf-8")
        self.assertFalse(runner._safe_output(target)[0])
        with self.assertRaises(RuntimeError):
            runner.main(["--config", str(self.config_path), "--dry-run", "--report", str(self.dataset)])

    def test_real_training_never_called_when_config_is_proposed(self) -> None:
        report_path = self.root / "training/reports/saved.json"
        write_json(report_path, self.dry_run())
        with mock.patch.object(runner, "execute_training") as train:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                runner.main(["--config", str(self.config_path), "--train", "--report", str(report_path), "--dry-run-report", str(report_path), "--confirm", runner.CONFIRMATION])
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
