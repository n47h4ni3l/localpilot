from __future__ import annotations

import copy
import hashlib
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


def native_trace(identifier: str = "native-trace", split: str = "train") -> dict:
    row = record(identifier, "Look up the build, then summarize it.", split)
    row["metadata"] = {
        "native_tools": [{
            "name": "lookup_build",
            "description": "Read a build result.",
            "parameters": {
                "type": "object",
                "properties": {"build_id": {"type": "string"}},
                "required": ["build_id"],
            },
        }],
        "native_messages": [
            {"role": "user", "content": "Look up build 42, then summarize it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-42",
                    "type": "function",
                    "function": {"name": "lookup_build", "arguments": '{"build_id":"42"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call-42", "content": '{"status":"green"}'},
            {"role": "assistant", "content": "Build 42 is green."},
        ],
    }
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
        self.corpus_manifest_path = self.root / self.config["data"]["corpus_manifest"]
        write_json(self.corpus_manifest_path, {
            "artifact_type": "external_corpus_manifest", "records": 2,
            "split_counts": {"train": 1, "validation": 1},
            "training_example_counts": {"train": 1, "validation": 1, "total": 2},
            "corpus_sha256": hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
        })
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
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: list(range(len(messages) * 10))
        )
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

    def test_tracked_config_targets_external_corpus_and_epoch_cadence(self) -> None:
        self.assertEqual(self.config["data"]["corpus_path"], "training/datasets/external_corpus_v1.jsonl")
        self.assertEqual(self.config["data"]["corpus_manifest"], "training/manifests/external_corpus_v1_manifest.json")
        manifest_path = Path(__file__).resolve().parents[1] / "manifests/external_corpus_v1_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_examples = manifest["training_example_counts"]["train"]
        steps_per_epoch = (
            train_examples + self.config["training"]["effective_batch_size"] - 1
        ) // self.config["training"]["effective_batch_size"]
        self.assertEqual(self.config["training"]["estimated_optimizer_steps"], steps_per_epoch * self.config["training"]["epochs"])
        self.assertEqual(self.config["training"]["validation_steps"], steps_per_epoch)
        self.assertEqual(self.config["training"]["checkpoint_steps"], steps_per_epoch)

    def test_native_trace_expands_every_assistant_turn_and_preserves_tools(self) -> None:
        row = native_trace()
        original = copy.deepcopy(row)
        examples = runner.expand_training_examples([row])
        self.assertEqual(row, original)
        self.assertEqual(len(examples), 2)
        first, second = examples
        self.assertEqual(first["source_record_id"], "native-trace")
        self.assertEqual([first["assistant_turn"], second["assistant_turn"]], [0, 1])
        self.assertEqual(first["completion"][0]["content"], "")
        self.assertEqual(
            first["completion"][0]["tool_calls"][0]["function"]["arguments"],
            {"build_id": "42"},
        )
        self.assertEqual(second["prompt"][-1]["role"], "tool")
        self.assertEqual(second["completion"], [{"role": "assistant", "content": "Build 42 is green."}])
        for example in examples:
            self.assertEqual(example["tools"][0]["type"], "function")
            self.assertEqual(example["tools"][0]["function"]["name"], "lookup_build")

    def test_xlam_parallel_calls_become_independent_native_targets(self) -> None:
        row = record("xlam-example", "Book both legs.", "train")
        row["metadata"] = {
            "native_tools": [
                {"name": "book_flight", "description": "Book one leg", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
            ],
            "native_call_targets": [
                {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "book_flight", "arguments": {"city": "SYD"}}}]},
                {"function": {"name": "book_flight", "arguments": '{"city":"ADL"}'}},
            ],
        }
        examples = runner.expand_training_examples([row])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["prompt"], examples[1]["prompt"])
        self.assertEqual(examples[0]["prompt"], [{"role": "user", "content": "Book both legs."}])
        self.assertNotIn("A verified response.", json.dumps(examples))
        self.assertEqual(
            [item["completion"][0]["tool_calls"][0]["function"]["arguments"]["city"] for item in examples],
            ["SYD", "ADL"],
        )

    def test_native_expansion_rejects_unsupported_or_unrenderable_calls(self) -> None:
        for mutation, expected in (
            (lambda row: row["metadata"]["native_messages"][1]["tool_calls"].append(row["metadata"]["native_messages"][1]["tool_calls"][0]), "exactly one"),
            (lambda row: row["metadata"].update(native_tools=[]), "absent"),
            (lambda row: row["metadata"]["native_messages"][2].update(tool_call_id="wrong"), "does not match"),
            (lambda row: row["metadata"]["native_messages"][2].pop("tool_call_id"), "does not match"),
            (lambda row: row["metadata"]["native_messages"][2].update(name="wrong_tool"), "name does not match"),
            (lambda row: row["metadata"]["native_messages"][1].update(thinking="hidden"), "unsupported"),
        ):
            with self.subTest(expected=expected):
                row = native_trace()
                mutation(row)
                with self.assertRaisesRegex(RuntimeError, expected):
                    runner.expand_training_examples([row])

    def test_token_preflight_passes_tools_and_proves_prefix_and_completion(self) -> None:
        examples = runner.expand_training_examples([native_trace()])
        calls: list[tuple[list[dict], dict]] = []

        def render(messages: list[dict], **kwargs) -> list[int]:
            calls.append((messages, kwargs))
            base = [10, 11, 12]
            return base if kwargs["add_generation_prompt"] else base + [13, 14]

        counts = runner.validate_tokenized_examples(types.SimpleNamespace(apply_chat_template=render), examples, 32)
        self.assertEqual(counts, {
            "total": 10, "maximum": 5, "minimum_completion": 2,
            "training_examples": 2, "source_records": 1,
        })
        self.assertEqual(len(calls), 4)
        self.assertEqual([call[1]["add_generation_prompt"] for call in calls], [True, False, True, False])
        self.assertTrue(all(call[1]["truncation"] is False for call in calls))
        self.assertTrue(all(call[1]["tools"][0]["function"]["name"] == "lookup_build" for call in calls))

    def test_token_preflight_refuses_prefix_drift_empty_target_and_overlength(self) -> None:
        example = runner.expand_training_examples([record("basic-example", "Do it.", "train")])
        renderers = (
            (lambda messages, **kwargs: [1, 2] if kwargs["add_generation_prompt"] else [9, 3], "prefix"),
            (lambda messages, **kwargs: [1, 2], "empty"),
            (lambda messages, **kwargs: [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3, 4], "exceeds"),
        )
        for renderer, expected in renderers:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    runner.validate_tokenized_examples(types.SimpleNamespace(apply_chat_template=renderer), example, 3)

    def test_mocked_target_passes_without_training_or_implicit_download(self) -> None:
        report_path = self.root / "training/reports/custom.json"
        result = self.dry_run(report_path=report_path)
        self.assertTrue(result["passed"], [item for item in result["checks"] if not item["passed"]])
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["dataset_sha256"], sha256_json(self.rows))
        self.assertIn(str(report_path.resolve()), result["resolved_training_command"])
        self.assertEqual(self.importer.call_args_list[0].args, ("unsloth",))
        self.assertTrue(self.snapshot_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(self.tokenizer_loader.call_args.kwargs["local_files_only"])
        self.lora_loader.assert_called_once()
        self.assertEqual(result["model_evidence"]["token_counts"]["total"], 40)
        self.assertEqual(result["model_evidence"]["token_counts"]["training_examples"], 2)
        self.assertEqual(result["model_evidence"]["token_counts"]["minimum_completion"], 10)

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

    def test_manifest_freezes_expanded_training_example_counts(self) -> None:
        self.rows[0] = native_trace("example-train", "train")
        write_jsonl(self.dataset, self.rows)
        manifest = json.loads(self.corpus_manifest_path.read_text(encoding="utf-8"))
        manifest["corpus_sha256"] = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        write_json(self.corpus_manifest_path, manifest)
        result = self.dry_run()
        self.assertFalse(next(item for item in result["checks"] if item["name"] == "dataset_schema_and_splits")["passed"])
        self.importer.assert_not_called()

    def test_manifest_cannot_omit_expanded_training_example_counts(self) -> None:
        manifest = json.loads(self.corpus_manifest_path.read_text(encoding="utf-8"))
        del manifest["training_example_counts"]
        write_json(self.corpus_manifest_path, manifest)
        result = self.dry_run()
        check = next(item for item in result["checks"] if item["name"] == "dataset_schema_and_splits")
        self.assertFalse(check["passed"])
        self.assertIn("Expanded training-example counts", str(check["detail"]))
        self.importer.assert_not_called()

    def test_long_tokenized_example_is_refused_instead_of_truncated(self) -> None:
        self.tokenizer_loader.return_value.apply_chat_template = lambda *args, **kwargs: list(
            range(1024 if kwargs["add_generation_prompt"] else 1025)
        )
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

    def test_execute_training_uses_expanded_native_examples(self) -> None:
        rows = [native_trace("trace-train", "train"), record("plain-validation", "Check it.", "validation")]
        saved_model = types.SimpleNamespace(save_pretrained=mock.Mock())
        rendered_tools: list[list[dict]] = []

        def render(messages: list[dict], **kwargs) -> list[int]:
            rendered_tools.append(kwargs["tools"])
            return list(range(5 if kwargs["add_generation_prompt"] else 8))

        saved_tokenizer = types.SimpleNamespace(save_pretrained=mock.Mock(), apply_chat_template=render)
        fast_language_model = types.SimpleNamespace(
            from_pretrained=mock.Mock(return_value=(saved_model, saved_tokenizer)),
            get_peft_model=mock.Mock(return_value=saved_model),
        )
        dataset_from_list = mock.Mock(side_effect=lambda values: values)
        trainer = types.SimpleNamespace(train=mock.Mock())
        trainer_factory = mock.Mock(return_value=trainer)
        modules = {
            "unsloth": types.SimpleNamespace(FastLanguageModel=fast_language_model),
            "torch": types.SimpleNamespace(bfloat16=object()),
            "datasets": types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=dataset_from_list)),
            "trl": types.SimpleNamespace(SFTConfig=lambda **kwargs: kwargs, SFTTrainer=trainer_factory),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(runner, "load_jsonl", return_value=rows):
            runner.execute_training(self.config, str(self.snapshot))

        train_dataset = trainer_factory.call_args.kwargs["train_dataset"]
        validation_dataset = trainer_factory.call_args.kwargs["eval_dataset"]
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(validation_dataset), 1)
        self.assertTrue(all(item["completion_mask"] == [0] * 5 + [1] * 3 for item in train_dataset))
        self.assertTrue(all(item["input_ids"] == list(range(8)) for item in train_dataset))
        self.assertEqual([tools[0]["function"]["name"] for tools in rendered_tools[:4]], ["lookup_build"] * 4)
        self.assertEqual(rendered_tools[-2:], [[], []])
        trainer.train.assert_called_once_with()
        saved_model.save_pretrained.assert_called_once()
        saved_tokenizer.save_pretrained.assert_called_once()

    def test_real_training_never_called_when_config_is_proposed(self) -> None:
        report_path = self.root / "training/reports/saved.json"
        write_json(report_path, self.dry_run())
        with mock.patch.object(runner, "execute_training") as train:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                runner.main(["--config", str(self.config_path), "--train", "--report", str(report_path), "--dry-run-report", str(report_path), "--confirm", runner.CONFIRMATION])
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
