from __future__ import annotations

import copy
import io
import importlib.util
import json
import os
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from training.scripts import train_adapter as runner
from training.scripts import evaluate_adapter as evaluation
from training.tests import test_train_adapter as fixtures


class EnduranceRecoveryTests(unittest.TestCase):
    setUp = fixtures.TrainAdapterTests.setUp
    save_config = fixtures.TrainAdapterTests.save_config
    dry_run = fixtures.TrainAdapterTests.dry_run
    checkpoint_at_output = fixtures.TrainAdapterTests.checkpoint_at_output

    def recovery(self):
        self.config["backend"]["compile_mode"] = "native"
        self.config["training"].update(first_checkpoint_step=3704, checkpoint_steps=3704)
        self.config["diagnostics"] = {
            "enabled": True, "stop_after_step": None, "skip_final_adapter_save": False,
            "memory_log_path": "training/reports/source.jsonl", "exit_report_path": "training/reports/source.exit.json",
        }
        self.save_config()
        source_report = self.dry_run()
        self.assertTrue(source_report["passed"], source_report["checks"])
        source_report_path = self.root / "training/reports/source_dry_run.json"
        runner.write_json(source_report_path, source_report)
        source_identity = runner._run_identity(self.config, source_report)
        source_output = self.root / self.config["output"]["directory"]
        runner.write_json(source_output / runner.RUN_IDENTITY_FILE, source_identity)
        checkpoint = self.checkpoint_at_output(source_output, 3704, source_identity)
        self.checkpoint_at_output(source_output, 7408, source_identity, mark_complete=False)
        self.source_config = copy.deepcopy(self.config)
        self.config.update(name="Recovery", status="proposed", status_note="Recovery fixture")
        self.config["training"]["eval_strategy"] = "no"
        self.config["output"]["directory"] = "training/outputs/recovery"
        self.config["diagnostics"].update(memory_log_path="training/reports/recovery.jsonl", exit_report_path="training/reports/recovery.exit.json")
        self.config["recovery"] = {
            "source_report": "training/reports/source_dry_run.json",
            "source_report_sha256": runner._sha256_file(source_report_path),
            "source_identity_sha256": runner.sha256_json(source_identity), "checkpoint_step": 3704,
            "checkpoint_marker_sha256": runner._sha256_file(checkpoint / runner.CHECKPOINT_MARKER_FILE),
        }
        self.save_config()
        report = self.dry_run(recover=True)
        self.assertTrue(report["passed"], report["checks"])
        return checkpoint, runner._run_identity(self.config, report), report

    def test_recovery_uses_verified_3704_and_keeps_source_and_partial_7408_unchanged(self):
        checkpoint, identity, report = self.recovery()
        source_output = checkpoint.parent.parent
        source_bytes = {str(p.relative_to(source_output)): p.read_bytes() for p in source_output.rglob("*") if p.is_file()}
        self.assertEqual(runner._recovery_checkpoint(self.config, identity), checkpoint.resolve())
        self.assertIn("--recover", report["resolved_training_command"])
        with fixtures.stub_training_runtime() as (factory, backend), mock.patch.object(runner, "_mark_training_complete"), redirect_stdout(io.StringIO()):
            runner.execute_training(self.config, str(self.snapshot), run_identity=identity, resume_checkpoint=checkpoint, recover=True)
        factory.return_value.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint.resolve()))
        args = factory.return_value.instance.args
        self.assertEqual(args.eval_strategy, "no")
        self.assertFalse(args.do_eval)
        self.assertFalse(args.eval_on_start)
        self.assertFalse(args.load_best_model_at_end)
        self.assertIsNone(args.eval_steps)
        self.assertIsNone(factory.call_args.kwargs["eval_dataset"])
        self.assertIsNone(args.torch_empty_cache_steps)
        self.assertFalse(backend.from_pretrained.call_args.kwargs["offload_embedding"])
        settings = runner._training_runtime_settings(factory.return_value.instance, self.config["resources"])
        self.assertEqual(settings["embedding_devices"]["input"], "cuda:0")
        self.assertEqual(settings["embedding_devices"]["output"], "cuda:0")
        self.assertFalse(settings["embedding_devices"]["offload_active"])
        with self.assertRaisesRegex(RuntimeError, "In-process evaluation is disabled"):
            factory.return_value.instance.evaluate()
        self.assertEqual(source_bytes, {str(p.relative_to(source_output)): p.read_bytes() for p in source_output.rglob("*") if p.is_file()})
        output = self.root / self.config["output"]["directory"]
        self.assertEqual(json.loads((output / runner.RUN_IDENTITY_FILE).read_text()), identity)
        self.assertNotEqual(identity, json.loads((source_output / runner.RUN_IDENTITY_FILE).read_text()))

    def test_recovery_saves_7408_and_11112_without_evaluation_and_resumes_new_identity(self):
        checkpoint, identity, _ = self.recovery()
        output = self.root / self.config["output"]["directory"]
        with fixtures.stub_training_runtime() as (factory, _), mock.patch.object(runner, "_mark_training_complete"), redirect_stdout(io.StringIO()):
            runner.execute_training(self.config, str(self.snapshot), run_identity=identity, resume_checkpoint=checkpoint, recover=True)
        trainer = factory.return_value.instance
        self.assertEqual(trainer.args.save_strategy, "steps")
        self.assertEqual(trainer.args.save_steps, 3704)
        self.assertFalse(trainer.args.save_only_model)
        callback = trainer.callbacks[0]
        for step in (7408, 11112):
            saved = self.checkpoint_at_output(output, step, identity, mark_complete=False)
            callback.on_save(trainer.args, types.SimpleNamespace(global_step=step), types.SimpleNamespace())
            self.assertTrue((saved / runner.CHECKPOINT_MARKER_FILE).is_file())
            self.assertEqual(runner._select_resume_checkpoint(output, identity), saved.resolve())
        with self.assertRaisesRegex(RuntimeError, "implementation changed"):
            runner._select_resume_checkpoint(output, {**identity, "trainer_script_sha256": "changed"})

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Target-machine Transformers integration")
    def test_real_trainer_flow_saves_without_evaluation(self):
        from transformers import TrainingArguments, TrainerState, TrainerControl
        from transformers.trainer_callback import DefaultFlowCallback

        args = TrainingArguments(
            output_dir=str(self.root / "flow"), use_cpu=True, report_to="none",
            eval_strategy="no", do_eval=False, eval_on_start=False,
            save_strategy="steps", save_steps=3704, torch_empty_cache_steps=None,
        )
        state = TrainerState(max_steps=11112, save_steps=3704, eval_steps=3704)
        for step in (3704, 7408, 11112):
            state.global_step = step
            control = DefaultFlowCallback().on_step_end(args, state, TrainerControl())
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_evaluate)
        self.assertTrue(control.should_training_stop)

    def test_recovery_rejects_modified_source_checkpoint_or_missing_marker(self):
        checkpoint, identity, _ = self.recovery()
        (checkpoint / "optimizer.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "no complete matching checkpoint"):
            runner._recovery_checkpoint(self.config, identity)

    def test_recovery_refuses_changed_pins_and_runtime_evidence(self):
        _, identity, _ = self.recovery()
        for key in ("source_report_sha256", "source_identity_sha256", "checkpoint_marker_sha256"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.config)
                changed["recovery"][key] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    runner._recovery_checkpoint(changed, identity)
        for key in ("dataset_sha256", "corpus_manifest_sha256", "eval_manifest_sha256", "environment_sha256", "model_evidence_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "changed source evidence"):
                runner._recovery_checkpoint(self.config, {**identity, key: "changed"})

    def test_recovery_refuses_changes_to_stable_settings_or_source_paths(self):
        _, identity, _ = self.recovery()
        for section, key, value in (
            ("backend", "compile_mode", "eager"), ("training", "seed", 1),
            ("training", "learning_rate", 0.0002), ("training", "gradient_accumulation_steps", 2),
            ("training", "checkpoint_steps", 1), ("data", "max_sequence_length", 512),
            ("adapter", "rank", 16), ("resources", "offload_embeddings", True),
            ("output", "directory", self.source_config["output"]["directory"]),
            ("diagnostics", "exit_report_path", self.source_config["diagnostics"]["exit_report_path"]),
        ):
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(RuntimeError):
                    runner._recovery_checkpoint(changed, identity)

    def test_recovery_mode_never_silently_starts_fresh_or_restarts(self):
        self.recovery()
        for kwargs in ({}, {"restart": True}, {"resume": True, "recover": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.dry_run(**kwargs)
        self.config["recovery"]["checkpoint_step"] = 7408
        with self.assertRaisesRegex(RuntimeError, "checkpoint-3704"):
            runner._validate_config(self.config)

    def test_old_dry_run_is_not_approval_for_recovery(self):
        self.recovery()
        report = json.loads((self.root / "training/reports/source_dry_run.json").read_text())
        self.config["status"] = "approved_after_dry_run"
        with self.assertRaisesRegex(RuntimeError, "does not match current training settings"):
            runner._verify_training_gate(self.config, report, self.config_path)

    def test_runtime_rechecks_evaluation_and_checkpoint_settings(self):
        checkpoint, identity, _ = self.recovery()
        for key, value in (("eval_strategy", "steps"), ("eval_on_start", True), ("save_strategy", "no"), ("save_only_model", True)):
            with self.subTest(key=key):
                def rewrite(**kwargs):
                    return {**fixtures.stub_sft_config(**kwargs), key: value}
                with fixtures.stub_training_runtime(config_factory=rewrite) as (factory, _), redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Effective"):
                        runner.execute_training(self.config, str(self.snapshot), run_identity=identity, resume_checkpoint=checkpoint, recover=True)
                factory.return_value.train.assert_not_called()
                # Startup diagnostics are deliberately immutable for each attempt.
                for key in ("memory_log_path", "exit_report_path"):
                    (self.root / self.config["diagnostics"][key]).unlink(missing_ok=True)

    def completed(self):
        _, identity, report = self.recovery()
        output = self.root / self.config["output"]["directory"]
        runner.write_json(output / runner.RUN_IDENTITY_FILE, identity)
        self.checkpoint_at_output(output, 11112, identity)
        (output / "adapter").mkdir()
        (output / "adapter/adapter_model.safetensors").write_bytes(b"adapter")
        (output / "adapter/adapter_config.json").write_text("{}")
        runner.write_json(output / "training_config.json", self.config)
        runner._mark_training_complete(output, self.config, identity, 11112)
        return output, report

    def test_post_eval_requires_completion_fresh_process_and_unmodified_adapter(self):
        output, report = self.completed()
        with mock.patch.object(evaluation, "training", runner):
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                evaluation.completed_run(self.config, report)
            with mock.patch.object(evaluation.os, "getpid", return_value=os.getpid() + 1), mock.patch.object(evaluation, "_training_process_running", return_value=False):
                self.assertEqual(evaluation.completed_run(self.config, report)[0], output / "adapter")
                (output / "adapter/adapter_model.safetensors").write_bytes(b"altered")
                with self.assertRaisesRegex(RuntimeError, "adapter files changed"):
                    evaluation.completed_run(self.config, report)
        (output / runner.TRAINING_COMPLETE_FILE).unlink()
        with mock.patch.object(evaluation, "training", runner), self.assertRaises(FileNotFoundError):
            evaluation.completed_run(self.config, report)

    def test_partial_training_cannot_publish_completion(self):
        _, identity, _ = self.recovery()
        output = self.root / self.config["output"]["directory"]
        with self.assertRaisesRegex(RuntimeError, "before natural completion"):
            runner._mark_training_complete(output, self.config, identity, 7408)
        self.assertFalse((output / runner.TRAINING_COMPLETE_FILE).exists())

    def test_post_eval_loads_only_adapter_and_evaluates_loss_without_optimizer(self):
        output, report = self.completed()
        model = fixtures.embedding_model()
        model.config = types.SimpleNamespace(use_cache=True)
        model.eval = mock.Mock()
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kw: list(range(5 if kw["add_generation_prompt"] else 8)))
        backend = mock.Mock(return_value=(model, tokenizer))
        peft = mock.Mock(return_value=model)
        evaluated = types.SimpleNamespace(optimizer=None, lr_scheduler=None, evaluate=mock.Mock(return_value={"eval_loss": 0.5}))
        def trainer_factory(**kwargs):
            evaluated.model = model
            evaluated.args = types.SimpleNamespace(**kwargs["args"])
            return evaluated
        trainer = mock.Mock(side_effect=trainer_factory)
        modules = {
            "unsloth": types.SimpleNamespace(FastLanguageModel=types.SimpleNamespace(from_pretrained=backend)),
            "torch": types.SimpleNamespace(bfloat16=object()),
            "peft": types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=peft)),
            "datasets": types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda values: values)),
            "transformers": types.SimpleNamespace(Trainer=trainer, TrainingArguments=lambda **kw: kw, DataCollatorForSeq2Seq=mock.Mock()),
        }
        with mock.patch.object(evaluation, "training", runner), mock.patch.object(evaluation.os, "getpid", return_value=os.getpid() + 1), mock.patch.object(evaluation, "_training_process_running", return_value=False), mock.patch.dict("sys.modules", modules):
            result = evaluation.evaluate(self.config, report)
        self.assertFalse(peft.call_args.kwargs["is_trainable"])
        self.assertEqual(peft.call_args.args[1], str(output / "adapter"))
        self.assertNotIn("train_dataset", trainer.call_args.kwargs)
        self.assertTrue(evaluated.args.prediction_loss_only)
        self.assertIsNone(evaluated.args.torch_empty_cache_steps)
        self.assertEqual(trainer.call_args.kwargs["eval_dataset"][0]["labels"], [-100] * 5 + [5, 6, 7])
        evaluated.evaluate.assert_called_once_with()
        self.assertFalse(result["optimizer_loaded"])
        self.assertFalse(result["training_performed"])

    def test_tracked_recovery_preserves_all_stable_settings(self):
        root = Path(__file__).resolve().parents[1]
        source = runner.load_config(root / "configs/qlora_v1_native_compile_endurance.yaml")
        recovered = runner.load_config(root / "configs/qlora_v1_native_compile_recovery.yaml")
        for section in ("model", "data", "adapter", "quantization", "resources", "backend", "promotion"):
            self.assertEqual(source[section], recovered[section])
        expected_training = {**source["training"], "eval_strategy": "no"}
        self.assertEqual(expected_training, recovered["training"])
        self.assertEqual(recovered["recovery"]["checkpoint_step"], 3704)
        self.assertEqual(recovered["backend"]["compile_mode"], "native")
        self.assertFalse(recovered["resources"]["offload_embeddings"])
        self.assertEqual(recovered["resources"]["cpu_offload"], "none")


if __name__ == "__main__":
    unittest.main()
