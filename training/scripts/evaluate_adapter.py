#!/usr/bin/env python3
"""Evaluate a completed adapter in a new process, without training/optimizer state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_adapter as training


def _training_process_running(pid: int) -> bool:
    # This command targets the same WSL host as training. Fail closed even if
    # the recorded PID has since been reused by another process.
    return Path(f"/proc/{pid}").exists()


def completed_run(config: dict[str, Any], report: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Refuse failed/partial training, stale inputs, or an altered final adapter."""
    training._validate_config(config)
    output = training._under(training.ROOT / config["output"]["directory"], training.TRAINING_OUTPUT_ROOT)
    identity = training._run_identity(config, report)
    saved = json.loads(training._under(output / training.RUN_IDENTITY_FILE, output).read_text(encoding="utf-8"))
    if saved != identity:
        raise RuntimeError("Post-training evaluation requires the matching run identity")
    marker = json.loads(training._under(output / training.TRAINING_COMPLETE_FILE, output).read_text(encoding="utf-8"))
    if (
        marker.get("schema_version") != 1
        or marker.get("global_step") != config["training"]["estimated_optimizer_steps"]
        or marker.get("run_identity_sha256") != training.sha256_json(identity)
        or marker.get("training_config_sha256") != training.training_spec_digest(config)
    ):
        raise RuntimeError("Post-training evaluation requires verified natural training completion")
    if marker.get("training_pid") == os.getpid():
        raise RuntimeError("Post-training evaluation must run in a fresh process after training exits")
    if type(marker.get("training_pid")) is not int:
        raise RuntimeError("Completion record is missing its training process identity")
    if _training_process_running(marker["training_pid"]):
        raise RuntimeError("Training process is still running; wait for it to exit before evaluation")
    checkpoint = training._select_resume_checkpoint(output, identity)
    if checkpoint.name != f"checkpoint-{marker['global_step']}":
        raise RuntimeError("Post-training evaluation requires the verified final checkpoint")
    adapter_path = training._under(output / "adapter", output)
    if marker.get("adapter_files") != training._artifact_inventory(adapter_path, ("adapter_model.safetensors", "adapter_config.json")):
        raise RuntimeError("Completed adapter files changed")
    saved_config = training.load_config(training._under(output / "training_config.json", output))
    if training.training_spec_digest(saved_config) != training.training_spec_digest(config):
        raise RuntimeError("Completed training configuration changed")
    return adapter_path, marker


def evaluate(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    adapter_path, completion = completed_run(config, report)
    training._configure_compile_mode(config)
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    data = config["data"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=report["model_evidence"]["snapshot_path"], use_exact_model_name=True,
        fast_inference=False, max_seq_length=data["max_sequence_length"], dtype=torch.bfloat16,
        load_in_4bit=True, full_finetuning=False, offload_embedding=False,
        local_files_only=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    rows = training.load_jsonl(training.ROOT / data["corpus_path"])
    examples = training.expand_training_examples(rows, data["validation_split"])
    tokens = training.tokenize_training_examples(tokenizer, examples, data["max_sequence_length"])
    dataset = Dataset.from_list([
        {"input_ids": row["input_ids"], "attention_mask": [1] * len(row["input_ids"]),
         "labels": [token if mask else -100 for token, mask in zip(row["input_ids"], row["completion_mask"])]}
        for row in tokens
    ])
    # Trainer.evaluate never creates or restores an optimizer/scheduler. Loss-only
    # evaluation also avoids retaining full-vocabulary predictions between batches.
    trainer = Trainer(
        model=model, processing_class=tokenizer, eval_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=str(adapter_path.parent / "evaluation"), do_train=False, do_eval=True,
            eval_strategy="no", save_strategy="no", per_device_eval_batch_size=1,
            prediction_loss_only=True, eval_accumulation_steps=1, bf16=True, fp16=False,
            torch_empty_cache_steps=None, dataloader_num_workers=0, dataloader_pin_memory=False,
            report_to="none", push_to_hub=False, seed=config["training"]["seed"],
        ),
    )
    training._verify_training_runtime(training._training_runtime_settings(trainer, config["resources"]))
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise RuntimeError("Evaluation process must not contain an optimizer or scheduler")
    metrics = trainer.evaluate()
    return {
        "schema_version": 1, "artifact_type": "adapter_post_training_validation",
        "global_step": completion["global_step"], "evaluation_pid": os.getpid(),
        "run_identity_sha256": completion["run_identity_sha256"],
        "validation_examples": len(tokens), "metrics": metrics,
        "optimizer_loaded": False, "training_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    config = training.load_config(args.config.resolve())
    report = json.loads(training._under(args.dry_run_report, training.ROOT / "training/reports").read_text(encoding="utf-8"))
    target = training._under(args.report, training.ROOT / "training/reports")
    if target.exists():
        raise RuntimeError("Evaluation report already exists; choose a fresh report path")
    training._verify_training_gate(config, report, args.config, resume=True)
    # Verify completion before importing/loading the model; repeat the lightweight
    # target-machine preflight to detect changed runtime/model/data evidence.
    completed_run(config, report)
    fresh = training.dry_run(args.config, resume=True)
    if not fresh["passed"] or fresh["environment"] != report["environment"] or fresh["model_evidence"] != report["model_evidence"]:
        raise RuntimeError("Post-training evaluation preflight changed or failed")
    result = evaluate(config, fresh)
    training._atomic_json(target, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Post-training evaluation refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
