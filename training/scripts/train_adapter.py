#!/usr/bin/env python3
"""Reproducible, non-training preflight and explicitly gated Unsloth adapter run.

No model weights are allocated on the GPU during dry-run. It validates the local
snapshot, tokenizer, PEFT config and small ROCm operations; memory remains an
estimate until a separately authorized training run measures its actual peak.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import ROOT, find_manifest_leaks, load_eval_manifest, load_jsonl, sha256_json, write_json
from validate_dataset import validate_files


DEFAULT_CONFIG = ROOT / "training/configs/qlora_v1.yaml"
DEFAULT_REPORT = ROOT / "training/reports/adapter_v1_dry_run.json"
TRAINING_OUTPUT_ROOT = ROOT / "training/outputs"
RUN_IDENTITY_FILE = "run_identity.json"
CHECKPOINT_MARKER_FILE = "localpilot_checkpoint_complete.json"
CHECKPOINT_REQUIRED_FILES = (
    "adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "rng_state.pth",
)
# Measure clean GPU capacity after Torch/Triton initialize but before Unsloth's
# ROCm patching changes the process-local free-memory reading. Unsloth still
# loads before Transformers/PEFT, as required by its import contract.
GPU_PREFLIGHT_IMPORTS = ("torch", "triton")
REQUIRED_IMPORTS = GPU_PREFLIGHT_IMPORTS + (
    "unsloth", "unsloth_zoo", "transformers", "peft", "trl",
    "datasets", "bitsandbytes", "huggingface_hub",
)
CONFIRMATION = "TRAIN_LOCALPILOT_ADAPTER_V1"
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _configure_compile_mode(config: dict[str, Any]) -> None:
    """Apply the pinned eager mode before Torch and Unsloth are imported."""
    if config["backend"].get("compile_mode") != "eager":
        raise RuntimeError("This machine requires the measured eager training path")
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


def _adapter_only_state_dict(model: Any) -> dict[str, Any]:
    """Avoid traversing quantized base weights when PEFT writes a checkpoint."""
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only nonempty LoRA parameters may be checkpointed")
    return {name: parameter.detach().cpu() for name, parameter in trainable}


def _native_tool(tool: Any, *, record_id: str) -> dict[str, Any]:
    """Return one tool in the OpenAI shape expected by the gpt-oss template."""
    if not isinstance(tool, dict):
        raise RuntimeError(f"{record_id}: native tool must be an object")
    function = tool.get("function", tool)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
        raise RuntimeError(f"{record_id}: native tool requires a function name")
    parameters = function.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise RuntimeError(f"{record_id}: native tool parameters must be an object")
    normalized_parameters = copy.deepcopy(parameters)
    properties = normalized_parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise RuntimeError(f"{record_id}: native tool parameter properties must be an object")
    for parameter_name, parameter_schema in properties.items():
        if not isinstance(parameter_name, str) or not parameter_name.strip():
            raise RuntimeError(f"{record_id}: native tool parameter names must be nonempty text")
        if not isinstance(parameter_schema, dict):
            raise RuntimeError(f"{record_id}: native tool parameter schemas must be objects")
        # The pinned gpt-oss template concatenates every top-level parameter
        # description directly. Some valid upstream JSON Schemas omit this
        # optional annotation, so supply an empty rendering-only default without
        # mutating the signed corpus row.
        parameter_schema["description"] = str(parameter_schema.get("description", ""))
    normalized_function = copy.deepcopy(function)
    normalized_function["name"] = function["name"].strip()
    normalized_function["description"] = str(function.get("description", ""))
    normalized_function["parameters"] = normalized_parameters
    return {"type": "function", "function": normalized_function}


def _native_call(call: Any, *, record_id: str) -> dict[str, Any]:
    """Normalize a native call without retaining provider-specific hidden fields."""
    if not isinstance(call, dict):
        raise RuntimeError(f"{record_id}: native tool call must be an object")
    function = call.get("function", call)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
        raise RuntimeError(f"{record_id}: native tool call requires a function name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as exc:
            raise RuntimeError(f"{record_id}: native tool-call arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise RuntimeError(f"{record_id}: native tool-call arguments must be an object")
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {"name": function["name"].strip(), "arguments": copy.deepcopy(arguments)},
    }
    if isinstance(call.get("id"), str) and call["id"].strip():
        normalized["id"] = call["id"].strip()
    return normalized


def _native_message(message: Any, *, record_id: str) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
        raise RuntimeError(f"{record_id}: malformed native message")
    unknown = set(message) - {"role", "content", "name", "tool_call_id", "tool_calls"}
    if unknown:
        raise RuntimeError(f"{record_id}: unsupported native message fields: {', '.join(sorted(unknown))}")
    role = message["role"]
    content = message.get("content", "")
    # The pinned gpt-oss template checks membership in assistant content and cannot
    # consume None, even though None is conventional in OpenAI tool-call messages.
    if content is None and role == "assistant" and message.get("tool_calls"):
        content = ""
    if not isinstance(content, str):
        raise RuntimeError(f"{record_id}: native message content must be text")
    normalized: dict[str, Any] = {"role": role, "content": content}
    for key in ("name", "tool_call_id"):
        if key in message:
            if not isinstance(message[key], str) or not message[key].strip():
                raise RuntimeError(f"{record_id}: native message {key} must be nonempty text")
            normalized[key] = message[key].strip()
    calls = message.get("tool_calls")
    if calls is not None:
        if role != "assistant" or not isinstance(calls, list) or len(calls) != 1:
            raise RuntimeError(f"{record_id}: gpt-oss requires exactly one tool call per assistant message")
        normalized["tool_calls"] = [_native_call(calls[0], record_id=record_id)]
    if role == "assistant":
        if not content.strip() and "tool_calls" not in normalized:
            raise RuntimeError(f"{record_id}: assistant training target is empty")
    elif not content.strip():
        raise RuntimeError(f"{record_id}: native {role} message content is empty")
    return normalized


def _target_message(target: Any, *, record_id: str) -> dict[str, Any]:
    """Accept the canonical assistant-message target and legacy bare call shape."""
    if isinstance(target, dict) and target.get("role") == "assistant":
        return _native_message(target, record_id=record_id)
    if isinstance(target, dict) and "tool_calls" in target:
        return _native_message({"role": "assistant", "content": target.get("content", ""), "tool_calls": target["tool_calls"]}, record_id=record_id)
    return _native_message({"role": "assistant", "content": "", "tool_calls": [target]}, record_id=record_id)


def _validate_native_sequence(messages: list[dict[str, Any]], *, record_id: str, require_resolved_call: bool) -> None:
    outstanding_id: str | None = None
    outstanding_name: str | None = None
    outstanding = False
    seen_ids: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            if outstanding:
                raise RuntimeError(f"{record_id}: assistant message appears before the outstanding tool result")
            calls = message.get("tool_calls", [])
            if calls:
                outstanding = True
                outstanding_id = calls[0].get("id")
                outstanding_name = calls[0]["function"]["name"]
                if outstanding_id:
                    if outstanding_id in seen_ids:
                        raise RuntimeError(f"{record_id}: native tool-call ids must be unique")
                    seen_ids.add(outstanding_id)
        elif message["role"] == "tool":
            if not outstanding:
                raise RuntimeError(f"{record_id}: tool result has no preceding native tool call")
            result_id = message.get("tool_call_id")
            if outstanding_id and result_id != outstanding_id:
                raise RuntimeError(f"{record_id}: tool result id does not match its native tool call")
            result_name = message.get("name")
            if result_name and outstanding_name and result_name != outstanding_name:
                raise RuntimeError(f"{record_id}: tool result name does not match its native tool call")
            outstanding = False
            outstanding_id = None
            outstanding_name = None
        elif outstanding:
            raise RuntimeError(f"{record_id}: native tool call is not followed by its tool result")
    if outstanding and require_resolved_call:
        raise RuntimeError(f"{record_id}: native tool call is missing its tool result")


def expand_training_examples(rows: Sequence[dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    """Expand records into one prompt/completion example per assistant target.

    Native agent traces retain structured calls/results and their tool schemas. xLAM
    parallel calls are represented as independent single-call targets because the
    pinned gpt-oss template deliberately supports at most one call per message.
    """
    expanded: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise RuntimeError("Training record requires a nonempty id")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{record_id}: metadata must be an object")

        native_tools_value = metadata.get("native_tools")
        if "native_tools" in metadata and not isinstance(native_tools_value, list):
            raise RuntimeError(f"{record_id}: native_tools must be a list")
        tools = [_native_tool(tool, record_id=record_id) for tool in (native_tools_value or [])]
        tool_names = {tool["function"]["name"] for tool in tools}
        if len(tool_names) != len(tools):
            raise RuntimeError(f"{record_id}: native tool names must be unique")

        native_messages_value = metadata.get("native_messages")
        if "native_messages" in metadata:
            if not isinstance(native_messages_value, list) or not native_messages_value:
                raise RuntimeError(f"{record_id}: native_messages must be a nonempty list")
            messages = [_native_message(message, record_id=record_id) for message in native_messages_value]
        else:
            value = row.get("messages")
            if not isinstance(value, list) or not value:
                raise RuntimeError(f"{record_id}: messages must be a nonempty list")
            messages = [_native_message(message, record_id=record_id) for message in value]

        message_call_names = {
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls", [])
        }
        if not message_call_names.issubset(tool_names):
            raise RuntimeError(f"{record_id}: native message calls a tool absent from native_tools")

        call_targets = metadata.get("native_call_targets")
        _validate_native_sequence(
            messages,
            record_id=record_id,
            require_resolved_call="native_messages" in metadata and "native_call_targets" not in metadata,
        )
        if "native_call_targets" in metadata:
            if not isinstance(call_targets, list) or not call_targets:
                raise RuntimeError(f"{record_id}: native_call_targets must be a nonempty list")
            first_assistant = next((index for index, message in enumerate(messages) if message["role"] == "assistant"), len(messages))
            prompt = messages[:first_assistant]
            if not prompt or not any(message["role"] == "user" for message in prompt):
                raise RuntimeError(f"{record_id}: native call targets require a user prompt")
            for turn, target in enumerate(call_targets):
                completion = _target_message(target, record_id=record_id)
                called_name = completion["tool_calls"][0]["function"]["name"]
                if called_name not in tool_names:
                    raise RuntimeError(f"{record_id}: native call target names a tool absent from native_tools")
                example: dict[str, Any] = {
                    "prompt": copy.deepcopy(prompt),
                    "completion": [completion],
                    "tools": copy.deepcopy(tools),
                    "source_record_id": record_id,
                    "assistant_turn": turn,
                }
                expanded.append(example)
            continue

        assistant_turn = 0
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = messages[:index]
            if not prompt or not any(item["role"] == "user" for item in prompt):
                raise RuntimeError(f"{record_id}: assistant target has no preceding user prompt")
            example = {
                "prompt": copy.deepcopy(prompt),
                "completion": [copy.deepcopy(message)],
                "tools": copy.deepcopy(tools),
                "source_record_id": record_id,
                "assistant_turn": assistant_turn,
            }
            expanded.append(example)
            assistant_turn += 1
        if assistant_turn == 0:
            raise RuntimeError(f"{record_id}: no assistant training targets")
    if not expanded:
        raise RuntimeError("No assistant training examples were produced")
    return expanded


def _template_tokens(tokenizer: Any, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, add_generation_prompt: bool) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "truncation": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    tokens = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not isinstance(tokens, list) or (tokens and not isinstance(tokens[0], int)):
        raise RuntimeError("Tokenizer returned an unexpected token structure")
    return tokens


def _tokenized_example(tokenizer: Any, example: dict[str, Any], max_sequence_length: int) -> tuple[list[int], int]:
    tools = example.get("tools")
    prompt = _template_tokens(tokenizer, example["prompt"], tools=tools, add_generation_prompt=True)
    full = _template_tokens(
        tokenizer,
        example["prompt"] + example["completion"],
        tools=tools,
        add_generation_prompt=False,
    )
    if not prompt or full[: len(prompt)] != prompt:
        raise RuntimeError(f"{example['source_record_id']}: tokenized prompt is not a prefix of the full target")
    completion_length = len(full) - len(prompt)
    if completion_length < 1:
        raise RuntimeError(f"{example['source_record_id']}: tokenized completion is empty")
    if len(full) > max_sequence_length:
        raise RuntimeError(
            f"{example['source_record_id']}: tokenized example exceeds {max_sequence_length} tokens ({len(full)})"
        )
    return full, len(prompt)


def validate_tokenized_examples(tokenizer: Any, examples: Sequence[dict[str, Any]], max_sequence_length: int) -> dict[str, int]:
    """Prove every completion is nonempty, prefix-aligned, and untruncated."""
    lengths: list[int] = []
    completion_lengths: list[int] = []
    for example in examples:
        full, prompt_length = _tokenized_example(tokenizer, example, max_sequence_length)
        completion_length = len(full) - prompt_length
        lengths.append(len(full))
        completion_lengths.append(completion_length)
    if not lengths:
        raise RuntimeError("No tokenized training examples were validated")
    return {
        "total": sum(lengths),
        "maximum": max(lengths),
        "minimum_completion": min(completion_lengths),
        "training_examples": len(lengths),
        "source_records": len({example["source_record_id"] for example in examples}),
    }


def tokenize_training_examples(tokenizer: Any, examples: Sequence[dict[str, Any]], max_sequence_length: int) -> list[dict[str, list[int]]]:
    """Render before Arrow ingestion so heterogeneous JSON tool schemas stay exact."""
    tokenized: list[dict[str, list[int]]] = []
    for example in examples:
        full, prompt_length = _tokenized_example(tokenizer, example, max_sequence_length)
        tokenized.append({
            "input_ids": full,
            "completion_mask": [0] * prompt_length + [1] * (len(full) - prompt_length),
        })
    if not tokenized:
        raise RuntimeError("No tokenized training examples were produced")
    return tokenized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Config must be UTF-8 JSON syntax (a valid YAML subset)") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Training config must be an object")
    return value


def training_spec_digest(config: dict[str, Any]) -> str:
    """Approving status alone does not invalidate the tested training settings."""
    return sha256_json({key: value for key, value in config.items() if key not in {"status", "status_note"}})


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _under(path: Path, root: Path) -> Path:
    for component in [path, *path.parents]:
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise RuntimeError("Training paths must not use symlinks or junctions")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path must remain under {root}") from exc
    if resolved == root.resolve():
        raise RuntimeError(f"Path must be a child of {root}")
    return resolved


def _safe_output(path: Path, *, resume: bool = False) -> tuple[bool, str]:
    try:
        resolved = _under(path, TRAINING_OUTPUT_ROOT)
        if resolved.exists() and not resolved.is_dir():
            raise RuntimeError("Output must be a directory")
        if resume:
            if not resolved.is_dir():
                raise RuntimeError("Resume requires an existing output directory")
        elif resolved.exists():
            entries = list(resolved.iterdir())
            # A failed pre-checkpoint run can leave this otherwise empty directory.
            if entries and not (
                len(entries) == 1 and entries[0].name == "checkpoints"
                and entries[0].is_dir() and not entries[0].is_symlink()
                and not (hasattr(entries[0], "is_junction") and entries[0].is_junction())
                and not any(entries[0].iterdir())
            ):
                raise RuntimeError("Fresh training requires an absent or empty output directory")
        return True, str(resolved)
    except (OSError, RuntimeError) as exc:
        return False, str(exc)


def _run_identity(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Bind local checkpoints to the exact approved inputs and implementation."""
    return {
        "schema_version": 1,
        "training_spec_sha256": training_spec_digest(config),
        "dataset_sha256": report["dataset_sha256"],
        "corpus_manifest_sha256": report["corpus_manifest_sha256"],
        "eval_manifest_sha256": report["eval_manifest_sha256"],
        "model_evidence_sha256": sha256_json(report["model_evidence"]),
        "environment_sha256": sha256_json(report["environment"]),
        "trainer_script_sha256": _sha256_file(Path(__file__)),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a completion marker only after its full contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_inventory(checkpoint: Path, step: int, output: Path) -> dict[str, dict[str, Any]]:
    _under(checkpoint, output / "checkpoints")
    trainer_state = checkpoint / "trainer_state.json"
    inventory: dict[str, dict[str, Any]] = {}
    for name in CHECKPOINT_REQUIRED_FILES:
        path = _under(checkpoint / name, checkpoint)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Incomplete checkpoint-{step}: missing {name}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    try:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid checkpoint-{step} trainer state") from exc
    if type(state.get("global_step")) is not int or state["global_step"] != step:
        raise RuntimeError(f"Checkpoint-{step} trainer step mismatch")
    return inventory


def _mark_checkpoint_complete(checkpoint: Path, step: int, output: Path, identity: dict[str, Any]) -> None:
    inventory = _checkpoint_inventory(checkpoint, step, output)
    _atomic_json(checkpoint / CHECKPOINT_MARKER_FILE, {
        "schema_version": 1,
        "global_step": step,
        "run_identity_sha256": sha256_json(identity),
        "files": inventory,
    })
    print(f"LocalPilot checkpoint verified: step {step} ({checkpoint})", flush=True)


def _validate_restart_output(output: Path, identity: dict[str, Any]) -> None:
    """Require an explicit restart when the same run stopped before its first save."""
    safe, detail = _safe_output(output, resume=True)
    if not safe:
        raise RuntimeError(detail)
    identity_path = _under(output / RUN_IDENTITY_FILE, output)
    if not identity_path.is_file():
        raise RuntimeError("Restart requires a saved LocalPilot run identity")
    try:
        saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Saved LocalPilot run identity is invalid") from exc
    if saved_identity != identity:
        raise RuntimeError("Restart refused: training inputs or implementation changed")
    for entry in output.iterdir():
        if entry.name == RUN_IDENTITY_FILE:
            continue
        if entry.name == "checkpoints":
            checkpoint_root = _under(entry, output)
            if checkpoint_root.is_dir() and not any(checkpoint_root.iterdir()):
                continue
        raise RuntimeError("Restart refused: output has a checkpoint or other training data; preserve it for review")


def _select_resume_checkpoint(output: Path, identity: dict[str, Any]) -> Path:
    safe, detail = _safe_output(output, resume=True)
    if not safe:
        raise RuntimeError(detail)
    identity_path = _under(output / RUN_IDENTITY_FILE, output)
    if not identity_path.is_file():
        raise RuntimeError("Resume requires a saved LocalPilot run identity")
    try:
        saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Saved LocalPilot run identity is invalid") from exc
    if saved_identity != identity:
        raise RuntimeError("Resume refused: training inputs or implementation changed")
    checkpoint_root = _under(output / "checkpoints", output)
    if not checkpoint_root.is_dir():
        raise RuntimeError("Resume requires a checkpoint directory")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for step, path in sorted(candidates, reverse=True):
        try:
            inventory = _checkpoint_inventory(path, step, output)
            marker_path = _under(path / CHECKPOINT_MARKER_FILE, path)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker != {
                "schema_version": 1,
                "global_step": step,
                "run_identity_sha256": sha256_json(identity),
                "files": inventory,
            }:
                raise RuntimeError("completion marker mismatch")
        except (OSError, RuntimeError, ValueError):
            # A power loss can interrupt the newest checkpoint. Retain it for
            # diagnosis and use the previous verified checkpoint, if any.
            continue
        return path
    raise RuntimeError("Resume refused: no complete matching checkpoint was found")


def _is_wsl() -> bool:
    try:
        return platform.system() == "Linux" and "microsoft" in Path("/proc/version").read_text().casefold()
    except OSError:
        return False


def _os_release() -> dict[str, str]:
    try:
        return dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().replace('"', "").splitlines() if "=" in line)
    except OSError:
        return {}


def _system_ram_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        try:
            import psutil

            return psutil.virtual_memory().total / 1024**3
        except ImportError:
            return None


def _free_storage_gib(path: Path) -> float:
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def _validate_config(config: dict[str, Any]) -> None:
    for section in ("backend", "model", "adapter", "data", "quantization", "training", "resources", "output", "promotion"):
        if not isinstance(config.get(section), dict):
            raise RuntimeError(f"Missing config section: {section}")
    if config.get("status") not in {"proposed", "approved_after_dry_run"}:
        raise RuntimeError("Config status must be proposed or approved_after_dry_run")
    backend, model, adapter = config["backend"], config["model"], config["adapter"]
    data, quant, training, resources = (config[key] for key in ("data", "quantization", "training", "resources"))
    if backend.get("name") != "unsloth" or not isinstance(backend.get("package_versions"), dict):
        raise RuntimeError("Unsloth backend and explicit package pins are required")
    if backend.get("compile_mode") != "eager":
        raise RuntimeError("The measured training path requires eager mode")
    if model.get("base_identity") != "openai/gpt-oss-20b" or model.get("use_exact_model_name") is not True:
        raise RuntimeError("Require the gpt-oss-20b base and disable implicit model remapping")
    for key in ("revision", "base_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get(key, ""))):
            raise RuntimeError(f"Model {key} must be an immutable 40-character commit")
    if adapter.get("type") != "qlora" or set(adapter.get("target_modules", [])) != TARGET_MODULES:
        raise RuntimeError("Adapter must use the documented gpt-oss QLoRA target modules")

    def number(section: dict[str, Any], name: str, minimum: float, maximum: float, *, integer: bool = False) -> None:
        value = section.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"{name} must be a finite number")
        if not minimum <= value <= maximum or (integer and not isinstance(value, int)):
            raise RuntimeError(f"Invalid {name}: {value}")

    number(adapter, "rank", 1, 128, integer=True)
    number(adapter, "alpha", 1, 256, integer=True)
    number(adapter, "dropout", 0, 0.5)
    number(data, "max_sequence_length", 256, 4096, integer=True)
    for key in ("micro_batch_size", "gradient_accumulation_steps", "validation_steps", "first_checkpoint_step", "checkpoint_steps", "logging_steps", "save_total_limit"):
        number(training, key, 1, 10000, integer=True)
    if training["first_checkpoint_step"] > training["checkpoint_steps"]:
        raise RuntimeError("The first checkpoint must not come after the regular checkpoint cadence")
    number(training, "learning_rate", 1e-7, 1e-2)
    number(training, "epochs", 0.01, 100)
    number(training, "warmup_ratio", 0, 1)
    number(training, "weight_decay", 0, 1)
    number(training, "seed", 0, 2**32 - 1, integer=True)
    if training.get("max_steps") is not None:
        number(training, "max_steps", 1, 100000, integer=True)
    if training.get("effective_batch_size") != training["micro_batch_size"] * training["gradient_accumulation_steps"]:
        raise RuntimeError("Effective batch size must equal micro batch times accumulation")
    if training.get("precision") != "bf16" or training.get("optimizer") != "adamw_8bit":
        raise RuntimeError("The verified proposal requires BF16 compute and adamw_8bit")
    if quant.get("load_in_4bit") is not True or quant.get("type") != "nf4" or quant.get("double_quantization") is not True or quant.get("compute_dtype") != "bfloat16":
        raise RuntimeError("Require NF4 double quantization with BF16 compute")
    if data.get("packing") is not False or data.get("train_split") != "train" or data.get("validation_split") != "validation":
        raise RuntimeError("Require unpacked train and validation splits")
    if not isinstance(data.get("corpus_manifest"), str) or not data["corpus_manifest"]:
        raise RuntimeError("A frozen corpus manifest is required")
    for key in ("minimum_vram_gib", "estimated_peak_vram_gib", "minimum_system_ram_gib", "minimum_storage_free_gib"):
        number(resources, key, 1, 10000)
    if resources.get("offload_embeddings") is not True or resources.get("cpu_offload") != "embeddings_only":
        raise RuntimeError("The recovery training specification requires embedding offload to CPU")
    promotion = config["promotion"]
    if promotion.get("requires_eval_v1") is not True or promotion.get("requires_evolution_execution") is not True or promotion.get("training_loss_is_sufficient") is not False:
        raise RuntimeError("Held-out and execution promotion gates must remain enabled")


def _weight_inventory(snapshot: Path) -> dict[str, Any]:
    """Check every safetensors shard is complete without loading any tensor."""
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    if not names:
        raise RuntimeError("Model index has no weight shards")
    inventory = []
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise RuntimeError("Unsafe model shard name")
        path = snapshot / name
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 2 <= header_size <= 32 * 1024 * 1024:
                raise RuntimeError(f"Invalid safetensors header: {name}")
            header = json.loads(handle.read(header_size))
        ends = [item["data_offsets"][1] for key, item in header.items() if key != "__metadata__"]
        if not ends or path.stat().st_size != 8 + header_size + max(ends):
            raise RuntimeError(f"Missing or truncated model shard: {name}")
        inventory.append({"file": name, "bytes": path.stat().st_size})
    return {"shards": inventory, "total_bytes": sum(item["bytes"] for item in inventory)}


def _model_preflight(config: dict[str, Any], rows: list[dict[str, Any]], modules: dict[str, Any], allow_downloads: bool) -> dict[str, Any]:
    model = config["model"]
    cache = Path(model["cache_directory"]).expanduser().resolve() / "hub"
    snapshot = Path(modules["huggingface_hub"].snapshot_download(
        repo_id=model["training_model_id"], revision=model["revision"], cache_dir=str(cache),
        local_files_only=not allow_downloads,
        allow_patterns=["*.json", "*.jinja", "*.model", "*.tiktoken", "*.txt", "*.safetensors"],
    )).resolve()
    inventory = _weight_inventory(snapshot)
    transformers = modules["transformers"]
    model_config = transformers.AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if getattr(model_config, "model_type", None) != "gpt_oss":
        raise RuntimeError("Pinned snapshot is not a gpt_oss model")
    quant = getattr(model_config, "quantization_config", {})
    if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or quant.get("bnb_4bit_use_double_quant") is not True:
        raise RuntimeError("Snapshot does not match the proposed prequantized NF4 weights")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    examples = expand_training_examples(rows)
    token_counts = validate_tokenized_examples(tokenizer, examples, config["data"]["max_sequence_length"])
    adapter = config["adapter"]
    modules["peft"].LoraConfig(
        task_type="CAUSAL_LM", r=adapter["rank"], lora_alpha=adapter["alpha"],
        lora_dropout=adapter["dropout"], bias=adapter["bias"], target_modules=adapter["target_modules"],
    )
    return {
        "model_id": model["training_model_id"], "revision": model["revision"],
        "snapshot_path": str(snapshot), "weight_inventory": inventory,
        "token_counts": token_counts,
        "adapter_config_validated": True,
    }


def _gpu_preflight(torch: Any, triton: Any, config: dict[str, Any]) -> dict[str, Any]:
    hip = str(getattr(torch.version, "hip", "") or "")
    if not hip or not torch.cuda.is_available():
        raise RuntimeError("A working ROCm PyTorch GPU is required")
    if str(torch.__version__) != config["backend"]["pytorch_version"]:
        raise RuntimeError(f"Unexpected PyTorch build: {torch.__version__}")
    if not hip.startswith(config["backend"]["rocm_version"].rsplit(".", 1)[0] + "."):
        raise RuntimeError(f"Unexpected HIP build: {hip}")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", ""))
    if not arch.startswith("gfx1201"):
        raise RuntimeError(f"Expected RX 9070 gfx1201; detected {arch or 'unknown'}")
    free, total = torch.cuda.mem_get_info(0)
    required = max(config["resources"]["minimum_vram_gib"], config["resources"]["estimated_peak_vram_gib"])
    if min(free, total) / 1024**3 < required:
        raise RuntimeError(f"Need {required} GiB free VRAM; detected {free / 1024**3:.2f}; close GPU workloads")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 compute is unavailable")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        raise RuntimeError(f"Triton backend must be hip, detected {target.backend}")
    probe = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    result = (probe @ probe).float().sum().item()
    torch.cuda.synchronize()
    if result != 4096:
        raise RuntimeError("ROCm BF16 matrix-operation smoke check failed")
    del probe
    torch.cuda.empty_cache()
    return {"name": props.name, "architecture": arch, "hip": hip, "torch": str(torch.__version__), "triton": str(triton.__version__), "free_vram_gib": round(free / 1024**3, 3), "total_vram_gib": round(total / 1024**3, 3)}


def dry_run(config_path: Path, *, allow_downloads: bool = False, resume: bool = False, restart: bool = False, report_path: Path = DEFAULT_REPORT, importer: Callable[[str], Any] = importlib.import_module) -> dict[str, Any]:
    if resume and restart:
        raise RuntimeError("--resume and --restart are mutually exclusive")
    config_path = config_path.resolve()
    config = load_config(config_path)
    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    corpus_manifest: dict[str, Any] | None = None
    model_evidence: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    versions: dict[str, str] = {}

    def check(name: str, operation: Callable[[], Any]) -> Any:
        try:
            result = operation()
            _add_check(checks, name, True, result if result is not None else "valid")
            return result
        except Exception as exc:
            _add_check(checks, name, False, f"{type(exc).__name__}: {exc}")
            return None

    check("config_resolution", lambda: _validate_config(config))
    release = _os_release()
    environment = {"platform": platform.platform(), "machine": platform.node(), "python": sys.executable, "prefix": sys.prefix, "distro": release, "versions": versions}
    _add_check(checks, "environment", _is_wsl() and release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "24.04", environment)
    if checks[0]["passed"]:
        check("eager_compile_mode", lambda: _configure_compile_mode(config))
        environment["compile_mode"] = config["backend"]["compile_mode"]
        environment["torch_compile_disable"] = os.environ.get("TORCH_COMPILE_DISABLE")
        environment["unsloth_compile_disable"] = os.environ.get("UNSLOTH_COMPILE_DISABLE")
        backend, data, resources = config["backend"], config["data"], config["resources"]
        _add_check(checks, "python_version", list(sys.version_info[:2]) == backend["python_version"], list(sys.version_info[:2]))
        ram = _system_ram_gib()
        _add_check(checks, "system_ram", ram is not None and ram >= resources["minimum_system_ram_gib"], {"detected_gib": ram, "minimum_gib": resources["minimum_system_ram_gib"]})
        safe, detail = _safe_output(ROOT / config["output"]["directory"], resume=resume or restart)
        _add_check(checks, "output_directory_safety", safe, detail)
        for name, path in (("model_cache_storage", Path(config["model"]["cache_directory"]).expanduser()), ("output_storage", ROOT / config["output"]["directory"])):
            available = check(name, lambda path=path: _free_storage_gib(path))
            if available is not None:
                checks[-1]["passed"] = available >= resources["minimum_storage_free_gib"]
        _add_check(checks, "resource_estimate", resources["estimated_peak_vram_gib"] <= resources["minimum_vram_gib"], {"estimated_peak_vram_gib": resources["estimated_peak_vram_gib"], "measured": False, "limitation": "No model allocation/backward pass occurs in dry-run"})

        def dataset_check() -> dict[str, Any]:
            nonlocal corpus_manifest
            path = _under(ROOT / data["corpus_path"], ROOT / "training/datasets")
            manifest_path = _under(ROOT / data["corpus_manifest"], ROOT / "training/manifests")
            corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if corpus_manifest.get("artifact_type") != "external_corpus_manifest":
                raise RuntimeError("Configured corpus manifest has the wrong artifact type")
            if corpus_manifest.get("corpus_sha256") != _sha256_file(path):
                raise RuntimeError("Corpus bytes differ from the frozen corpus manifest")
            report = validate_files([path])
            if not report.valid:
                raise RuntimeError(f"Dataset schema errors: {report.errors}")
            rows.extend(load_jsonl(path))
            splits = {row["split"] for row in rows}
            if not rows or splits != {"train", "validation"}:
                raise RuntimeError("Nonempty train and validation splits are both required")
            split_counts = {split: sum(row["split"] == split for row in rows) for split in sorted(splits)}
            if corpus_manifest.get("records") != len(rows) or corpus_manifest.get("split_counts") != split_counts:
                raise RuntimeError("Corpus counts differ from the frozen corpus manifest")
            training_example_counts: dict[str, int] = {
                split: len(expand_training_examples(rows, split)) for split in sorted(splits)
            }
            training_example_counts["total"] = sum(training_example_counts.values())
            declared_examples = corpus_manifest.get("training_example_counts")
            if not isinstance(declared_examples, dict) or declared_examples != training_example_counts:
                raise RuntimeError("Expanded training-example counts differ from the frozen corpus manifest")
            return {
                "records": len(rows), "splits": sorted(splits),
                "training_example_counts": training_example_counts,
                "manifest": str(manifest_path), "corpus_sha256": corpus_manifest["corpus_sha256"],
            }

        check("dataset_schema_and_splits", dataset_check)

        def leakage_check() -> dict[str, Any]:
            nonlocal manifest
            path = _under(ROOT / data["eval_manifest"], ROOT / "training/manifests")
            manifest = load_eval_manifest(path)
            if data.get("eval_manifest_records_sha256") != manifest.get("records_sha256"):
                raise RuntimeError("Held-out manifest differs from the config's frozen digest")
            leaks = find_manifest_leaks(rows, manifest)
            if leaks:
                raise RuntimeError(f"Held-out collisions: {len(leaks)}")
            return {"collisions": 0, "records_sha256": manifest["records_sha256"]}

        check("held_out_leakage", leakage_check)
        # Resolve data before loading model components or permitting any downloads.
        data_valid = all(item["passed"] for item in checks if item["name"] in {"dataset_schema_and_splits", "held_out_leakage", "output_directory_safety", "model_cache_storage"})
        if data_valid:
            def import_required(names: Sequence[str]) -> None:
                for name in names:
                    module = check(f"import_{name}", lambda name=name: importer(name))
                    if module is not None:
                        modules[name] = module
                        versions[name] = str(getattr(module, "__version__", "unknown"))
                        checks[-1]["detail"] = versions[name]

            import_required(GPU_PREFLIGHT_IMPORTS)
            if all(name in modules for name in GPU_PREFLIGHT_IMPORTS):
                gpu = check("rocm_gpu_and_triton", lambda: _gpu_preflight(modules["torch"], modules["triton"], config)) or {}
            else:
                _add_check(checks, "rocm_gpu_and_triton", False, "Required GPU dependencies unavailable")

            import_required(REQUIRED_IMPORTS[len(GPU_PREFLIGHT_IMPORTS):])
            for name, expected in backend["package_versions"].items():
                _add_check(checks, f"version_{name}", versions.get(name) == expected, {"expected": expected, "actual": versions.get(name)})
            if all(name in modules for name in ("huggingface_hub", "transformers", "peft")):
                model_evidence = check("model_tokenizer_and_adapter", lambda: _model_preflight(config, rows, modules, allow_downloads)) or {}
            else:
                _add_check(checks, "model_tokenizer_and_adapter", False, "Required model dependencies unavailable")

    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train", "--dry-run-report", str(report_path.resolve()), "--confirm", CONFIRMATION]
    if resume:
        command.append("--resume")
    if restart:
        command.append("--restart")
    report = {
        "schema_version": 2, "artifact_type": "adapter_training_dry_run",
        "created_at": datetime.now(UTC).isoformat(), "passed": all(item["passed"] for item in checks),
        "config": str(config_path), "config_sha256": sha256_json(config), "training_spec_sha256": training_spec_digest(config),
        "dataset_sha256": sha256_json(rows), "corpus_manifest_sha256": sha256_json(corpus_manifest) if corpus_manifest else None,
        "eval_manifest_sha256": sha256_json(manifest) if manifest else None,
        "allow_downloads": allow_downloads, "checks": checks, "environment": environment,
        "gpu": gpu, "model_evidence": model_evidence, "resolved_config": config,
        "resolved_training_command": command, "training_performed": False,
    }
    if resume and report["passed"]:
        checkpoint = check(
            "resume_checkpoint",
            lambda: str(_select_resume_checkpoint(ROOT / config["output"]["directory"], _run_identity(config, report))),
        )
        report["passed"] = checkpoint is not None and all(item["passed"] for item in checks)
    if restart and report["passed"]:
        check("restart_before_first_checkpoint", lambda: _validate_restart_output(ROOT / config["output"]["directory"], _run_identity(config, report)))
        report["passed"] = all(item["passed"] for item in checks)
    return report


def _verify_training_gate(config: dict[str, Any], report: dict[str, Any], config_path: Path, *, resume: bool = False, restart: bool = False) -> None:
    if config.get("status") != "approved_after_dry_run":
        raise RuntimeError("Real training is disabled while status is not approved_after_dry_run")
    if report.get("schema_version") != 2 or report.get("artifact_type") != "adapter_training_dry_run" or report.get("passed") is not True:
        raise RuntimeError("A passing local dry-run report is required")
    if not report.get("checks") or not all(item.get("passed") is True for item in report["checks"]):
        raise RuntimeError("Dry-run evidence contains missing or failed checks")
    if report.get("training_spec_sha256") != training_spec_digest(config):
        raise RuntimeError("Dry-run report does not match current training settings")
    if report.get("config") != str(config_path.resolve()):
        raise RuntimeError("Dry-run report belongs to a different config location")
    environment = report.get("environment", {})
    if environment.get("python") != sys.executable or environment.get("machine") != platform.node():
        raise RuntimeError("Dry-run report belongs to a different local environment")
    data = config["data"]
    rows = load_jsonl(_under(ROOT / data["corpus_path"], ROOT / "training/datasets"))
    corpus_manifest = json.loads(_under(ROOT / data["corpus_manifest"], ROOT / "training/manifests").read_text(encoding="utf-8"))
    manifest = load_eval_manifest(_under(ROOT / data["eval_manifest"], ROOT / "training/manifests"))
    if (
        report.get("dataset_sha256") != sha256_json(rows)
        or report.get("corpus_manifest_sha256") != sha256_json(corpus_manifest)
        or report.get("eval_manifest_sha256") != sha256_json(manifest)
    ):
        raise RuntimeError("Dry-run report does not match the current dataset and held-out manifest")
    safe, detail = _safe_output(ROOT / config["output"]["directory"], resume=resume or restart)
    if not safe:
        raise RuntimeError(detail)


def execute_training(
    config: dict[str, Any], snapshot_path: str, *, run_identity: dict[str, Any],
    resume_checkpoint: Path | None = None, restart: bool = False,
) -> None:
    # Called only after main verifies saved evidence and repeats the local preflight.
    if resume_checkpoint is not None and restart:
        raise RuntimeError("Resume and restart are mutually exclusive")
    adapter, training, data, resources = (config[key] for key in ("adapter", "training", "data", "resources"))
    output = (ROOT / config["output"]["directory"]).resolve()
    safe, detail = _safe_output(output, resume=resume_checkpoint is not None or restart)
    if not safe:
        raise RuntimeError(detail)
    if resume_checkpoint is not None:
        selected = _select_resume_checkpoint(output, run_identity)
        if selected != _under(resume_checkpoint, output / "checkpoints"):
            raise RuntimeError("Selected resume checkpoint changed before model load")
        resume_checkpoint = selected
    elif restart:
        _validate_restart_output(output, run_identity)
    else:
        output.mkdir(parents=True, exist_ok=True)

    _configure_compile_mode(config)

    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    class CheckpointIntegrityCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step == training["first_checkpoint_step"]:
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = state.global_step
            checkpoint = output / "checkpoints" / f"checkpoint-{step}"
            _mark_checkpoint_complete(checkpoint, step, output, run_identity)
            return control

    class AdapterOnlySFTTrainer(SFTTrainer):
        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            if state_dict is not None:
                raise RuntimeError("Unexpected full state dict during adapter checkpoint")
            super()._save(output_dir=output_dir, state_dict=_adapter_only_state_dict(self.model))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=snapshot_path, use_exact_model_name=True, fast_inference=False,
        max_seq_length=data["max_sequence_length"], dtype=torch.bfloat16, load_in_4bit=True,
        full_finetuning=False, offload_embedding=resources["offload_embeddings"],
        local_files_only=True, trust_remote_code=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=adapter["rank"], lora_alpha=adapter["alpha"], lora_dropout=adapter["dropout"],
        bias=adapter["bias"], target_modules=adapter["target_modules"],
        use_gradient_checkpointing=resources["gradient_checkpointing"], random_state=training["seed"],
    )
    rows = load_jsonl(ROOT / data["corpus_path"])

    def prepared(split: str) -> Any:
        # Prompt/completion masking targets every assistant turn while preserving
        # native tool schemas, calls and results for the chat-template renderer.
        # Render before Arrow sees the rows: JSON-schema property names vary by
        # tool, and coercing those schemas into one Arrow struct can alter them.
        examples = expand_training_examples(rows, split)
        return Dataset.from_list(tokenize_training_examples(tokenizer, examples, data["max_sequence_length"]))

    trainer = AdapterOnlySFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=prepared("train"), eval_dataset=prepared("validation"),
        callbacks=[CheckpointIntegrityCallback()],
        args=SFTConfig(
            output_dir=str(output / "checkpoints"), max_length=data["max_sequence_length"], packing=False,
            completion_only_loss=True, per_device_train_batch_size=training["micro_batch_size"],
            per_device_eval_batch_size=1, gradient_accumulation_steps=training["gradient_accumulation_steps"],
            learning_rate=training["learning_rate"], num_train_epochs=training["epochs"],
            max_steps=training["max_steps"] if training["max_steps"] is not None else -1,
            warmup_ratio=training["warmup_ratio"], weight_decay=training["weight_decay"],
            optim=training["optimizer"], bf16=True, fp16=False, eval_strategy="steps", save_strategy="steps",
            eval_steps=training["validation_steps"], save_steps=training["checkpoint_steps"],
            save_total_limit=training["save_total_limit"], save_only_model=False,
            save_safetensors=True, logging_steps=training["logging_steps"],
            dataloader_num_workers=resources["dataloader_workers"], dataloader_pin_memory=resources["pin_memory"],
            seed=training["seed"], data_seed=training["seed"], report_to="none", push_to_hub=False,
        ),
    )
    if resume_checkpoint is None:
        _atomic_json(output / RUN_IDENTITY_FILE, run_identity)
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    model.save_pretrained(str(output / "adapter"), state_dict=_adapter_only_state_dict(model), safe_serialization=True)
    tokenizer.save_pretrained(str(output / "adapter"))
    write_json(output / "training_config.json", config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LocalPilot adapter training without training; launch only with --train.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow the pinned tokenizer AND model weights to be downloaded to configured cache")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run-report", type=Path)
    parser.add_argument("--confirm")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true", help="Resume the latest complete checkpoint from the exact same approved run; never restart silently")
    recovery.add_argument("--restart", action="store_true", help="Explicitly restart the same approved run only if it stopped before writing any checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    report_path = _under(args.report, ROOT / "training/reports")
    if args.dry_run:
        report = dry_run(args.config, allow_downloads=args.allow_downloads, resume=args.resume, restart=args.restart, report_path=report_path)
        write_json(report_path, report)
        print(json.dumps({"passed": report["passed"], "checks": [{"name": item["name"], "passed": item["passed"]} for item in report["checks"]], "report": str(report_path), "training_performed": False}, indent=2))
        return 0 if report["passed"] else 1
    if args.confirm != CONFIRMATION or args.dry_run_report is None:
        raise RuntimeError(f"Real training requires --dry-run-report PATH and --confirm {CONFIRMATION}")
    saved_path = _under(args.dry_run_report, ROOT / "training/reports")
    report = json.loads(saved_path.read_text(encoding="utf-8"))
    _verify_training_gate(config, report, args.config, resume=args.resume, restart=args.restart)
    fresh = dry_run(args.config, allow_downloads=False, resume=args.resume, restart=args.restart, report_path=saved_path)
    if not fresh["passed"]:
        failed = [{"name": item["name"], "detail": item["detail"]} for item in fresh["checks"] if not item["passed"]]
        raise RuntimeError(f"Current local preflight failed: {json.dumps(failed, ensure_ascii=True)}")
    if fresh["environment"] != report["environment"] or fresh["model_evidence"] != report["model_evidence"]:
        raise RuntimeError("Installed environment or cached model changed after the approved dry-run")
    identity = _run_identity(config, fresh)
    checkpoint = _select_resume_checkpoint(ROOT / config["output"]["directory"], identity) if args.resume else None
    if checkpoint is not None:
        print(f"Resuming LocalPilot training from {checkpoint}", flush=True)
    execute_training(config, fresh["model_evidence"]["snapshot_path"], run_identity=identity, resume_checkpoint=checkpoint, restart=args.restart)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Adapter preflight refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
