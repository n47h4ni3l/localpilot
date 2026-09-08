#!/usr/bin/env python3
"""Reproducible, non-training preflight and explicitly gated Unsloth adapter run.

No model weights are allocated on the GPU during dry-run. It validates the local
snapshot, tokenizer, PEFT config and small ROCm operations; memory remains an
estimate until a separately authorized training run measures its actual peak.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
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
# Unsloth must patch Transformers/PEFT before they are imported.
REQUIRED_IMPORTS = (
    "unsloth", "torch", "unsloth_zoo", "transformers", "peft", "trl",
    "datasets", "bitsandbytes", "triton", "huggingface_hub",
)
CONFIRMATION = "TRAIN_LOCALPILOT_ADAPTER_V1"
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


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


def _safe_output(path: Path) -> tuple[bool, str]:
    try:
        resolved = _under(path, TRAINING_OUTPUT_ROOT)
        if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
            raise RuntimeError("Output must be an absent or empty directory")
        return True, str(resolved)
    except (OSError, RuntimeError) as exc:
        return False, str(exc)


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
    for key in ("micro_batch_size", "gradient_accumulation_steps", "validation_steps", "checkpoint_steps", "logging_steps", "save_total_limit"):
        number(training, key, 1, 10000, integer=True)
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
    for key in ("minimum_vram_gib", "estimated_peak_vram_gib", "minimum_system_ram_gib", "minimum_storage_free_gib"):
        number(resources, key, 1, 10000)
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
    lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)) for row in rows]
    if not lengths or min(lengths) < 1 or max(lengths) > config["data"]["max_sequence_length"]:
        raise RuntimeError(f"Empty or overlength tokenized examples; maximum={max(lengths, default=0)}")
    adapter = config["adapter"]
    modules["peft"].LoraConfig(
        task_type="CAUSAL_LM", r=adapter["rank"], lora_alpha=adapter["alpha"],
        lora_dropout=adapter["dropout"], bias=adapter["bias"], target_modules=adapter["target_modules"],
    )
    return {
        "model_id": model["training_model_id"], "revision": model["revision"],
        "snapshot_path": str(snapshot), "weight_inventory": inventory,
        "token_counts": {"total": sum(lengths), "maximum": max(lengths), "records": len(lengths)},
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


def dry_run(config_path: Path, *, allow_downloads: bool = False, report_path: Path = DEFAULT_REPORT, importer: Callable[[str], Any] = importlib.import_module) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
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
        backend, data, resources = config["backend"], config["data"], config["resources"]
        _add_check(checks, "python_version", list(sys.version_info[:2]) == backend["python_version"], list(sys.version_info[:2]))
        ram = _system_ram_gib()
        _add_check(checks, "system_ram", ram is not None and ram >= resources["minimum_system_ram_gib"], {"detected_gib": ram, "minimum_gib": resources["minimum_system_ram_gib"]})
        safe, detail = _safe_output(ROOT / config["output"]["directory"])
        _add_check(checks, "output_directory_safety", safe, detail)
        for name, path in (("model_cache_storage", Path(config["model"]["cache_directory"]).expanduser()), ("output_storage", ROOT / config["output"]["directory"])):
            available = check(name, lambda path=path: _free_storage_gib(path))
            if available is not None:
                checks[-1]["passed"] = available >= resources["minimum_storage_free_gib"]
        _add_check(checks, "resource_estimate", resources["estimated_peak_vram_gib"] <= resources["minimum_vram_gib"], {"estimated_peak_vram_gib": resources["estimated_peak_vram_gib"], "measured": False, "limitation": "No model allocation/backward pass occurs in dry-run"})

        def dataset_check() -> dict[str, Any]:
            path = _under(ROOT / data["corpus_path"], ROOT / "training/datasets")
            report = validate_files([path])
            if not report.valid:
                raise RuntimeError(f"Dataset schema errors: {report.errors}")
            rows.extend(load_jsonl(path))
            splits = {row["split"] for row in rows}
            if not rows or splits != {"train", "validation"}:
                raise RuntimeError("Nonempty train and validation splits are both required")
            return {"records": len(rows), "splits": sorted(splits)}

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
            for name in REQUIRED_IMPORTS:
                module = check(f"import_{name}", lambda name=name: importer(name))
                if module is not None:
                    modules[name] = module
                    versions[name] = str(getattr(module, "__version__", "unknown"))
                    checks[-1]["detail"] = versions[name]
            for name, expected in backend["package_versions"].items():
                _add_check(checks, f"version_{name}", versions.get(name) == expected, {"expected": expected, "actual": versions.get(name)})
            if "torch" in modules and "triton" in modules:
                gpu = check("rocm_gpu_and_triton", lambda: _gpu_preflight(modules["torch"], modules["triton"], config)) or {}
            else:
                _add_check(checks, "rocm_gpu_and_triton", False, "Required GPU dependencies unavailable")
            if all(name in modules for name in ("huggingface_hub", "transformers", "peft")):
                model_evidence = check("model_tokenizer_and_adapter", lambda: _model_preflight(config, rows, modules, allow_downloads)) or {}
            else:
                _add_check(checks, "model_tokenizer_and_adapter", False, "Required model dependencies unavailable")

    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train", "--dry-run-report", str(report_path.resolve()), "--confirm", CONFIRMATION]
    return {
        "schema_version": 2, "artifact_type": "adapter_training_dry_run",
        "created_at": datetime.now(UTC).isoformat(), "passed": all(item["passed"] for item in checks),
        "config": str(config_path), "config_sha256": sha256_json(config), "training_spec_sha256": training_spec_digest(config),
        "dataset_sha256": sha256_json(rows), "eval_manifest_sha256": sha256_json(manifest) if manifest else None,
        "allow_downloads": allow_downloads, "checks": checks, "environment": environment,
        "gpu": gpu, "model_evidence": model_evidence, "resolved_config": config,
        "resolved_training_command": command, "training_performed": False,
    }


def _verify_training_gate(config: dict[str, Any], report: dict[str, Any], config_path: Path) -> None:
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
    manifest = load_eval_manifest(_under(ROOT / data["eval_manifest"], ROOT / "training/manifests"))
    if report.get("dataset_sha256") != sha256_json(rows) or report.get("eval_manifest_sha256") != sha256_json(manifest):
        raise RuntimeError("Dry-run report does not match the current dataset and held-out manifest")
    safe, detail = _safe_output(ROOT / config["output"]["directory"])
    if not safe:
        raise RuntimeError(detail)


def execute_training(config: dict[str, Any], snapshot_path: str) -> None:
    # Called only after main verifies saved evidence and repeats the local preflight.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    adapter, training, data, resources = (config[key] for key in ("adapter", "training", "data", "resources"))
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
        # Explicit prompt/completion structure masks user text from target loss.
        return Dataset.from_list([
            {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}
            for row in rows if row["split"] == split
        ])

    output = (ROOT / config["output"]["directory"]).resolve()
    safe, detail = _safe_output(output)
    if not safe:
        raise RuntimeError(detail)
    output.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=prepared("train"), eval_dataset=prepared("validation"),
        args=SFTConfig(
            output_dir=str(output / "checkpoints"), max_length=data["max_sequence_length"], packing=False,
            completion_only_loss=True, per_device_train_batch_size=training["micro_batch_size"],
            per_device_eval_batch_size=1, gradient_accumulation_steps=training["gradient_accumulation_steps"],
            learning_rate=training["learning_rate"], num_train_epochs=training["epochs"],
            max_steps=training["max_steps"] if training["max_steps"] is not None else -1,
            warmup_ratio=training["warmup_ratio"], weight_decay=training["weight_decay"],
            optim=training["optimizer"], bf16=True, fp16=False, eval_strategy="steps",
            eval_steps=training["validation_steps"], save_steps=training["checkpoint_steps"],
            save_total_limit=training["save_total_limit"], logging_steps=training["logging_steps"],
            dataloader_num_workers=resources["dataloader_workers"], dataloader_pin_memory=resources["pin_memory"],
            seed=training["seed"], data_seed=training["seed"], report_to="none", push_to_hub=False,
        ),
    )
    trainer.train()
    model.save_pretrained(str(output / "adapter"), safe_serialization=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    report_path = _under(args.report, ROOT / "training/reports")
    if args.dry_run:
        report = dry_run(args.config, allow_downloads=args.allow_downloads, report_path=report_path)
        write_json(report_path, report)
        print(json.dumps({"passed": report["passed"], "checks": [{"name": item["name"], "passed": item["passed"]} for item in report["checks"]], "report": str(report_path), "training_performed": False}, indent=2))
        return 0 if report["passed"] else 1
    if args.confirm != CONFIRMATION or args.dry_run_report is None:
        raise RuntimeError(f"Real training requires --dry-run-report PATH and --confirm {CONFIRMATION}")
    saved_path = _under(args.dry_run_report, ROOT / "training/reports")
    report = json.loads(saved_path.read_text(encoding="utf-8"))
    _verify_training_gate(config, report, args.config)
    fresh = dry_run(args.config, allow_downloads=False, report_path=saved_path)
    if not fresh["passed"]:
        raise RuntimeError("Current local preflight failed; rerun --dry-run and inspect its report")
    if fresh["environment"] != report["environment"] or fresh["model_evidence"] != report["model_evidence"]:
        raise RuntimeError("Installed environment or cached model changed after the approved dry-run")
    execute_training(config, fresh["model_evidence"]["snapshot_path"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Adapter preflight refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
