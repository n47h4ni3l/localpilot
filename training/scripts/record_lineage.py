#!/usr/bin/env python3
"""Record an owner-accepted Nestra training lineage head from verified local artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_adapter import (
    CHECKPOINT_MARKER_FILE,
    ROOT,
    TRAINING_COMPLETE_FILE,
    TRAINING_OUTPUT_ROOT,
    _artifact_inventory,
    _sha256_file,
    _under,
    load_config,
)


def _sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{name} must be a 64-character SHA256")
    return value


def build_manifest(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    config = load_config(config_path)
    output = _under(ROOT / config["output"]["directory"], TRAINING_OUTPUT_ROOT)
    marker_path = _under(output / TRAINING_COMPLETE_FILE, output)
    if not marker_path.is_file():
        raise RuntimeError("Training completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    adapter_path = _under(output / "adapter", output)
    adapter_files = _artifact_inventory(
        adapter_path, ("adapter_model.safetensors", "adapter_config.json")
    )
    if marker.get("adapter_files") != adapter_files:
        raise RuntimeError("Final adapter differs from the natural-completion marker")
    step = marker.get("global_step")
    if type(step) is not int or step < 1:
        raise RuntimeError("Training completion marker has an invalid global step")
    checkpoint = _under(
        output / "checkpoints" / f"checkpoint-{step}",
        output / "checkpoints",
    )
    checkpoint_marker_path = _under(
        checkpoint / CHECKPOINT_MARKER_FILE, checkpoint
    )
    try:
        checkpoint_marker = json.loads(
            checkpoint_marker_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("Final checkpoint completion marker is missing or invalid") from exc
    if (
        checkpoint_marker.get("global_step") != step
        or checkpoint_marker.get("run_identity_sha256")
        != marker.get("run_identity_sha256")
    ):
        raise RuntimeError("Final checkpoint evidence does not match natural completion")

    package = int(args.package)
    lineage = config.get("lineage")
    if package == 1:
        if lineage is not None:
            raise RuntimeError("Package 1 must not declare a parent lineage")
    else:
        if not isinstance(lineage, dict) or lineage.get("package") != package:
            raise RuntimeError("Package number must match config.lineage.package")
        parent_path = _under(ROOT / lineage["parent_manifest"], ROOT / "training/lineage")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if (
            parent.get("artifact_type") != "nestra_lineage_head"
            or parent.get("accepted_as_lineage_head") is not True
            or parent.get("package") != package - 1
        ):
            raise RuntimeError("Configured parent is not the immediately previous accepted lineage")

    model = config["model"]
    return {
        "schema_version": 1,
        "artifact_type": "nestra_lineage_head",
        "recorded_at": datetime.now(UTC).isoformat(),
        "package": package,
        "accepted_as_lineage_head": True,
        "benchmark_promoted": bool(args.benchmark_promoted),
        "base": {
            "identity": model["base_identity"],
            "revision": model["base_revision"],
            "training_model_id": model["training_model_id"],
            "training_revision": model["revision"],
        },
        "training": {
            "config": str(config_path.relative_to(ROOT).as_posix()),
            "output_directory": config["output"]["directory"],
            "global_step": marker["global_step"],
            "run_identity_sha256": marker["run_identity_sha256"],
            "training_config_sha256": marker["training_config_sha256"],
            "completion_marker_sha256": _sha256_file(marker_path),
            "checkpoint_marker_sha256": _sha256_file(checkpoint_marker_path),
            "checkpoint_files": checkpoint_marker.get("files"),
            "adapter_files": adapter_files,
        },
        "deployment": {
            "model": args.candidate_model,
            "digest": _sha256(args.candidate_digest, "candidate digest"),
            "gguf_sha256": _sha256(args.candidate_gguf_sha256, "candidate GGUF SHA256"),
        },
        "rollback": {
            "model": args.rollback_model,
            "digest": _sha256(args.rollback_digest, "rollback digest"),
        },
        "original_baseline": {
            "model": args.baseline_model,
            "digest": _sha256(args.baseline_digest, "baseline digest"),
        },
        "evaluation": {
            "eval_v1": {
                "report": args.eval_report_ref,
                "overall": args.eval_overall,
                "critical": args.eval_critical,
                "hard_failures": args.eval_hard_failures,
            },
            "evolution_execution_v1": {
                "report": args.execution_report_ref,
                "overall": args.execution_overall,
                "hard_failures": args.execution_hard_failures,
                "scope_violations": args.execution_scope_violations,
            },
        },
        "owner_acceptance": {
            "decision": "accepted_as_working_lineage",
            "note": args.note,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package", type=int, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-digest", required=True)
    parser.add_argument("--candidate-gguf-sha256", required=True)
    parser.add_argument("--rollback-model", required=True)
    parser.add_argument("--rollback-digest", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-digest", required=True)
    parser.add_argument("--eval-report-ref", required=True)
    parser.add_argument("--eval-overall", type=float, required=True)
    parser.add_argument("--eval-critical", type=float, required=True)
    parser.add_argument("--eval-hard-failures", type=int, required=True)
    parser.add_argument("--execution-report-ref", required=True)
    parser.add_argument("--execution-overall", type=float, required=True)
    parser.add_argument("--execution-hard-failures", type=int, required=True)
    parser.add_argument("--execution-scope-violations", type=int, required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--benchmark-promoted", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.package < 1:
        raise RuntimeError("package must be >= 1")
    if not 0 <= args.eval_overall <= 4 or not 0 <= args.eval_critical <= 4:
        raise RuntimeError("Eval scores must be between 0 and 4")
    if not 0 <= args.execution_overall <= 4:
        raise RuntimeError("Execution score must be between 0 and 4")
    if min(args.eval_hard_failures, args.execution_hard_failures, args.execution_scope_violations) < 0:
        raise RuntimeError("Failure and scope counts cannot be negative")

    target = (
        args.output.resolve()
        if args.output is not None
        else (ROOT / f"training/lineage/package-{args.package}.json").resolve()
    )
    lineage_root = (ROOT / "training/lineage").resolve()
    try:
        target.relative_to(lineage_root)
    except ValueError as exc:
        raise RuntimeError("Lineage output must remain under training/lineage") from exc
    if target.exists():
        raise RuntimeError("Lineage record already exists; preserve it and choose a fresh output")

    manifest = build_manifest(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lineage record refused: {exc}")
        raise SystemExit(2) from exc
