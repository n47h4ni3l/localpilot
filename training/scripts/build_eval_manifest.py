#!/usr/bin/env python3
"""Freeze IDs and one-way content hashes for the held-out Eval v1 corpus.

This is the only training-preparation utility permitted to read ``training/evals``.
Its output contains no prompts, rubrics, expected behavior, or model answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import (
    MANIFEST_SCHEMA_VERSION, ROOT, load_eval_manifest, record_fingerprints,
    sha256_json, sha256_text, validate_eval_manifest, write_json,
)
from validate_dataset import Location, _validate_record_shape


DEFAULT_EVAL_ROOT = ROOT / "training" / "evals"
DEFAULT_OUTPUT = ROOT / "training" / "manifests" / "eval_v1_manifest.json"


def _load_held_out(path: Path) -> list[tuple[int, dict[str, Any], str]]:
    records: list[tuple[int, dict[str, Any], str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number}: record must be an object")
        if value.get("split") != "held_out_eval":
            raise RuntimeError(f"{path}:{line_number}: non-held-out record in Eval v1")
        if any(message.get("role") == "assistant" for message in value.get("messages", []) if isinstance(message, dict)):
            raise RuntimeError(f"{path}:{line_number}: held-out record contains an assistant answer")
        errors: list[str] = []
        _validate_record_shape(value, Location(path, line_number), errors, [])
        if errors:
            raise RuntimeError("Malformed held-out record: " + "; ".join(errors))
        records.append((line_number, value, raw))
    return records


def build_manifest(eval_root: Path = DEFAULT_EVAL_ROOT) -> dict[str, Any]:
    if not eval_root.is_dir():
        raise RuntimeError(f"Held-out eval root does not exist: {eval_root}")
    output_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(eval_root.rglob("*.jsonl")):
        for component in [path, *path.parents]:
            if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
                raise RuntimeError("Held-out manifest input must not use symlinks or junctions.")
        if not path.resolve().is_relative_to(eval_root.resolve()):
            raise RuntimeError("Held-out manifest input escapes its root.")
        for line_number, record, raw in _load_held_out(path):
            record_id = str(record.get("id") or "")
            if not record_id or record_id in seen_ids:
                raise RuntimeError(f"{path}:{line_number}: missing or duplicate held-out ID")
            seen_ids.add(record_id)
            fingerprints = record_fingerprints(record)
            output_records.append(
                {
                    "id": record_id,
                    "id_sha256": sha256_text(record_id),
                    "source_file": path.relative_to(eval_root).as_posix(),
                    "raw_record_sha256": sha256_text(raw.strip()),
                    **fingerprints,
                }
            )
    if not output_records:
        raise RuntimeError("No held-out Eval v1 records were found.")
    output_records.sort(key=lambda item: item["id"])
    frozen_payload = [{key: value for key, value in item.items() if key != "source_file"} for item in output_records]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "held_out_eval_manifest",
        "eval_name": "LocalPilot Eval v1",
        "created_at": datetime.now(UTC).isoformat(),
        "record_count": len(output_records),
        "content_shingle_tokens": 16,
        "records_sha256": sha256_json(frozen_payload),
        "contains_plaintext_eval_content": False,
        "contains_prompts": False,
        "contains_rubrics": False,
        "contains_answers": False,
        "records": output_records,
    }
    validate_eval_manifest(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze held-out Eval v1 IDs and leakage hashes.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="verify the tracked manifest is current")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.absolute()
    allowed_root = ROOT / "training" / "manifests"
    if not output.resolve().is_relative_to(allowed_root.resolve()) or output.suffix != ".json":
        raise RuntimeError("Manifest output must be a .json file under training/manifests.")
    for component in [output, *output.parents]:
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise RuntimeError("Manifest output must not use symlinks or junctions.")
    manifest = build_manifest(args.eval_root.absolute())
    if args.check:
        existing = load_eval_manifest(output)
        manifest = json.loads(json.dumps(manifest))
        for value in (manifest, existing):
            value.pop("created_at", None)
        if manifest != existing:
            raise RuntimeError("Tracked Eval v1 manifest is stale; regenerate it.")
        print(f"Eval v1 manifest is current ({manifest['record_count']} held-out IDs/hashes).")
        return 0
    write_json(output, manifest)
    print(f"Frozen {manifest['record_count']} held-out Eval v1 IDs/hashes; no plaintext eval content written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
