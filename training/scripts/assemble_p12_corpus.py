#!/usr/bin/env python3
"""Assemble immutable P1.2 corpus shards, validate schema, splits, hashes and held-out isolation.

This never creates training examples or downloads data. The SHA-256 in the
manifest binds the exact output bytes. On mismatch it refuses to overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))

from training_common import find_manifest_leaks, load_eval_manifest
from train_adapter import expand_training_examples
from validate_dataset import validate_files

MANIFEST = ROOT / "training/manifests/nestra_p12_curated_1000_manifest.json"
DESTINATION = ROOT / "training/datasets/nestra_p12_curated_1000.jsonl"
EVAL_MANIFEST = ROOT / "training/manifests/eval_v1_manifest.json"


def build(*, check_only: bool = False) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "external_corpus_manifest":
        raise RuntimeError("Unexpected corpus manifest type")
    names = manifest.get("shards")
    if not isinstance(names, list) or len(names) != 12:
        raise RuntimeError("Expected exactly 12 named corpus shards")
    payload = bytearray()
    for index, name in enumerate(names, 1):
        expected_name = f"training/datasets/p12_shards/part-{index:02d}.jsonl"
        if name != expected_name:
            raise RuntimeError(f"Unexpected shard path: {name}")
        shard = (ROOT / name).resolve()
        if not shard.is_relative_to(ROOT / "training/datasets/p12_shards"):
            raise RuntimeError("Corpus shard escaped the allowed directory")
        content = shard.read_bytes()
        if not content or not content.endswith(b"\n"):
            raise RuntimeError(f"Shard is empty or lacks newline: {name}")
        payload.extend(content)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["corpus_sha256"]:
        raise RuntimeError(f"P1.2 corpus digest differs: {digest}")
    records = [json.loads(line) for line in payload.splitlines()]
    if len(records) != manifest["records"] or len(records) != 1000:
        raise RuntimeError("P1.2 record-count mismatch")

    # Check the final combined dataset using existing project validators.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".jsonl", prefix="p12-", dir=DESTINATION.parent,
            delete=False
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        checked = validate_files([temp_path])
        if checked.errors:
            raise RuntimeError(f"P1.2 schema/duplicate errors: {checked.errors[:12]}")
        groups, prompts = {}, set()
        for record in records:
            meta = record.get("metadata") or {}
            family = meta.get("problem_identity_sha256") or meta.get("family_id") or record["id"]
            if family in groups:
                raise RuntimeError(f"Duplicate source-family identifier: {family}")
            groups[family] = record["split"]
            prompt = tuple(" ".join(msg["content"].split()).casefold() for msg in record["messages"] if msg["role"] == "user")
            if prompt in prompts:
                raise RuntimeError(f"Duplicate user prompt: {record['id']}")
            prompts.add(prompt)
            if record["verification_status"] not in {"source_verified", "ci_verified", "human_verified"}:
                raise RuntimeError(f"Unverified record: {record['id']}")
        counts = dict(Counter(record["split"] for record in records))
        expanded = {
            split: len(expand_training_examples(records, split))
            for split in ("train", "validation")
        }
        expanded["total"] = expanded["train"] + expanded["validation"]
        if counts != manifest["split_counts"] or expanded != manifest["training_example_counts"]:
            raise RuntimeError(f"Expanded example counts mismatch: {counts}, {expanded}")
        model_eval = load_eval_manifest(EVAL_MANIFEST)
        leaks = find_manifest_leaks(records, model_eval)
        if leaks:
            raise RuntimeError(f"Held-out Eval v1 overlap: {leaks[:6]}")
        stats = {
            "records": len(records),
            "split_counts": counts,
            "training_example_counts": expanded,
            "sha256": digest,
            "bytes": len(payload),
            "eval_v1_leaks": 0,
            "warnings": checked.warnings[:8],
        }
        if DESTINATION.exists():
            if DESTINATION.read_bytes() != payload:
                raise RuntimeError("Existing generated corpus differs; refusing overwrite")
        elif not check_only:
            os.replace(temp_path, DESTINATION)
            temp_path = None
        return stats
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without publishing assembled corpus")
    args = parser.parse_args()
    try:
        report = build(check_only=args.check)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
