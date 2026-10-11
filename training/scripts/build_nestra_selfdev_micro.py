#!/usr/bin/env python3
"""Fail-closed promotion of independently reviewed micro-corpus candidates.

Staged examples are Tier D/review_required and NEVER trainable as-is.
This script has no model calls, no network and cannot approve an example.
A separate reviewer must inspect every source claim and record a signed-off
decision; this code verifies structure/provenance, not the reviewer's judgment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
from training_common import find_manifest_leaks, load_eval_manifest  # noqa: E402
from validate_dataset import validate_files  # noqa: E402
from train_adapter import expand_training_examples  # noqa: E402

STAGED = ROOT / "training/sources/nestra_p1_selfdev_micro_candidates.jsonl"
DATASET = ROOT / "training/datasets/nestra_p1_selfdev_micro.jsonl"
MANIFEST = ROOT / "training/manifests/nestra_p1_selfdev_micro_manifest.json"
EVAL_MANIFEST = ROOT / "training/manifests/eval_v1_manifest.json"
REQUIRED_COUNTS = {"train": 64, "validation": 12}
PROVENANCE_COMMIT = "417656c8bdb541ff4aba1ee91b5d6985842fc3b7"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_candidates(path: Path = STAGED) -> tuple[list[dict], str]:
    data = path.read_bytes()
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    ids = [r.get("id") for r in records]
    splits = Counter(r.get("split") for r in records)
    if dict(splits) != REQUIRED_COUNTS or len(ids) != len(set(ids)):
        raise ValueError("The frozen 64/12 split, IDs or record count changed")
    prompts: set[str] = set()
    for record in records:
        if record.get("quality_tier") != "D" or record.get("verification_status") != "review_required":
            raise ValueError("Staging records must remain unapproved Tier D")
        if record.get("source") != "project_owned_synthetic_candidate":
            raise ValueError("Unexpected source authority")
        if record.get("metadata", {}).get("candidate_only") is not True:
            raise ValueError("Staging records must be explicitly non-trainable")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2 or [
            item.get("role") for item in messages
        ] != ["user", "assistant"]:
            raise ValueError("Every micro-example must have one user and one assistant turn")
        if any(not isinstance(item.get("content"), str) or not item["content"].strip()
               for item in messages):
            raise ValueError("Empty prompt or target")
        prompt = " ".join(messages[0]["content"].casefold().split())
        if prompt in prompts:
            raise ValueError("Duplicated prompt")
        prompts.add(prompt)
        evidence = record.get("provenance") or {}
        if (evidence.get("reference") != f"github:commit/{PROVENANCE_COMMIT}"
                or not isinstance(evidence.get("files"), list)
                or len(evidence["files"]) != 1):
            raise ValueError("Missing pinned candidate source reference")
    return records, sha(data)


def verify_review(
    records: list[dict], candidate_sha: str, review: dict,
    *, verify_git: bool = True,
) -> list[dict]:
    if review.get("candidate_sha256") != candidate_sha:
        raise ValueError("Review is not bound to the frozen candidate bytes")
    approvals = review.get("approvals")
    if not isinstance(approvals, list) or len(approvals) != len(records):
        raise ValueError("Independent source review is incomplete")
    by_id = {}
    for entry in approvals:
        if not isinstance(entry, dict) or entry.get("id") in by_id:
            raise ValueError("Malformed or duplicate review")
        rid = entry.get("id")
        if (entry.get("approved") is not True
                or not isinstance(entry.get("reviewer"), str)
                or len(entry["reviewer"].strip()) < 3
                or not isinstance(entry.get("rationale"), str)
                or len(entry["rationale"].strip()) < 25
                or entry.get("source_commit") != PROVENANCE_COMMIT):
            raise ValueError(f"{rid}: missing explicit reviewer approval or reasoning")
        by_id[rid] = entry
    if set(by_id) != {r["id"] for r in records}:
        raise ValueError("Review IDs differ from frozen candidate IDs")
    reviewed = []
    verified_sources: set[str] = set()
    for record in records:
        approval = by_id[record["id"]]
        source = record["provenance"]["files"][0]
        if approval.get("source_path") != source:
            raise ValueError(f"{record['id']}: reviewed source does not match provenance")
        if verify_git and source not in verified_sources:
            check = subprocess.run(
                ["git", "cat-file", "-e", f"{PROVENANCE_COMMIT}:{source}"],
                cwd=ROOT, capture_output=True, check=False,
            )
            if check.returncode:
                raise ValueError(f"Missing immutable source blob for {source}")
            verified_sources.add(source)
        converted = json.loads(json.dumps(record))
        converted["quality_tier"] = "C"
        converted["verification_status"] = "synthetic_validated"
        converted["source"] = "project_owned_independently_reviewed_synthetic"
        converted["tags"] = [
            item for item in converted.get("tags", []) if item != "candidate_unapproved"
        ]
        converted["metadata"] = {
            "candidate_sha256": candidate_sha,
            "reviewed_by": approval["reviewer"].strip(),
            "review_source_commit": PROVENANCE_COMMIT,
            "synthetic_validated": True,
        }
        converted["provenance"]["review"] = {
            "source_commit": PROVENANCE_COMMIT,
            "source_path": source,
            "reviewer": approval["reviewer"].strip(),
            "rationale": approval["rationale"].strip(),
        }
        reviewed.append(converted)
    return reviewed


def make_dataset(
    records: list[dict], *,
    holdout_manifest: Path,
    eval_manifest: Path = EVAL_MANIFEST,
) -> tuple[bytes, dict]:
    with tempfile.TemporaryDirectory(prefix="nestra-micro-validation-") as temp:
        path = Path(temp) / "reviewed.jsonl"
        content = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                          for r in records).encode("utf-8")
        path.write_bytes(content)
        report = validate_files([path])
        if not report.valid:
            raise ValueError(f"Reviewed dataset failed schema validation: {report.errors}")
        for manifest_path in (eval_manifest, holdout_manifest):
            leaks = find_manifest_leaks(records, load_eval_manifest(manifest_path))
            if leaks:
                raise ValueError(f"Heldout leakage detected against {manifest_path.name}: {leaks[:3]}")
    split_counts = dict(sorted(Counter(r["split"] for r in records).items()))
    expanded = {
        split: len(expand_training_examples(records, split))
        for split in ("train", "validation")
    }
    expanded["total"] = expanded["train"] + expanded["validation"]
    manifest = {
        "artifact_type": "external_corpus_manifest",
        "schema_version": 1,
        "name": "Nestra self-development P1 micro-continuation",
        "records": len(records),
        "split_counts": split_counts,
        "training_example_counts": expanded,
        "corpus_sha256": sha(content),
        "frozen_eval_manifest_sha256": sha(eval_manifest.read_bytes()),
        "private_holdout_manifest_sha256": sha(holdout_manifest.read_bytes()),
    }
    return content, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--private-heldout-manifest", type=Path)
    args = parser.parse_args()
    if args.inspect == args.build:
        parser.error("Specify exactly one of --inspect or --build")
    try:
        records, candidate_digest = read_candidates()
        if args.inspect:
            print(json.dumps({
                "approved_for_training": False,
                "candidate_records": len(records),
                "splits": REQUIRED_COUNTS,
                "candidate_sha256": candidate_digest,
                "next": "Independent source review and private holdout are required",
            }, indent=2))
            return 0
        if not args.review_file or not args.private_heldout_manifest:
            raise ValueError("--build requires a complete review and frozen private heldout manifest")
        review_bytes = args.review_file.read_bytes()
        rows = verify_review(records, candidate_digest, json.loads(review_bytes))
        content, manifest = make_dataset(rows, holdout_manifest=args.private_heldout_manifest)
        manifest["review_file_sha256"] = sha(review_bytes)
        # Never silently replace an existing immutable training corpus/manifest.
        for path, expected in ((DATASET, content),
                               (MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())):
            if path.exists() and path.read_bytes() != expected:
                raise ValueError(f"Refuse to overwrite existing frozen artifact: {path}")
        DATASET.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        DATASET.write_bytes(content)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": str(DATASET), "manifest": str(MANIFEST),
                          "records": len(rows), "expanded": manifest["training_example_counts"]}, indent=2))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Micro-corpus is NOT approved for training: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
