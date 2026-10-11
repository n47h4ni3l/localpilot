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
EXTERNAL_STAGED = ROOT / "training/sources/nestra_p1_selfdev_external_candidates.jsonl"
DATASET = ROOT / "training/datasets/nestra_p1_selfdev_micro.jsonl"
MANIFEST = ROOT / "training/manifests/nestra_p1_selfdev_micro_manifest.json"
EVAL_MANIFEST = ROOT / "training/manifests/eval_v1_manifest.json"
REQUIRED_COUNTS = {"train": 68, "validation": 12}
LOCAL_COUNTS = {"train": 64, "validation": 12}
EXTERNAL_COUNT = 4
UPSTREAM_COMMIT = "0b26ccaa415992100fa32df62d41b994cf928e23"
UPSTREAM_PATH = "data/seed_tasks.jsonl"
EXTERNAL_SOURCE_IDS = frozenset({"seed_task_92", "seed_task_70", "seed_task_37", "seed_task_19"})
PROVENANCE_COMMIT = "417656c8bdb541ff4aba1ee91b5d6985842fc3b7"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_candidates(
    path: Path = STAGED, *, external_path: Path = EXTERNAL_STAGED,
) -> tuple[list[dict], str]:
    """Bind every candidate to both frozen inputs; four external rows are required."""
    local_bytes = path.read_bytes()
    external_bytes = external_path.read_bytes()
    local = [json.loads(line) for line in local_bytes.splitlines() if line.strip()]
    external = [json.loads(line) for line in external_bytes.splitlines() if line.strip()]
    if (dict(Counter(r.get("split") for r in local)) != LOCAL_COUNTS
            or len(external) != EXTERNAL_COUNT
            or any(r.get("split") != "train" for r in external)):
        raise ValueError("Required 64 local train, 12 validation, 4 external train records")
    records = local + external
    ids = [r.get("id") for r in records]
    splits = Counter(r.get("split") for r in records)
    if dict(splits) != REQUIRED_COUNTS or len(ids) != len(set(ids)):
        raise ValueError("Frozen 68/12 split, IDs or record count changed")
    if {r.get("provenance", {}).get("source_id") for r in external} != EXTERNAL_SOURCE_IDS:
        raise ValueError("Missing or substituted required Self-Instruct seed ID")
    prompts: set[str] = set()
    for record in records:
        if record.get("quality_tier") != "D" or record.get("verification_status") != "review_required":
            raise ValueError("Staging records must remain unapproved Tier D")
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
        if record in local:
            if (record.get("source") != "project_owned_synthetic_candidate"
                    or evidence.get("reference") != f"github:commit/{PROVENANCE_COMMIT}"
                    or not isinstance(evidence.get("files"), list)
                    or len(evidence["files"]) != 1):
                raise ValueError("Missing pinned LocalPilot candidate source")
        else:
            if (record.get("source") != "external_self_instruct_human_seed_candidate"
                    or record.get("license") != "Apache-2.0"
                    or evidence.get("repository") != "yizhongw/self-instruct"
                    or evidence.get("upstream_commit") != UPSTREAM_COMMIT
                    or evidence.get("upstream_path") != UPSTREAM_PATH
                    or evidence.get("reference") != (
                        f"github:yizhongw/self-instruct@{UPSTREAM_COMMIT}:"
                        f"{UPSTREAM_PATH}#{evidence.get('source_id')}"
                    )):
                raise ValueError("Missing pinned licensed external source or provenance")
    # Digest is length-delimited over both source-file byte sequences: a review
    # cannot remain valid if either candidate file changes, even only metadata.
    bound_bytes = (
        b"nestra-micro-68-12-v1\n" + str(len(local_bytes)).encode("ascii")
        + b"\n" + local_bytes + b"\n" + str(len(external_bytes)).encode("ascii")
        + b"\n" + external_bytes
    )
    return records, sha(bound_bytes)


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
                or len(entry["rationale"].strip()) < 25):
            raise ValueError(f"{rid}: missing explicit reviewer approval or reasoning")
        by_id[rid] = entry
    if set(by_id) != {r["id"] for r in records}:
        raise ValueError("Review IDs differ from frozen candidate IDs")
    reviewed = []
    verified_sources: set[str] = set()
    for record in records:
        approval = by_id[record["id"]]
        external = record.get("source") == "external_self_instruct_human_seed_candidate"
        evidence = record["provenance"]
        source_commit = UPSTREAM_COMMIT if external else PROVENANCE_COMMIT
        source = UPSTREAM_PATH if external else evidence["files"][0]
        if (approval.get("source_path") != source
                or approval.get("source_commit") != source_commit):
            raise ValueError(f"{record['id']}: reviewed source does not match provenance")
        if verify_git and not external and source not in verified_sources:
            check = subprocess.run(
                ["git", "cat-file", "-e", f"{PROVENANCE_COMMIT}:{source}"],
                cwd=ROOT, capture_output=True, check=False,
            )
            if check.returncode:
                raise ValueError(f"Missing immutable source blob for {source}")
            verified_sources.add(source)
        # External source comparison is an independent reviewer duty: the
        # upstream Git blob is not present in this checkout. Preserve the exact
        # upstream commit, licence, source ID and reviewer attribution instead
        # of misrepresenting the example as local synthetic material.
        converted = json.loads(json.dumps(record))
        converted["quality_tier"] = "B" if external else "C"
        converted["verification_status"] = "source_verified" if external else "synthetic_validated"
        converted["source"] = (
            "self_instruct_upstream_reviewed_seed"
            if external else "project_owned_independently_reviewed_synthetic"
        )
        converted["tags"] = [
            item for item in converted.get("tags", [])
            if item not in {"candidate_unapproved", "unapproved_external"}
        ]
        converted["metadata"] = {
            "candidate_sha256": candidate_sha,
            "reviewed_by": approval["reviewer"].strip(),
            "review_source_commit": source_commit,
            "synthetic_validated": not external,
            "source_verified": external,
        }
        converted["provenance"]["review"] = {
            "source_commit": source_commit,
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
