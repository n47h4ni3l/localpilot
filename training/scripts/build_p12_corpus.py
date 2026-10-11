#!/usr/bin/env python3
"""Materialize Nestra P1.2's locked, provenance-preserving 1,000-record corpus.

All sources are already checked in. This command does not download data, alter
model weights, start training, or incorporate private evaluation transcripts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "training/manifests/p12_selection.json"
DATASET = ROOT / "training/datasets/p12_corpus_1000.jsonl"
MANIFEST = ROOT / "training/manifests/p12_corpus_manifest.json"


def git_blob_sha1(contents: bytes) -> str:
    prefix = b"blob " + str(len(contents)).encode("ascii") + bytes([0])
    return hashlib.sha1(prefix + contents).hexdigest()


def read_source(path: Path, expected_sha: str) -> list[dict]:
    raw = path.read_bytes()
    actual = git_blob_sha1(raw)
    if actual != expected_sha:
        raise RuntimeError(f"Source blob changed: {path.relative_to(ROOT)}: {actual} != {expected_sha}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build() -> tuple[bytes, dict]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("artifact_type") != "nestra_p12_locked_selection":
        raise RuntimeError("Unexpected selection manifest")
    if selection.get("parent_manifest") != "training/lineage/package-1.json":
        raise RuntimeError("Wrong cumulative P1 training parent")
    ids = selection["selection_ids"]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise RuntimeError("Selection must contain exactly 1,000 unique record IDs")

    available: dict[str, dict] = {}
    for relative, sha in selection["source_blob_shas"].items():
        if not relative.startswith("training/datasets/"):
            raise RuntimeError("Selection source escaped the datasets directory")
        for item in read_source(ROOT / relative, sha):
            record_id = item["id"]
            if record_id in available:
                raise RuntimeError(f"Duplicate source ID: {record_id}")
            available[record_id] = item

    rows = []
    normalized_prompts: set[str] = set()
    families: dict[str, str] = {}
    source_counts: dict[str, Counter] = {}
    splits = Counter()
    expanded = Counter()
    for record_id in ids:
        if record_id not in available:
            raise RuntimeError(f"Selected record missing from verified sources: {record_id}")
        row = available[record_id]
        if row["split"] not in {"train", "validation"}:
            raise RuntimeError(f"Unwanted evaluation split: {record_id}")
        if not row["messages"] or row["messages"][-1].get("role") != "assistant":
            raise RuntimeError(f"Unfinished assistant trajectory: {record_id}")
        if not str(row["messages"][-1].get("content") or "").strip():
            raise RuntimeError(f"Blank final assistant reply: {record_id}")
        question = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        normalized = " ".join(question.split()).casefold()
        if not normalized or normalized in normalized_prompts:
            raise RuntimeError(f"Duplicate/empty user prompt: {record_id}")
        normalized_prompts.add(normalized)
        family = (row.get("metadata") or {}).get("family_id")
        if family:
            prior = families.setdefault(str(family), row["split"])
            if prior != row["split"]:
                raise RuntimeError(f"Source family crosses train/validation: {record_id}")
        source_counts.setdefault(row["source"], Counter())[row["split"]] += 1
        splits[row["split"]] += 1
        expanded[row["split"]] += int(row.get("metadata", {}).get("training_example_count", 1))
        rows.append(row)

    split_counts = {"train": splits["train"], "validation": splits["validation"]}
    examples = {
        "train": expanded["train"], "validation": expanded["validation"],
        "total": expanded["train"] + expanded["validation"],
    }
    if split_counts != selection["split_counts"]:
        raise RuntimeError(f"Split counts changed: {split_counts}")
    if examples != selection["training_example_counts"]:
        raise RuntimeError(f"Expanded target counts changed: {examples}")
    source_summary = {k: {"train": v["train"], "validation": v["validation"]}
                      for k, v in sorted(source_counts.items())}
    if source_summary != selection["source_counts"]:
        raise RuntimeError("Source composition changed")

    # The pinned Eval v1 one-way manifest is checked without exposing eval answers.
    from training_common import find_manifest_leaks, load_eval_manifest
    heldout = load_eval_manifest(ROOT / "training/manifests/eval_v1_manifest.json")
    overlaps = find_manifest_leaks(rows, heldout)
    if overlaps:
        raise RuntimeError(f"Locked corpus collides with frozen Eval v1: {overlaps[:5]}")

    content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                      for r in rows).encode("utf-8")
    corpus_manifest = {
        "schema_version": 1,
        # Current verified trainer's dry-run contract requires this artifact type.
        "artifact_type": "external_corpus_manifest",
        "name": "Nestra P1.2 curated corpus (locked selection, 2026-10-12)",
        "selection_manifest": str(SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "records": 1000,
        "split_counts": split_counts,
        "training_examples": examples["total"],
        "training_example_counts": examples,
        "corpus_sha256": hashlib.sha256(content).hexdigest(),
        "source_counts": source_summary,
        "source_blob_shas": selection["source_blob_shas"],
        "parent_manifest": selection["parent_manifest"],
        "notes": "50 author-curated source-informed examples are not independently scored; all others preserve existing source provenance. The selection does not use private P1.1 regression transcripts.",
    }
    return content, corpus_manifest


def save_immutable(path: Path, content: bytes, *, check_only: bool) -> str:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing artifact differs; refusing overwrite: {path}")
        return "verified"
    if check_only:
        return "validated-in-memory (not yet written)"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".p12-new")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate locked selection and existing materialization without writing")
    args = parser.parse_args()
    corpus, manifest = build()
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    corpus_status = save_immutable(DATASET, corpus, check_only=args.check)
    manifest_status = save_immutable(MANIFEST, manifest_bytes, check_only=args.check)
    print(json.dumps({
        "dataset": str(DATASET), "dataset_status": corpus_status,
        "manifest": str(MANIFEST), "manifest_status": manifest_status,
        "records": manifest["records"], "split_counts": manifest["split_counts"],
        "training_example_counts": manifest["training_example_counts"],
        "corpus_sha256": manifest["corpus_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
