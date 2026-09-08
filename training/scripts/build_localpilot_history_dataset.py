#!/usr/bin/env python3
"""Build a verified LocalPilot-owned training corpus without reading held-out evals."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import (
    ROOT,
    find_manifest_leaks,
    load_eval_manifest,
    load_jsonl,
    message_text,
    normalize_text,
    sha256_json,
    write_json,
    write_jsonl,
)
from validate_dataset import Location, _validate_record_shape


DEFAULT_SOURCE = ROOT / "training" / "sources" / "corpus_v1_seed_sources.jsonl"
DEFAULT_OUTPUT = ROOT / "training" / "datasets" / "corpus_v1_seed.jsonl"
DEFAULT_MANIFEST = ROOT / "training" / "manifests" / "eval_v1_manifest.json"
DEFAULT_REPORT = ROOT / "training" / "reports" / "corpus_v1_stats.json"

FORBIDDEN_PATH_MARKERS = (
    "training/evals/",
    "training/reports/",
    "training/evolution_execution/acceptance/",
    "scorecard",
    "hidden_acceptance",
    "acceptance_fixture",
    "target_patch",
    "benchmark_answer",
    "rubric",
    "solution_patch",
    "expected_patch",
)
ALLOWED_SOURCE_PREFIXES = (
    "training/sources/",
    "localpilot/",
    "tests/",
    "training/tests/",
    "training/scripts/",
    "training/schema/",
    "docs/",
    "scripts/",
    "training/baselines/",
    "training/manifests/",
    "training/evolution_execution/README.md",
    "README.md",
    "ARCHITECTURE.md",
    "SECURITY.md",
    "config.example.toml",
    "pyproject.toml",
)


def repository_relative(path: Path, root: Path = ROOT) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Source path escapes the LocalPilot repository: {path}") from exc
    return relative


def assert_source_path_allowed(path: Path, root: Path = ROOT) -> str:
    # Check the spelling before resolving so an eval-path alias cannot point to
    # an allowed source. Then reject links, including Windows directory junctions.
    try:
        lexical = path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Source path escapes the LocalPilot repository: {path}") from exc
    if ".." in Path(lexical).parts or any(marker in (lexical.casefold() + "/") for marker in FORBIDDEN_PATH_MARKERS):
        raise RuntimeError(f"Forbidden training source path: {lexical}")
    for component in [path, *path.parents]:
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise RuntimeError(f"Training source must not use symlinks or junctions: {lexical}")
    relative = repository_relative(path, root)
    lowered = relative.casefold()
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        raise RuntimeError(f"Forbidden training source path: {relative}")
    if not any(relative == prefix.rstrip("/") or (prefix.endswith("/") and relative.startswith(prefix)) for prefix in ALLOWED_SOURCE_PREFIXES):
        raise RuntimeError(f"Training source is outside the project-owned allowlist: {relative}")
    return relative


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def changed_paths(repo: Path, commit: str) -> set[str]:
    _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    output = _git(
        repo,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-m",
        commit,
    )
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def validate_provenance(record: dict[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{record.get('id')}: missing provenance")
    commit = str(provenance.get("commit") or "")
    pr = provenance.get("pr")
    files = provenance.get("files")
    tests = provenance.get("tests")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or type(pr) is not int or pr < 1 or not isinstance(files, list) or not files:
        raise RuntimeError(f"{record.get('id')}: provenance requires commit, numeric PR, and files")
    if provenance.get("repository") != "n47h4ni3l/localpilot":
        raise RuntimeError(f"{record.get('id')}: provenance repository must be n47h4ni3l/localpilot")
    if not isinstance(tests, list) or not tests:
        raise RuntimeError(f"{record.get('id')}: provenance requires at least one verifying test")
    for relative in [*files, *tests]:
        if not isinstance(relative, str) or not relative or ":" in relative or "\\" in relative:
            raise RuntimeError(f"{record.get('id')}: invalid provenance path")
        candidate = repo / str(relative)
        assert_source_path_allowed(candidate, repo)
    remote = _git(repo, "remote", "get-url", "origin").strip().removesuffix(".git").casefold()
    if remote not in {"https://github.com/n47h4ni3l/localpilot", "git@github.com:n47h4ni3l/localpilot", "ssh://git@github.com/n47h4ni3l/localpilot"}:
        raise RuntimeError("Source repository origin must be n47h4ni3l/localpilot.")
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise RuntimeError("Verified history extraction requires complete Git history; fetch with --unshallow (CI checkout fetch-depth: 0).")
    observed = changed_paths(repo, commit)
    try:
        _git(repo, "rev-parse", "--verify", "refs/remotes/origin/main")
        history_ref = "refs/remotes/origin/main"
    except RuntimeError:
        history_ref = "refs/heads/main"
    _git(repo, "merge-base", "--is-ancestor", commit, history_ref)
    subject = _git(repo, "log", "-1", "--format=%s", commit).strip()
    if not re.search(rf"(?:\(#{pr}\)|Merge pull request #{pr}(?:\s|$))", subject):
        raise RuntimeError(f"{record.get('id')}: declared PR does not match the merged commit subject")
    for relative in files:
        if str(relative).replace("\\", "/") not in observed:
            raise RuntimeError(f"{record.get('id')}: {relative} was not changed by {commit}")
    evidence: dict[str, Any] = {
        "kind": "reviewed_history_derivation",
        "commit_subject": subject,
        "files": [],
        "tests": [],
    }
    for kind, paths in (("files", files), ("tests", tests)):
        for relative in paths:
            tree_entry = _git(repo, "ls-tree", commit, "--", relative).strip()
            descriptor, separator, observed_path = tree_entry.partition("\t")
            fields = descriptor.split()
            if not separator or observed_path != relative or len(fields) != 3 or fields[0] not in {"100644", "100755"} or fields[1] != "blob":
                raise RuntimeError(f"{record.get('id')}: provenance must reference a regular Git file at the commit: {relative}")
            # Extract immutable blob identity only. Test/fixture bytes are never
            # copied into the corpus; a test reference does not verify prose.
            evidence[kind].append({"path": relative, "git_blob_id": fields[2]})
    expected_reference = f"github:pr/{pr}@{commit}"
    if provenance.get("reference") != expected_reference:
        raise RuntimeError(f"{record.get('id')}: provenance.reference must be {expected_reference}")
    return evidence


def _provenance_complete(record: dict[str, Any]) -> bool:
    provenance = record.get("provenance")
    return bool(
        isinstance(provenance, dict)
        and provenance.get("repository")
        and provenance.get("reference")
        and provenance.get("commit")
        and provenance.get("pr")
        and provenance.get("files")
        and provenance.get("tests")
    )


def corpus_statistics(
    rows: list[dict[str, Any]], *, duplicates_rejected: int, leakage_rejected: int
) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))

    characters = sum(len(message_text(row)) for row in rows)
    split_counts = Counter(str(row.get("split")) for row in rows)
    train = split_counts.get("train", 0)
    validation = split_counts.get("validation", 0)
    active = train + validation
    return {
        "schema_version": 1,
        "artifact_type": "corpus_statistics",
        "record_count": len(rows),
        "counts": {
            "task_type": counts("task_type"),
            "quality_tier": counts("quality_tier"),
            "split": counts("split"),
            "source": counts("source"),
            "license": counts("license"),
        },
        "provenance": {
            "complete": sum(_provenance_complete(row) for row in rows),
            "incomplete": sum(not _provenance_complete(row) for row in rows),
            "completeness_percent": round(100 * sum(_provenance_complete(row) for row in rows) / len(rows), 2) if rows else 0,
        },
        "rejections": {
            "duplicates": duplicates_rejected,
            "held_out_leakage": leakage_rejected,
        },
        "size": {
            "characters": characters,
            "estimated_tokens": round(characters / 4),
        },
        "train_validation": {
            "train": train,
            "validation": validation,
            "train_ratio": round(train / active, 4) if active else None,
            "validation_ratio": round(validation / active, 4) if active else None,
        },
        "dataset_sha256": sha256_json(rows),
    }


def build_dataset(
    source_paths: Sequence[Path],
    eval_manifest_path: Path,
    *,
    repo: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_source_path_allowed(eval_manifest_path, repo)
    manifest = load_eval_manifest(eval_manifest_path)
    # Validate the entire source inventory before opening any candidate source.
    for path in source_paths:
        assert_source_path_allowed(path, repo)
    rows: list[dict[str, Any]] = []
    for path in sorted(source_paths, key=lambda value: value.as_posix()):
        for index, row in enumerate(load_jsonl(path), 1):
            if row.get("quality_tier") == "D":
                raise RuntimeError(f"{row.get('id')}: Tier D can never enter a corpus build")
            errors: list[str] = []
            _validate_record_shape(row, Location(path, index), errors, [])
            if errors:
                raise RuntimeError("Invalid corpus source schema: " + "; ".join(errors))
            rows.append(row)

    accepted: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    seen_prompts: set[str] = set()
    seen_ids: dict[str, str] = {}
    duplicates = 0
    for row in sorted(rows, key=lambda value: value["id"]):
        if row.get("quality_tier") == "D":
            raise RuntimeError(f"{row.get('id')}: Tier D can never enter a corpus build")
        if row.get("split") not in {"train", "validation"}:
            raise RuntimeError(f"{row.get('id')}: corpus source split must be train or validation")
        if row.get("source") != "localpilot_verified_history" or row.get("license") != "project_owned":
            raise RuntimeError(f"{row.get('id')}: Corpus v1 seed accepts project-owned verified history only")
        if row.get("quality_tier") == "C" and row.get("verification_status") != "synthetic_validated":
            raise RuntimeError(f"{row.get('id')}: Tier C requires independent synthetic validation evidence")
        if row.get("quality_tier") == "C":
            raise RuntimeError(f"{row.get('id')}: this history-only builder has no synthetic validation executor; use reviewed Tier A source derivations")
        evidence = validate_provenance(row, repo)
        row = {**row, "provenance": {**row["provenance"], "source_evidence": evidence}}
        fingerprint = normalize_text(message_text(row))
        prompt_fingerprint = normalize_text(message_text(row, prompt_only=True))
        record_id = str(row.get("id") or "")
        if record_id in seen_ids and seen_ids[record_id] != sha256_json(row):
            raise RuntimeError(f"{record_id}: duplicate ID has conflicting content/provenance")
        if record_id in seen_ids or fingerprint in seen_messages or prompt_fingerprint in seen_prompts:
            duplicates += 1
            continue
        seen_ids[record_id] = sha256_json(row)
        seen_messages.add(fingerprint)
        seen_prompts.add(prompt_fingerprint)
        accepted.append(row)

    leaks = find_manifest_leaks(accepted, manifest)
    leaking_ids = {item["record_id"] for item in leaks}
    filtered = [row for row in accepted if str(row.get("id")) not in leaking_ids]
    stats = corpus_statistics(filtered, duplicates_rejected=duplicates, leakage_rejected=len(leaking_ids))
    stats["leakage_collisions"] = leaks
    stats["eval_manifest_sha256"] = manifest["records_sha256"]
    return filtered, stats


def assert_output_path_allowed(path: Path, directory: str, suffix: str, repo: Path = ROOT) -> None:
    root = repo / "training" / directory
    if not path.absolute().is_relative_to(root.absolute()) or not path.resolve().is_relative_to(root.resolve()) or path.suffix != suffix:
        raise RuntimeError(f"Output must be a {suffix} file under training/{directory}.")
    for component in [path, *path.parents]:
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise RuntimeError("Training output must not use symlinks or junctions.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the verified LocalPilot Corpus v1 seed.")
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true", help="verify generated output is reproducible")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = args.source or [DEFAULT_SOURCE]
    assert_output_path_allowed(args.output, "datasets", ".jsonl")
    assert_output_path_allowed(args.report, "reports", ".json")
    rows, stats = build_dataset([path.absolute() for path in sources], args.eval_manifest.absolute())
    if stats["rejections"]["held_out_leakage"]:
        write_json(args.report, stats)
        raise RuntimeError("Held-out leakage detected; no corpus was written.")
    if args.check:
        existing = load_jsonl(args.output)
        if existing != rows:
            raise RuntimeError("Tracked Corpus v1 seed is stale; regenerate it.")
    else:
        write_jsonl(args.output, rows)
    write_json(args.report, stats)
    print(
        f"Corpus v1: {len(rows)} verified examples; duplicates rejected={stats['rejections']['duplicates']}; "
        f"leakage rejected={stats['rejections']['held_out_leakage']}; digest={stats['dataset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
