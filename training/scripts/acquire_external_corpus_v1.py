#!/usr/bin/env python3
"""Acquire only the immutable upstream files approved for External Corpus v1.

Existing raw files are verified and reused in place. Downloads are staged in a
temporary directory on the source volume, verified against the tracked lock,
and installed with an exclusive hard-link operation. The script never deletes,
renames, edits, or overwrites an owner-provided raw file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_corpus_v1 import DATASETS, canonical_json, sha256_file


DEFAULT_ROOT = Path(r"E:\LocalPilot-Training-Data")
LOCK_PATH = SCRIPT_DIR.parent / "manifests" / "external_corpus_v1_acquisition_lock.json"
SnapshotDownload = Callable[..., str]
LOWER_HEX_40_RE = re.compile(r"[0-9a-f]{40}\Z")
LOWER_HEX_64_RE = re.compile(r"[0-9a-f]{64}\Z")


def _safe_relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or any(character in value for character in "*?[]")
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeError(f"unsafe {field} in acquisition lock: {value!r}")
    normalized = path.as_posix()
    if normalized != value or "\\" in value:
        raise RuntimeError(f"non-canonical {field} in acquisition lock: {value!r}")
    return value


def load_acquisition_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read acquisition lock: {path}") from exc
    if lock.get("format_version") != 1 or not isinstance(lock.get("datasets"), dict):
        raise RuntimeError("unsupported External Corpus v1 acquisition lock")
    if set(lock["datasets"]) != set(DATASETS):
        raise RuntimeError("acquisition lock dataset set does not match the corpus specification")

    for dataset, source in lock["datasets"].items():
        spec = DATASETS[dataset]
        if source.get("repo_id") != spec["repo_id"] or source.get("revision") != spec["revision"]:
            raise RuntimeError(f"acquisition lock does not match the pinned {dataset} source")
        _safe_relative_path(str(source.get("local_directory", "")), field="local_directory")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"acquisition lock has no files for {dataset}")
        remote_paths: set[str] = set()
        local_paths: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise RuntimeError(f"invalid file entry in acquisition lock for {dataset}")
            remote = _safe_relative_path(str(entry.get("remote_path", "")), field="remote_path")
            local = _safe_relative_path(str(entry.get("local_path", "")), field="local_path")
            if remote in remote_paths or local in local_paths:
                raise RuntimeError(f"duplicate file path in acquisition lock for {dataset}")
            remote_paths.add(remote)
            local_paths.add(local)
            if not isinstance(entry.get("size"), int) or entry["size"] < 0:
                raise RuntimeError(f"invalid locked size for {dataset}/{remote}")
            sha256 = entry.get("sha256")
            git_oid = entry.get("git_oid")
            if sha256 is not None and (
                not isinstance(sha256, str) or LOWER_HEX_64_RE.fullmatch(sha256) is None
            ):
                raise RuntimeError(f"invalid locked SHA-256 for {dataset}/{remote}")
            if git_oid is not None and (
                not isinstance(git_oid, str) or LOWER_HEX_40_RE.fullmatch(git_oid) is None
            ):
                raise RuntimeError(f"invalid locked Git object ID for {dataset}/{remote}")
            if sha256 is None and git_oid is None:
                raise RuntimeError(f"file has no content identity in acquisition lock: {dataset}/{remote}")
            if sha256 is not None and git_oid is not None:
                raise RuntimeError(f"file has ambiguous content identities in acquisition lock: {dataset}/{remote}")
        excluded = source.get("permanently_excluded", [])
        for remote in excluded:
            _safe_relative_path(str(remote), field="permanently_excluded")
            if remote in remote_paths:
                raise RuntimeError(f"excluded path is also approved for {dataset}: {remote}")
    return lock


ACQUISITION_LOCK = load_acquisition_lock()
ACQUISITION = {
    dataset: {
        "directory": source["local_directory"],
        # Kept as ``patterns`` for callers from PR #90, but every value is an
        # exact file name. Wildcards are deliberately forbidden.
        "patterns": [entry["remote_path"] for entry in source["files"]],
        "files": source["files"],
        "permanently_excluded": source.get("permanently_excluded", []),
    }
    for dataset, source in ACQUISITION_LOCK["datasets"].items()
}


def git_blob_oid(path: Path, *, size: int | None = None, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return Git's SHA-1 object ID for a regular (non-LFS-pointer) file."""

    file_size = path.stat().st_size if size is None else size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {file_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_file(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``path`` has exactly the locked length and identity."""

    if path.is_symlink():
        raise RuntimeError(f"locked source path must not be a symbolic link: {path}")
    if not path.is_file():
        raise RuntimeError(f"locked source path must be a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != entry["size"]:
        raise RuntimeError(
            f"locked source size mismatch for {path}: expected {entry['size']}, got {actual_size}"
        )
    if entry.get("sha256") is not None:
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"locked source SHA-256 mismatch for {path}")
        return {"size": actual_size, "sha256": actual, "verified": True}
    actual = git_blob_oid(path, size=actual_size)
    if actual != entry["git_oid"]:
        raise RuntimeError(f"locked source Git object mismatch for {path}")
    return {"size": actual_size, "git_oid": actual, "verified": True}


def _local_target(root: Path, directory: str, relative: str) -> Path:
    root = root.resolve()
    target = root / directory / Path(*PurePosixPath(relative).parts)
    # resolve(strict=False) follows any existing directory links while allowing
    # the final file to be absent. This prevents a crafted directory link from
    # redirecting an acquisition outside the selected source root.
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"locked local path escapes the source root: {relative}") from exc
    return target


def _dataset_paths(dataset: str, root: Path) -> list[tuple[dict[str, Any], Path]]:
    acquisition = ACQUISITION[dataset]
    return [
        (entry, _local_target(root, acquisition["directory"], entry["local_path"]))
        for entry in acquisition["files"]
    ]


def verify_dataset(dataset: str, root: Path, *, require_all: bool = True) -> dict[str, Any]:
    root = root.resolve()
    files: dict[str, Any] = {}
    missing: list[str] = []
    for entry, path in _dataset_paths(dataset, root):
        if not path.exists() and not path.is_symlink():
            missing.append(entry["local_path"])
            continue
        files[entry["local_path"]] = verify_locked_file(path, entry)
    if missing and require_all:
        raise RuntimeError(f"{dataset} is missing locked files: {', '.join(missing)}")
    return {
        "repo_id": ACQUISITION_LOCK["datasets"][dataset]["repo_id"],
        "revision": ACQUISITION_LOCK["datasets"][dataset]["revision"],
        "verified": len(files),
        "missing": missing,
        "files": files,
    }


def _default_snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the corpus extra: pip install -e .[corpus]") from exc
    return snapshot_download(**kwargs)


def _place_without_overwrite(staged: Path, target: Path, entry: dict[str, Any]) -> str:
    """Atomically expose a verified staged file, never replacing ``target``."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, target)
    except FileExistsError:
        verify_locked_file(target, entry)
        return "reused_after_race"
    except OSError as exc:
        # A same-volume hard link is the only installation operation used: it
        # is atomic and fails if the destination already exists. Falling back
        # to replace/copy would weaken the no-overwrite guarantee.
        raise RuntimeError(f"cannot atomically place locked source file {target}") from exc
    # A hard link names the already-verified staged inode; no second multi-GB
    # read is needed after the atomic link succeeds.
    return "installed"


def acquire(
    dataset: str,
    root: Path,
    max_workers: int,
    *,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> dict[str, Any]:
    """Verify all existing files and acquire only exact missing locked paths."""

    source = ACQUISITION_LOCK["datasets"][dataset]
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    before = verify_dataset(dataset, root, require_all=False)
    missing = set(before["missing"])
    if not missing:
        return {
            "dataset": dataset,
            "downloaded": [],
            "reused": sorted(before["files"]),
            "verified": len(before["files"]),
            "revision": source["revision"],
        }

    missing_entries = [entry for entry in source["files"] if entry["local_path"] in missing]
    remote_paths = [entry["remote_path"] for entry in missing_entries]
    if any(any(character in path for character in "*?[") for path in remote_paths):
        raise RuntimeError("wildcards are forbidden in locked acquisition paths")
    downloader = snapshot_download_fn or _default_snapshot_download

    downloaded: list[str] = []
    reused: list[str] = sorted(before["files"])
    # Preserve Hugging Face metadata and .incomplete chunks across process/app
    # interruptions. Revision scoping prevents bytes from a different snapshot
    # being resumed into this acquisition.
    stage = (
        root / ".external-corpus-v1-staging" / dataset / str(source["revision"])
    ).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        downloader(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            local_dir=stage,
            allow_patterns=remote_paths,
            token=os.environ.get("HF_TOKEN"),
            max_workers=max_workers,
        )
    ).resolve()
    # snapshot_download normally returns local_dir. Enforce that even an
    # injected or future downloader cannot direct us outside our staging
    # directory.
    try:
        snapshot.relative_to(stage)
    except ValueError as exc:
        raise RuntimeError("download snapshot escaped the acquisition staging directory") from exc

    staged_files: list[tuple[dict[str, Any], Path, Path]] = []
    for entry in missing_entries:
        staged = snapshot / Path(*PurePosixPath(entry["remote_path"]).parts)
        if staged.is_symlink() or not staged.is_file():
            raise RuntimeError(f"pinned download did not provide {entry['remote_path']}")
        verify_locked_file(staged, entry)
        target = _local_target(root, source["local_directory"], entry["local_path"])
        staged_files.append((entry, staged, target))

    # Nothing is placed until every downloaded byte has passed the lock.
    for entry, staged, target in staged_files:
        outcome = _place_without_overwrite(staged, target, entry)
        if outcome == "installed":
            downloaded.append(entry["local_path"])
        else:
            reused.append(entry["local_path"])

    return {
        "dataset": dataset,
        "downloaded": sorted(downloaded),
        "reused": sorted(reused),
        "verified": len(downloaded) + len(reused),
        "revision": source["revision"],
        "persistent_staging": str(stage),
    }


def verify_known_hashes(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    """Compatibility entry point that now verifies every locked source file."""

    selected = datasets or list(DATASETS)
    return {dataset: verify_dataset(dataset, root, require_all=True) for dataset in selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(ACQUISITION),
        help="Repeat to acquire a subset; default is all six",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--use-xet", action="store_true", help="Opt into Xet transport; standard resumable HTTP is the stable default")
    args = parser.parse_args(argv)
    if not 1 <= args.max_workers <= 8:
        parser.error("--max-workers must be between 1 and 8")
    root = args.source_root.resolve()
    if not args.use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    selected = args.dataset or list(DATASETS)
    acquired: list[dict[str, Any]] = []
    if args.verify_only:
        verified = verify_known_hashes(root, selected)
    else:
        root.mkdir(parents=True, exist_ok=True)
        for dataset in selected:
            acquired.append(acquire(dataset, root, args.max_workers))
        # acquire() has already verified every reused file and every staged
        # payload. Avoid re-reading tens of gigabytes solely to print output.
        verified = {
            result["dataset"]: {"revision": result["revision"], "verified": result["verified"], "missing": []}
            for result in acquired
        }
    print(canonical_json({"acquisition": acquired, "lock": str(LOCK_PATH), "verified": verified}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
