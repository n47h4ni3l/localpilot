"""Preserve the accepted Nestra P1.2 Ollama identity in an offline bundle.

No Ollama server calls, model execution, or global-store mutations occur here.
Destinations must be new directories. Failed partial copies are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


ACCEPTED_MODEL = "nestra:p1.2"
ACCEPTED_MANIFEST_SHA256 = "c356171d9a54f773c0064a56b5af028b209f61355ba68f67119d857c83bf7f46"
ACCEPTED_GGUF_SHA256 = "1f9c085c7333c6ea98709327f956e046da9e9241e75edae31bc81ff875bba4a1"
COPY_CHUNK_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_BLOB_BYTES = 256 * 1024**3
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})\Z")
_MANIFEST_PATHS = (
    "manifests-v2/ollama.com/library/nestra/p1.2",
    "manifests/registry.ollama.ai/library/nestra/p1.2",
    "manifests-v2/registry.ollama.ai/library/nestra/p1.2",
    "manifests/ollama.com/library/nestra/p1.2",
)


def _check_no_links(path: Path) -> None:
    """Reject Windows junctions/reparse points as well as ordinary symlinks."""
    current = path.absolute()
    for item in (current, *current.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Linked or reparse-point paths are not allowed: {item}")


def _root(path: str | Path) -> Path:
    result = Path(path).absolute()
    _check_no_links(result)
    if not result.is_dir():
        raise ValueError(f"Expected an existing directory: {result}")
    return result.resolve()


def _safe_file(root: Path, relative: str) -> Path:
    # All callers derive relative names from constants or validated digests.
    path = root / relative
    _check_no_links(path)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Expected a regular file inside the model directory: {relative}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Not a regular file: {relative}")
    return path


def _small_read(path: Path) -> bytes:
    _check_no_links(path)
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError(f"Model metadata is too large: {path.name}")
    with path.open("rb") as stream:
        content = stream.read(_MAX_METADATA_BYTES + 1)
    if len(content) > _MAX_METADATA_BYTES:
        raise ValueError(f"Model metadata is too large: {path.name}")
    return content


def _json(content: bytes) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(content, object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("Model metadata must be a JSON object")
    return value


def validate_lineage(path: str | Path) -> None:
    """Check the release selects the owner-accepted production P1.2 lineage."""
    lineage = _json(_small_read(Path(path)))
    deployment = lineage.get("deployment", {})
    if (
        lineage.get("package") != 3
        or lineage.get("accepted_as_lineage_head") is not True
        or deployment.get("model") != ACCEPTED_MODEL
        or deployment.get("digest") != ACCEPTED_MANIFEST_SHA256
        or deployment.get("gguf_sha256") != ACCEPTED_GGUF_SHA256
        or deployment.get("guide_only") is not True
        or type(deployment.get("temperature")) not in (int, float)
        or deployment["temperature"] != 1.0
    ):
        raise ValueError("Lineage is not the exact accepted Guide-only Nestra P1.2")


def _manifest(raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_MANIFEST_SHA256:
        raise ValueError("Ollama manifest digest does not match accepted Nestra P1.2")
    value = _json(raw)
    layers = value.get("layers")
    if value.get("schemaVersion") != 2 or not isinstance(layers, list) or not 1 <= len(layers) <= 32:
        raise ValueError("Unsupported Ollama manifest")
    descriptors = [value.get("config"), *layers]
    blobs: list[dict[str, Any]] = []
    seen: dict[str, tuple[int, str]] = {}
    model_hashes = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid Ollama blob descriptor")
        digest = descriptor.get("digest")
        size = descriptor.get("size")
        media_type = descriptor.get("mediaType")
        match = _DIGEST.fullmatch(digest) if isinstance(digest, str) else None
        if (
            match is None or type(size) is not int or not 0 < size <= _MAX_BLOB_BYTES
            or not isinstance(media_type, str) or not media_type.startswith("application/")
        ):
            raise ValueError("Invalid Ollama blob digest, size, or media type")
        sha256 = match[1]
        if media_type == "application/vnd.ollama.image.model":
            model_hashes.append(sha256)
        if sha256 in seen:
            if seen[sha256] != (size, media_type):
                raise ValueError("Conflicting Ollama blob descriptors")
            continue
        seen[sha256] = (size, media_type)
        blobs.append({"sha256": sha256, "size": size, "media_type": media_type})
    if model_hashes != [ACCEPTED_GGUF_SHA256] or sum(item["size"] for item in blobs) > _MAX_BLOB_BYTES:
        raise ValueError("Manifest does not contain exactly the accepted Nestra model")
    return value, blobs


def _spec(blobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": ACCEPTED_MODEL,
        "manifest_sha256": ACCEPTED_MANIFEST_SHA256,
        "gguf_sha256": ACCEPTED_GGUF_SHA256,
        "manifest_file": "manifest.json",
        "blobs": blobs,
        "total_blob_bytes": sum(item["size"] for item in blobs),
    }


def inspect_source(source_store: str | Path, lineage_path: str | Path) -> dict[str, Any]:
    """Inspect bounded metadata and file sizes; does not read/hash model weights."""
    validate_lineage(lineage_path)
    root = _root(source_store)
    manifests = []
    for relative in _MANIFEST_PATHS:
        path = root / relative
        _check_no_links(path)
        if path.exists():
            raw = _small_read(_safe_file(root, relative))
            _manifest(raw)  # A conflicting installed alias must fail, not be ignored.
            manifests.append(raw)
    if not manifests:
        raise ValueError("Accepted Nestra P1.2 alias is absent from the source store")
    raw = manifests[0]
    _, blobs = _manifest(raw)
    for item in blobs:
        path = _safe_file(root, f"blobs/sha256-{item['sha256']}")
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Source blob size mismatch: {item['sha256']}")
    return _spec(blobs)


def _fresh_destination(path: str | Path) -> Path:
    target = Path(path).absolute()
    _check_no_links(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing destination: {target}")
    if not target.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    target.mkdir(exist_ok=False)
    return target.resolve()


def _separate_destination(source: Path, destination: str | Path) -> None:
    target = Path(destination).absolute()
    _check_no_links(target)
    target = target.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Destination and source must be separate directories")


def _write_new(path: Path, content: bytes) -> None:
    _check_no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _stream_blob(source: Path, descriptor: dict[str, Any], destination: Path | None = None) -> None:
    """Verify while copying, with bounded memory and no redundant full-file pass."""
    _check_no_links(source)
    if destination is not None:
        _check_no_links(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    digest = hashlib.sha256()
    with source.open("rb") as input_stream:
        before = os.fstat(input_stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != descriptor["size"]:
            raise ValueError(f"Blob size mismatch: {source.name}")
        output_stream = destination.open("xb") if destination is not None else None
        try:
            while chunk := input_stream.read(COPY_CHUNK_BYTES):
                copied += len(chunk)
                if copied > descriptor["size"]:
                    raise ValueError(f"Blob grew while reading: {source.name}")
                digest.update(chunk)
                if output_stream is not None:
                    output_stream.write(chunk)
            after = os.fstat(input_stream.fileno())
            if copied != descriptor["size"] or digest.hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Blob SHA-256 or size mismatch: {source.name}")
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"Blob changed while reading: {source.name}")
            if output_stream is not None:
                output_stream.flush()
                os.fsync(output_stream.fileno())
        finally:
            if output_stream is not None:
                output_stream.close()


def export_model(source_store: str | Path, destination: str | Path, lineage_path: str | Path) -> dict[str, Any]:
    """Copy/hash the exact manifest and every referenced blob into a new pack."""
    spec = inspect_source(source_store, lineage_path)
    source = _root(source_store)
    raw = next(_small_read(_safe_file(source, relative)) for relative in _MANIFEST_PATHS if (source / relative).exists())
    _, blobs = _manifest(raw)
    if _spec(blobs) != spec:
        raise ValueError("Source manifest changed after inspection")
    _separate_destination(source, destination)
    target = _fresh_destination(destination)
    for item in blobs:
        name = f"blobs/sha256-{item['sha256']}"
        _stream_blob(_safe_file(source, name), item, target / name)
    _write_new(target / "manifest.json", raw)
    _write_new(target / "bundle-model.json", (json.dumps(spec, indent=2) + "\n").encode())
    return spec


def _read_bundle(bundle: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source = _root(bundle)
    raw = _small_read(_safe_file(source, "manifest.json"))
    _, blobs = _manifest(raw)
    spec = _spec(blobs)
    if _json(_small_read(_safe_file(source, "bundle-model.json"))) != spec:
        raise ValueError("Model bundle specification does not match its exact manifest")
    for item in blobs:
        blob = _safe_file(source, f"blobs/sha256-{item['sha256']}")
        if blob.stat().st_size != item["size"]:
            raise ValueError(f"Bundle blob size mismatch: {item['sha256']}")
    return source, raw, spec


def verify_bundle(bundle: str | Path) -> dict[str, Any]:
    """Read/hash a completed bundle without changing it or loading the model."""
    source, _, spec = _read_bundle(bundle)
    for item in spec["blobs"]:
        _stream_blob(_safe_file(source, f"blobs/sha256-{item['sha256']}"), item)
    return spec


def install_model(bundle: str | Path, destination: str | Path) -> dict[str, Any]:
    """Populate a NEW bundle-private OLLAMA_MODELS store, preserving raw identity.

    Start the separately packaged Ollama runtime with OLLAMA_MODELS pointing to
    this directory. Neither production aliases nor their blobs are modified.
    """
    source, raw, spec = _read_bundle(bundle)
    _separate_destination(source, destination)
    target = _fresh_destination(destination)
    for item in spec["blobs"]:
        name = f"blobs/sha256-{item['sha256']}"
        _stream_blob(_safe_file(source, name), item, target / name)
    # Both supported on-disk layouts receive the identical raw manifest. No
    # GGUF reimport/Modelfile regeneration can drop template/system/params.
    for relative in _MANIFEST_PATHS[:2]:
        _write_new(target / relative, raw)
    _write_new(target / "nestra-bundle-installed.json", (json.dumps(spec, indent=2) + "\n").encode())
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "export"):
        child = commands.add_parser(command)
        child.add_argument("--source-store", required=True)
        child.add_argument("--lineage", required=True)
        if command == "export":
            child.add_argument("--destination", required=True)
    for command in ("verify", "install"):
        child = commands.add_parser(command)
        child.add_argument("--bundle", required=True)
        if command == "install":
            child.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_source(args.source_store, args.lineage)
    elif args.command == "export":
        result = export_model(args.source_store, args.destination, args.lineage)
    elif args.command == "verify":
        result = verify_bundle(args.bundle)
    else:
        result = install_model(args.bundle, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
