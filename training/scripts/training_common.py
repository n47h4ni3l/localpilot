"""Shared, stdlib-only helpers for LocalPilot training tooling."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"[\w.+#:/-]+", re.UNICODE)
CONTENT_SHINGLE_TOKENS = 16
MANIFEST_SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return " ".join(value.casefold().split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def message_text(record: dict[str, Any], *, prompt_only: bool = False) -> str:
    parts: list[str] = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if prompt_only and role not in {"system", "user"}:
            continue
        parts.append(f"{role}: {message.get('content', '')}")
    return "\n".join(parts)


def content_text(record: dict[str, Any]) -> str:
    parts = [message_text(record)]
    expected = record.get("expected_behavior")
    if isinstance(expected, str):
        parts.append(expected)
    elif isinstance(expected, list):
        parts.extend(str(item) for item in expected)
    return "\n".join(parts)


def token_shingle_hashes(value: str, *, width: int = CONTENT_SHINGLE_TOKENS) -> set[str]:
    tokens = TOKEN_RE.findall(normalize_text(value))
    if len(tokens) < width:
        return set()
    return {
        sha256_text(" ".join(tokens[index : index + width]))
        for index in range(len(tokens) - width + 1)
    }


def record_fingerprints(record: dict[str, Any]) -> dict[str, Any]:
    prompt = normalize_text(message_text(record, prompt_only=True))
    content = normalize_text(content_text(record))
    return {
        "normalized_prompt_sha256": sha256_text(prompt),
        "normalized_content_sha256": sha256_text(content),
        "content_shingle_sha256": sorted(token_shingle_hashes(content)),
        # Role-independent fingerprints catch a short prompt/rubric copied into
        # an assistant response or metadata, where prompt-only hashes cannot.
        "content_fragments": sorted(
            {
                (len(normalize_text(value).split()), sha256_text(normalize_text(value)))
                for value in protected_fragments(record)
                if normalize_text(value)
            }
        ),
    }


def protected_fragments(record: dict[str, Any]) -> list[str]:
    fragments = [
        message["content"]
        for message in record.get("messages", [])
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
    ]
    expected = record.get("expected_behavior", [])
    if isinstance(expected, str):
        fragments.append(expected)
    elif isinstance(expected, list):
        fragments.extend(item for item in expected if isinstance(item, str))
    return fragments


def string_values(value: Any) -> Iterable[str]:
    """Scan all candidate text, including metadata and nested provenance."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from string_values(nested)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected an object in {path}:{line_number}.")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_eval_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_eval_manifest(value)
    return value


def validate_eval_manifest(value: Any) -> None:
    """Fail closed on incomplete or edited manifests; only hashes/IDs are valid."""
    if not isinstance(value, dict):
        raise RuntimeError("Held-out manifest must be an object.")
    if value.get("artifact_type") != "held_out_eval_manifest":
        raise RuntimeError("Not a held-out evaluation manifest.")
    allowed = {
        "schema_version", "artifact_type", "eval_name", "created_at", "record_count",
        "content_shingle_tokens", "records_sha256", "contains_plaintext_eval_content",
        "contains_prompts", "contains_rubrics", "contains_answers", "records",
    }
    if set(value) != allowed or value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Unsupported or malformed held-out manifest schema; regenerate it.")
    if any(value.get(field) is not False for field in (
        "contains_plaintext_eval_content", "contains_prompts", "contains_rubrics", "contains_answers"
    )):
        raise RuntimeError("Held-out manifest must contain no plaintext eval content.")
    rows = value.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Held-out manifest must have a non-empty record list.")
    if type(value.get("record_count")) is not int or value["record_count"] != len(rows):
        raise RuntimeError("Held-out manifest record count mismatch.")
    if value.get("content_shingle_tokens") != CONTENT_SHINGLE_TOKENS:
        raise RuntimeError("Unsupported held-out manifest shingle width.")
    keys = {
        "id", "id_sha256", "source_file", "raw_record_sha256", "normalized_prompt_sha256",
        "normalized_content_sha256", "content_shingle_sha256", "content_fragments",
    }
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or set(item) != keys:
            raise RuntimeError("Malformed held-out manifest record.")
        identifier = item["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", identifier) or identifier in seen:
            raise RuntimeError("Invalid or duplicate held-out manifest ID.")
        seen.add(identifier)
        if item["id_sha256"] != sha256_text(identifier):
            raise RuntimeError("Held-out manifest ID hash mismatch.")
        for key in ("id_sha256", "raw_record_sha256", "normalized_prompt_sha256", "normalized_content_sha256"):
            if not isinstance(item[key], str) or not SHA256_RE.fullmatch(item[key]):
                raise RuntimeError("Invalid held-out manifest hash.")
        source = item["source_file"]
        if not isinstance(source, str) or not source.endswith(".jsonl") or "\\" in source or source.startswith("/") or ".." in source.split("/"):
            raise RuntimeError("Invalid held-out manifest source path.")
        shingles = item["content_shingle_sha256"]
        if not isinstance(shingles, list) or any(not isinstance(item, str) or not SHA256_RE.fullmatch(item) for item in shingles) or shingles != sorted(set(shingles)):
            raise RuntimeError("Invalid held-out manifest shingles.")
        fragments = item["content_fragments"]
        if not isinstance(fragments, list) or not fragments:
            raise RuntimeError("Held-out manifest lacks protected fragments.")
        for fragment in fragments:
            if not isinstance(fragment, (tuple, list)) or len(fragment) != 2 or type(fragment[0]) is not int or fragment[0] < 1 or not isinstance(fragment[1], str) or not SHA256_RE.fullmatch(fragment[1]):
                raise RuntimeError("Invalid held-out manifest fragment hash.")
    if rows != sorted(rows, key=lambda item: item["id"]):
        raise RuntimeError("Held-out manifest IDs must be sorted.")
    payload = [{key: data for key, data in item.items() if key != "source_file"} for item in rows]
    if value.get("records_sha256") != sha256_json(payload):
        raise RuntimeError("Held-out manifest integrity digest mismatch.")


def find_manifest_leaks(
    records: Iterable[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, str]]:
    validate_eval_manifest(manifest)
    prompt_hashes: dict[str, str] = {}
    content_hashes: dict[str, str] = {}
    shingle_hashes: dict[str, str] = {}
    fragment_hashes: dict[int, dict[str, str]] = {}
    held_out_ids: set[str] = set()
    for item in manifest["records"]:
        if not isinstance(item, dict):
            continue
        eval_id = str(item.get("id") or item.get("id_sha256") or "unknown")
        held_out_ids.add(eval_id)
        prompt_hashes[str(item.get("normalized_prompt_sha256") or "")] = eval_id
        content_hashes[str(item.get("normalized_content_sha256") or "")] = eval_id
        for digest in item.get("content_shingle_sha256", []):
            shingle_hashes[str(digest)] = eval_id
        for width, digest in item["content_fragments"]:
            fragment_hashes.setdefault(width, {})[digest] = eval_id

    collisions: list[dict[str, str]] = []
    for record in records:
        record_id = str(record.get("id") or "unknown")
        if record_id in held_out_ids:
            collisions.append({"record_id": record_id, "kind": "held_out_id", "eval_id": record_id})
        fingerprints = record_fingerprints(record)
        prompt = fingerprints["normalized_prompt_sha256"]
        content = fingerprints["normalized_content_sha256"]
        if prompt in prompt_hashes:
            collisions.append({"record_id": record_id, "kind": "normalized_prompt", "eval_id": prompt_hashes[prompt]})
        if content in content_hashes:
            collisions.append({"record_id": record_id, "kind": "normalized_content", "eval_id": content_hashes[content]})
        candidate_values = list(string_values(record))
        candidate_shingles: set[str] = set(fingerprints["content_shingle_sha256"])
        for value in candidate_values:
            candidate_shingles.update(token_shingle_hashes(value))
        shared = candidate_shingles.intersection(shingle_hashes)
        if shared:
            digest = sorted(shared)[0]
            collisions.append({"record_id": record_id, "kind": "content_16_token_shingle", "eval_id": shingle_hashes[digest]})
        fragment_matches: set[str] = set()
        for value in candidate_values:
            tokens = normalize_text(value).split()
            for width, hashes in fragment_hashes.items():
                for offset in range(len(tokens) - width + 1):
                    digest = sha256_text(" ".join(tokens[offset : offset + width]))
                    if digest in hashes:
                        fragment_matches.add(hashes[digest])
        for eval_id in sorted(fragment_matches):
            collisions.append({"record_id": record_id, "kind": "normalized_content_fragment", "eval_id": eval_id})
    return collisions
