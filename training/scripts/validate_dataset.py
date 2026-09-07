#!/usr/bin/env python3
"""Validate LocalPilot training/evaluation JSONL datasets.

The validator is intentionally stdlib-only so it can run before any training
backend is selected. It validates record structure, data-authority policy,
duplicates, and held-out leakage across all supplied files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

QUALITY_TIERS = {"A", "B", "C", "D"}
SPLITS = {"train", "validation", "held_out_eval"}
VERIFICATION_STATUSES = {
    "human_verified",
    "ci_verified",
    "source_verified",
    "synthetic_validated",
    "review_required",
    "unverified",
}
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
DIFFICULTIES = {"easy", "medium", "hard", "expert"}
REQUIRED_FIELDS = {
    "id",
    "messages",
    "task_type",
    "source",
    "license",
    "quality_tier",
    "split",
    "verification_status",
    "provenance",
}
OPTIONAL_FIELDS = {
    "tags",
    "difficulty",
    "expected_behavior",
    "created_at",
    "metadata",
}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
TASK_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_:-]{1,79}$")

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class Location:
    path: Path
    line: int

    def label(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class LoadedRecord:
    location: Location
    value: dict[str, Any]
    exact_fingerprint: str
    normalized_messages_fingerprint: str
    normalized_prompt_fingerprint: str


@dataclass
class ValidationReport:
    records: list[LoadedRecord]
    errors: list[str]
    warnings: list[str]
    files_read: int = 0
    lines_read: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(message.get("role", "")), "content": _normalize_text(str(message.get("content", "")))}
        for message in messages
    ]


def _normalized_prompt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    prompt = [
        {"role": str(message.get("role", "")), "content": _normalize_text(str(message.get("content", "")))}
        for message in messages
        if message.get("role") in {"system", "user"}
    ]
    return prompt


def _exact_example_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return content-bearing fields while ignoring identity/provenance metadata."""
    return {
        "messages": record.get("messages"),
        "task_type": record.get("task_type"),
        "split": record.get("split"),
        "expected_behavior": record.get("expected_behavior"),
    }


def _parse_timestamp(value: str) -> bool:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _validate_expected_behavior(value: Any) -> bool:
    if _nonempty_string(value):
        return True
    if isinstance(value, list) and value:
        return all(_nonempty_string(item) for item in value)
    return False


def _validate_record_shape(record: Any, location: Location, errors: list[str], warnings: list[str]) -> dict[str, Any] | None:
    prefix = location.label()
    if not isinstance(record, dict):
        errors.append(f"{prefix}: record must be a JSON object")
        return None

    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append(f"{prefix}: missing required fields: {', '.join(missing)}")

    unknown = sorted(set(record) - ALLOWED_FIELDS)
    if unknown:
        errors.append(f"{prefix}: unknown top-level fields: {', '.join(unknown)}")

    record_id = record.get("id")
    if not _nonempty_string(record_id) or not ID_RE.fullmatch(str(record_id)):
        errors.append(f"{prefix}: id must match {ID_RE.pattern!r}")

    task_type = record.get("task_type")
    if not _nonempty_string(task_type) or not TASK_TYPE_RE.fullmatch(str(task_type)):
        errors.append(f"{prefix}: task_type must be lowercase snake_case (2-80 chars)")

    for field in ("source", "license"):
        if not _nonempty_string(record.get(field)):
            errors.append(f"{prefix}: {field} must be a non-empty string")

    tier = record.get("quality_tier")
    if tier not in QUALITY_TIERS:
        errors.append(f"{prefix}: quality_tier must be one of {sorted(QUALITY_TIERS)}")

    split = record.get("split")
    if split not in SPLITS:
        errors.append(f"{prefix}: split must be one of {sorted(SPLITS)}")

    status = record.get("verification_status")
    if status not in VERIFICATION_STATUSES:
        errors.append(
            f"{prefix}: verification_status must be one of {sorted(VERIFICATION_STATUSES)}"
        )

    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        errors.append(f"{prefix}: provenance must be a non-empty object")
    elif not _nonempty_string(provenance.get("reference")):
        errors.append(f"{prefix}: provenance.reference must be a non-empty string")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"{prefix}: messages must be a non-empty list")
        messages = []
    else:
        user_count = 0
        assistant_count = 0
        for index, message in enumerate(messages):
            item = f"{prefix}: messages[{index}]"
            if not isinstance(message, dict):
                errors.append(f"{item} must be an object")
                continue
            unknown_message_fields = sorted(set(message) - {"role", "content", "name"})
            if unknown_message_fields:
                errors.append(
                    f"{item} has unknown fields: {', '.join(unknown_message_fields)}"
                )
            role = message.get("role")
            if role not in MESSAGE_ROLES:
                errors.append(f"{item}.role must be one of {sorted(MESSAGE_ROLES)}")
            if not _nonempty_string(message.get("content")):
                errors.append(f"{item}.content must be a non-empty string")
            if "name" in message and not _nonempty_string(message.get("name")):
                errors.append(f"{item}.name must be a non-empty string when present")
            if role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
        if user_count == 0:
            errors.append(f"{prefix}: messages must contain at least one user message")

        if split in {"train", "validation"}:
            if assistant_count == 0:
                errors.append(f"{prefix}: {split} records require an assistant target message")
            elif isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
                errors.append(f"{prefix}: {split} records must end with an assistant message")
        elif split == "held_out_eval" and assistant_count:
            errors.append(
                f"{prefix}: held_out_eval records must not contain assistant answers; use expected_behavior"
            )

    if split in {"train", "validation"}:
        if tier == "D":
            errors.append(f"{prefix}: quality tier D may not enter {split}")
        if status in {"review_required", "unverified"}:
            errors.append(f"{prefix}: {split} records must be verified before use")
    elif split == "held_out_eval":
        if tier not in {"A", "B"}:
            errors.append(f"{prefix}: held_out_eval must use quality tier A or B")
        if status not in {"human_verified", "ci_verified", "source_verified"}:
            errors.append(f"{prefix}: held_out_eval must be independently verified")
        if not _validate_expected_behavior(record.get("expected_behavior")):
            errors.append(f"{prefix}: held_out_eval requires non-empty expected_behavior")

    tags = record.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(_nonempty_string(tag) for tag in tags):
            errors.append(f"{prefix}: tags must be a list of non-empty strings")
        elif len(tags) != len(set(tags)):
            warnings.append(f"{prefix}: tags contain duplicates")

    difficulty = record.get("difficulty")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        errors.append(f"{prefix}: difficulty must be one of {sorted(DIFFICULTIES)}")

    if "expected_behavior" in record and not _validate_expected_behavior(record.get("expected_behavior")):
        errors.append(f"{prefix}: expected_behavior must be a non-empty string or list of strings")

    created_at = record.get("created_at")
    if created_at is not None and (
        not _nonempty_string(created_at) or not _parse_timestamp(str(created_at))
    ):
        errors.append(f"{prefix}: created_at must be an ISO-8601 timestamp")

    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append(f"{prefix}: metadata must be an object")

    return record


def _expand_paths(paths: Sequence[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file()))
        else:
            expanded.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def validate_files(paths: Sequence[str | Path]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    loaded: list[LoadedRecord] = []
    expanded = _expand_paths(paths)
    report = ValidationReport(loaded, errors, warnings)

    if not expanded:
        errors.append("no JSONL files were found")
        return report

    for path in expanded:
        if not path.exists():
            errors.append(f"{path}: file does not exist")
            continue
        if not path.is_file():
            errors.append(f"{path}: not a regular file")
            continue
        if path.suffix.lower() != ".jsonl":
            warnings.append(f"{path}: validating non-.jsonl file")

        report.files_read += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    report.lines_read += 1
                    if not raw_line.strip():
                        continue
                    location = Location(path, line_number)
                    try:
                        raw = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            f"{location.label()}: invalid JSON ({exc.msg} at column {exc.colno})"
                        )
                        continue
                    record = _validate_record_shape(raw, location, errors, warnings)
                    if record is None or not isinstance(record.get("messages"), list):
                        continue
                    messages = [message for message in record["messages"] if isinstance(message, dict)]
                    loaded.append(
                        LoadedRecord(
                            location=location,
                            value=record,
                            exact_fingerprint=_sha256(_exact_example_payload(record)),
                            normalized_messages_fingerprint=_sha256(_normalized_messages(messages)),
                            normalized_prompt_fingerprint=_sha256(_normalized_prompt(messages)),
                        )
                    )
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: unable to read UTF-8 JSONL: {exc}")

    _validate_cross_record_rules(loaded, errors)
    return report


def _validate_cross_record_rules(records: Iterable[LoadedRecord], errors: list[str]) -> None:
    by_id: dict[str, LoadedRecord] = {}
    exact_seen: dict[str, LoadedRecord] = {}
    normalized_seen: dict[str, LoadedRecord] = {}
    prompt_splits: dict[str, list[LoadedRecord]] = defaultdict(list)

    for record in records:
        record_id = record.value.get("id")
        if isinstance(record_id, str):
            previous = by_id.get(record_id)
            if previous is not None:
                errors.append(
                    f"{record.location.label()}: duplicate id {record_id!r}; first seen at {previous.location.label()}"
                )
            else:
                by_id[record_id] = record

        previous_exact = exact_seen.get(record.exact_fingerprint)
        if previous_exact is not None:
            errors.append(
                f"{record.location.label()}: exact duplicate example content; first seen at {previous_exact.location.label()}"
            )
        else:
            exact_seen[record.exact_fingerprint] = record

        previous_normalized = normalized_seen.get(record.normalized_messages_fingerprint)
        if previous_normalized is not None:
            errors.append(
                f"{record.location.label()}: normalized-message duplicate; first seen at {previous_normalized.location.label()}"
            )
        else:
            normalized_seen[record.normalized_messages_fingerprint] = record

        prompt_splits[record.normalized_prompt_fingerprint].append(record)

    for same_prompt in prompt_splits.values():
        splits = {record.value.get("split") for record in same_prompt}
        if "held_out_eval" not in splits or not ({"train", "validation"} & splits):
            continue
        locations = ", ".join(
            f"{record.location.label()}[{record.value.get('split')}]" for record in same_prompt
        )
        errors.append(f"held-out prompt leakage across splits: {locations}")


def _print_summary(report: ValidationReport) -> None:
    split_counts = Counter(str(record.value.get("split")) for record in report.records)
    tier_counts = Counter(str(record.value.get("quality_tier")) for record in report.records)
    task_counts = Counter(str(record.value.get("task_type")) for record in report.records)

    print(
        f"Validated {len(report.records)} records from {report.files_read} files "
        f"({report.lines_read} physical lines)."
    )
    if split_counts:
        print("Splits: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    if tier_counts:
        print("Quality tiers: " + ", ".join(f"{key}={value}" for key, value in sorted(tier_counts.items())))
    if task_counts:
        print("Task types: " + ", ".join(f"{key}={value}" for key, value in sorted(task_counts.items())))

    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)

    print(f"Result: {'VALID' if report.valid else 'INVALID'} ({len(report.errors)} errors, {len(report.warnings)} warnings)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate LocalPilot JSONL training/evaluation datasets and held-out isolation."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories (directories are searched recursively for *.jsonl)",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="return a validation failure when warnings are present",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_files(args.paths)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    _print_summary(report)
    if not report.valid or (args.strict_warnings and report.warnings):
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
