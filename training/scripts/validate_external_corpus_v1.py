#!/usr/bin/env python3
"""Validate the frozen External Corpus v1 artifacts without optional packages.

This gate deliberately validates only checked-in artifacts and locks.  It never
opens the large raw source datasets, downloads data, or imports the corpus-build
dependency stack, so it is safe to run in the ordinary repository CI job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "training/datasets/external_corpus_v1.jsonl"
DEFAULT_SOURCES = ROOT / "training/sources/external_corpus_v1_sources.jsonl"
DEFAULT_MANIFEST = ROOT / "training/manifests/external_corpus_v1_manifest.json"
EXPECTED_DATASETS = {
    "nemotron_swe_v2",
    "open_code_instruct",
    "nemotron_agentic_v2",
    "xlam_60k",
    "codeact_instruct",
    "swe_care",
}
ALLOWED_SPLITS = {"train", "validation"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
SENSITIVE_CREDENTIAL_RE = re.compile(
    r"(?:"
    r"gh[pousr]_[A-Za-z0-9]{36,}"
    r"|github_pat_[A-Za-z0-9_]{40,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|xox[baprs]-[0-9A-Za-z-]{10,}"
    r"|[sr]k_live_[0-9A-Za-z]{16,}"
    r"|glpat-[0-9A-Za-z_-]{20,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|pypi-[A-Za-z0-9_-]{50,}"
    r"|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r")"
)


@dataclass
class ArtifactValidationReport:
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing {label}: {path}")
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def _read_jsonl(path: Path, label: str, errors: list[str]) -> list[dict[str, Any]] | None:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    errors.append(f"{label} line {line_number} is blank")
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{label} line {line_number} is invalid JSON: {exc.msg}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"{label} line {line_number} must be a JSON object")
                    continue
                records.append(value)
    except FileNotFoundError:
        errors.append(f"missing {label}: {path}")
        return None
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {label} {path}: {exc}")
        return None
    return records


def _manifest_int(value: Any, label: str, errors: list[str], *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"manifest {label} must be an integer >= {minimum}")
        return None
    return value


def _manifest_count_map(value: Any, label: str, errors: list[str]) -> dict[str, int] | None:
    if not isinstance(value, dict):
        errors.append(f"manifest {label} must be an object of nonnegative integer counts")
        return None
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(f"manifest {label} contains an invalid count")
            return None
        result[key] = count
    return result


def _check_digest(path: Path, expected: Any, label: str, errors: list[str]) -> None:
    if not isinstance(expected, str) or HEX_SHA256.fullmatch(expected) is None:
        errors.append(f"manifest {label} must be a lowercase SHA-256 digest")
        return
    try:
        actual = sha256_file(path)
    except OSError as exc:
        errors.append(f"cannot hash {label} file {path}: {exc}")
        return
    if actual != expected:
        errors.append(f"{label} digest mismatch: manifest={expected}, actual={actual}")


def _safe_manifest_path(repo_root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"manifest {label}.path must be nonempty text")
        return None
    relative = Path(value)
    if relative.is_absolute():
        errors.append(f"manifest {label}.path must be repository-relative")
        return None
    root = repo_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"manifest {label}.path escapes the repository root")
        return None
    return path


def _function_name(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    function = value.get("function", value)
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _validate_call(
    call: Any, *, record_id: str, location: str, tool_names: set[str], errors: list[str]
) -> tuple[str | None, str | None]:
    name = _function_name(call)
    if name is None:
        errors.append(f"{record_id}: {location} has a malformed native tool call")
        return None, None
    if name not in tool_names:
        errors.append(f"{record_id}: {location} calls native tool {name!r} absent from native_tools")
    function = call.get("function", call)
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            errors.append(f"{record_id}: {location} tool-call arguments are invalid JSON")
            arguments = None
    if not isinstance(arguments, dict):
        errors.append(f"{record_id}: {location} tool-call arguments must be an object")
    call_id = call.get("id")
    if call_id is not None and (not isinstance(call_id, str) or not call_id.strip()):
        errors.append(f"{record_id}: {location} native tool-call id must be nonempty text")
        call_id = None
    return name, call_id


def _validate_native_messages(
    messages: Any, *, record_id: str, location: str, tool_names: set[str],
    require_resolved_calls: bool, errors: list[str],
) -> tuple[int, int]:
    if not isinstance(messages, list) or not messages:
        errors.append(f"{record_id}: {location} must be a nonempty list")
        return 0, 0
    assistant_count = 0
    call_count = 0
    outstanding: tuple[str | None, str] | None = None
    seen_ids: set[str] = set()
    for index, message in enumerate(messages):
        item = f"{location}[{index}]"
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            errors.append(f"{record_id}: {item} is a malformed native message")
            continue
        role = message["role"]
        content = message.get("content", "")
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, str):
            errors.append(f"{record_id}: {item} content must be text")
        elif role != "assistant" and not content.strip():
            errors.append(f"{record_id}: {item} content must be nonempty text")
        calls = message.get("tool_calls")
        if role == "assistant":
            assistant_count += 1
            if outstanding is not None:
                errors.append(f"{record_id}: {item} appears before the outstanding native tool result")
            if calls is not None:
                if not isinstance(calls, list) or len(calls) != 1:
                    errors.append(f"{record_id}: {item} must contain exactly one native tool call")
                else:
                    name, call_id = _validate_call(
                        calls[0], record_id=record_id, location=item, tool_names=tool_names, errors=errors,
                    )
                    if name is not None:
                        call_count += 1
                        if call_id is not None:
                            if call_id in seen_ids:
                                errors.append(f"{record_id}: duplicate native tool-call id {call_id!r}")
                            seen_ids.add(call_id)
                        outstanding = (call_id, name)
            elif not isinstance(content, str) or not content.strip():
                errors.append(f"{record_id}: {item} is an empty assistant target")
        elif calls is not None:
            errors.append(f"{record_id}: {item} has tool_calls but is not an assistant message")
        if role == "tool":
            if outstanding is None:
                errors.append(f"{record_id}: {item} has no preceding native tool call")
            else:
                expected_id, expected_name = outstanding
                result_id = message.get("tool_call_id")
                result_name = message.get("name")
                if expected_id is not None and result_id != expected_id:
                    errors.append(f"{record_id}: {item} tool_call_id does not match its native call")
                if result_name is not None and result_name != expected_name:
                    errors.append(f"{record_id}: {item} tool name does not match its native call")
                outstanding = None
        elif outstanding is not None and role != "assistant":
            errors.append(f"{record_id}: {item} appears before the outstanding native tool result")
    if outstanding is not None and require_resolved_calls:
        errors.append(f"{record_id}: {location} ends with an unresolved native tool call")
    return assistant_count, call_count


def _validate_native_metadata(
    record: dict[str, Any], dataset: str, metadata: dict[str, Any], errors: list[str]
) -> int:
    record_id = str(record.get("id") or "<missing-id>")
    requires_trajectory = dataset in {"codeact_instruct", "nemotron_agentic_v2"} or (
        dataset == "nemotron_swe_v2" and record.get("task_type") == "software_engineering_agent"
    )
    requires_targets = dataset == "xlam_60k"
    has_native = any(key in metadata for key in ("native_tools", "native_messages", "native_call_targets"))
    if (requires_trajectory or requires_targets) and not has_native:
        errors.append(f"{record_id}: {dataset} tool record is missing native training metadata")

    tools = metadata.get("native_tools")
    tool_names: set[str] = set()
    if has_native or requires_trajectory or requires_targets:
        if not isinstance(tools, list) or not tools:
            errors.append(f"{record_id}: native_tools must be a nonempty list")
        else:
            for index, tool in enumerate(tools):
                name = _function_name(tool)
                if name is None:
                    errors.append(f"{record_id}: native_tools[{index}] has no function name")
                elif name in tool_names:
                    errors.append(f"{record_id}: duplicate native tool name {name!r}")
                else:
                    tool_names.add(name)

    native_messages = metadata.get("native_messages")
    assistant_count = 0
    message_calls = 0
    if has_native or requires_trajectory or requires_targets:
        assistant_count, message_calls = _validate_native_messages(
            native_messages,
            record_id=record_id,
            location="native_messages",
            tool_names=tool_names,
            require_resolved_calls="native_call_targets" not in metadata,
            errors=errors,
        )

    targets = metadata.get("native_call_targets")
    target_calls = 0
    if targets is not None or requires_targets:
        if not isinstance(targets, list) or not targets:
            errors.append(f"{record_id}: native_call_targets must be a nonempty list")
        else:
            for index, target in enumerate(targets):
                _, calls = _validate_native_messages(
                    [target],
                    record_id=record_id,
                    location=f"native_call_targets[{index}]",
                    tool_names=tool_names,
                    require_resolved_calls=False,
                    errors=errors,
                )
                target_calls += calls
                if calls != 1:
                    errors.append(f"{record_id}: native_call_targets[{index}] must contain one tool call")
    if requires_targets and message_calls:
        errors.append(f"{record_id}: xLAM native prompt must not duplicate calls from native_call_targets")
    if requires_trajectory and message_calls == 0:
        errors.append(f"{record_id}: native trajectory contains no tool calls")
    if requires_targets and target_calls == 0:
        errors.append(f"{record_id}: xLAM record contains no native call targets")

    if isinstance(targets, list) and targets:
        return len(targets)
    if isinstance(native_messages, list) and native_messages:
        return assistant_count
    messages = record.get("messages")
    if isinstance(messages, list):
        return sum(isinstance(message, dict) and message.get("role") == "assistant" for message in messages)
    return 0


def _contains_exact_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_exact_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_exact_key(item, forbidden) for item in value)
    return False


def _validate_lock_descriptor(
    manifest: dict[str, Any], *, key: str, repo_root: Path, errors: list[str]
) -> tuple[Path | None, dict[str, Any] | None]:
    descriptor = manifest.get(key)
    if not isinstance(descriptor, dict):
        errors.append(f"manifest {key} must be an object")
        return None, None
    path = _safe_manifest_path(repo_root, descriptor.get("path"), key, errors)
    if path is None:
        return None, None
    _check_digest(path, descriptor.get("sha256"), key, errors)
    if key != "acquisition_lock":
        return path, None
    return path, _read_json(path, key, errors)


def _validate_acquisition_lock(
    manifest: dict[str, Any], lock: dict[str, Any] | None, errors: list[str]
) -> None:
    if lock is None:
        return
    descriptor = manifest["acquisition_lock"]
    if descriptor.get("format_version") != lock.get("format_version"):
        errors.append("manifest acquisition_lock.format_version does not match the lock")
    datasets = lock.get("datasets")
    if not isinstance(datasets, dict):
        errors.append("acquisition lock datasets must be an object")
        return
    if set(datasets) != EXPECTED_DATASETS:
        errors.append("acquisition lock must contain exactly the six External Corpus v1 datasets")
    expected_inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, source in datasets.items():
        if not isinstance(source, dict) or not isinstance(source.get("files"), list):
            errors.append(f"acquisition lock dataset {dataset!r} has malformed files")
            continue
        for entry in source["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("remote_path"), str):
                errors.append(f"acquisition lock dataset {dataset!r} has a malformed file entry")
                continue
            expected_inventory[(dataset, entry["remote_path"])] = {
                "bytes": entry.get("size"),
                "revision": source.get("revision"),
                "repo_id": source.get("repo_id"),
                "path": (
                    f"{str(source.get('local_directory') or '').strip('/')}/"
                    f"{str(entry.get('local_path') or '').strip('/')}"
                ).strip("/"),
                "identity": "sha256" if "sha256" in entry else "git-blob-sha1",
                **({"sha256": entry["sha256"]} if "sha256" in entry else {}),
                **({"git_oid": entry["git_oid"]} if "git_oid" in entry else {}),
            }
    if descriptor.get("file_count") != len(expected_inventory):
        errors.append("manifest acquisition_lock.file_count does not match the lock")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        errors.append("manifest source_files must be a list derived from the acquisition lock")
        return
    actual_inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for item in source_files:
        if not isinstance(item, dict):
            errors.append("manifest source_files contains a malformed entry")
            continue
        key = (item.get("dataset"), item.get("remote_path"))
        if not all(isinstance(value, str) for value in key):
            errors.append("manifest source_files entry requires dataset and remote_path")
            continue
        if key in actual_inventory:
            errors.append(f"manifest source_files contains duplicate inventory key {key!r}")
        actual_inventory[key] = item
    if set(actual_inventory) != set(expected_inventory):
        errors.append("manifest source_files inventory does not exactly match the acquisition lock")
        return
    for key, expected in expected_inventory.items():
        actual = actual_inventory[key]
        for field_name, expected_value in expected.items():
            if actual.get(field_name) != expected_value:
                errors.append(f"manifest source_files {key!r} has incorrect {field_name}")


def _validate_manifest_rules(manifest: dict[str, Any], errors: list[str]) -> None:
    rules = manifest.get("rules")
    if not isinstance(rules, dict):
        errors.append("manifest rules must be an object")
        return
    for key in ("swe_care_test_split_used", "merged_patch_in_training_content"):
        if rules.get(key) is not False:
            errors.append(f"manifest rules.{key} must be false")


def _validate_records(
    records: list[dict[str, Any]], manifest: dict[str, Any], errors: list[str]
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, dict[str, Any]]]:
    datasets: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    training_examples: Counter[str] = Counter()
    by_id: dict[str, dict[str, Any]] = {}
    family_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for index, record in enumerate(records, 1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"corpus line {index} requires a nonempty id")
            record_id = f"<line-{index}>"
        elif record_id in by_id:
            errors.append(f"corpus contains duplicate id {record_id!r}")
        else:
            by_id[record_id] = record
        serialized_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if SENSITIVE_CREDENTIAL_RE.search(serialized_record):
            errors.append(f"{record_id}: contains a sensitive credential pattern")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{record_id}: provenance must be an object")
            provenance = {}
        dataset = provenance.get("dataset")
        if not isinstance(dataset, str) or dataset not in EXPECTED_DATASETS:
            errors.append(f"{record_id}: provenance.dataset is not an External Corpus v1 dataset")
            dataset = "<invalid>"
        datasets[dataset] += 1
        split = record.get("split")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{record_id}: split must be train or validation")
            split = "<invalid>"
        splits[str(split)] += 1
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{record_id}: metadata must be an object")
            metadata = {}
        family = metadata.get("family_id")
        if not isinstance(family, str) or not family:
            errors.append(f"{record_id}: metadata.family_id must be nonempty text")
        else:
            family_splits[("family_id", family)].add(str(split))
        problem = metadata.get("problem_identity_sha256")
        if problem is not None:
            if not isinstance(problem, str) or not problem:
                errors.append(f"{record_id}: metadata.problem_identity_sha256 must be nonempty text")
            else:
                family_splits[("problem_identity_sha256", problem)].add(str(split))

        expected_examples = _validate_native_metadata(record, dataset, metadata, errors)
        declared_examples = metadata.get("training_example_count")
        if isinstance(declared_examples, bool) or not isinstance(declared_examples, int) or declared_examples < 1:
            errors.append(f"{record_id}: metadata.training_example_count must be an integer >= 1")
        else:
            training_examples[str(split)] += declared_examples
            if expected_examples != declared_examples:
                errors.append(
                    f"{record_id}: training_example_count={declared_examples} but native expansion yields {expected_examples}"
                )

        if dataset == "swe_care":
            source_path = provenance.get("source_path")
            if not isinstance(source_path, str) or not source_path.startswith("data/dev-") or not source_path.endswith(".parquet"):
                errors.append(f"{record_id}: SWE-CARE source_path must be the dev Parquet split")
            if _contains_exact_key(record, "merged_patch"):
                errors.append(f"{record_id}: SWE-CARE record contains forbidden merged_patch data")
            if metadata.get("merged_patch_used_as_verifier_only") is not True:
                errors.append(f"{record_id}: SWE-CARE must record merged_patch as verifier-only evidence")

    for (kind, family), family_split_set in family_splits.items():
        if len(family_split_set) > 1:
            errors.append(f"{kind} {family!r} crosses train/validation splits")
    return datasets, splits, training_examples, by_id


def _validate_sources(
    sources: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]], errors: list[str]
) -> None:
    source_by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, 1):
        record_id = source.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"source ledger line {index} requires a nonempty id")
            continue
        if record_id in source_by_id:
            errors.append(f"source ledger contains duplicate id {record_id!r}")
            continue
        source_by_id[record_id] = source
        raw_digest = source.get("raw_record_sha256")
        if not isinstance(raw_digest, str) or HEX_SHA256.fullmatch(raw_digest) is None:
            errors.append(f"{record_id}: source ledger raw_record_sha256 is invalid")

    corpus_ids, source_ids = set(records_by_id), set(source_by_id)
    if corpus_ids != source_ids:
        missing = sorted(corpus_ids - source_ids)
        extra = sorted(source_ids - corpus_ids)
        errors.append(
            "source ledger IDs are not one-to-one with corpus IDs"
            f" (missing={missing[:5]}, extra={extra[:5]})"
        )
    for record_id in sorted(corpus_ids & source_ids):
        record = records_by_id[record_id]
        source = source_by_id[record_id]
        provenance = record["provenance"]
        metadata = record["metadata"]
        expected_fields = {
            "dataset": provenance.get("dataset"),
            "pinned_revision": provenance.get("revision"),
            "original_id": provenance.get("original_id"),
            "license": record.get("license"),
            "source_path": provenance.get("source_path"),
            "transformation_version": provenance.get("transformation_version"),
            "acquisition_date": provenance.get("acquisition_date"),
            "family_id": metadata.get("family_id"),
            "reference": provenance.get("reference"),
            "license_reference": provenance.get("license_reference"),
        }
        for field_name, expected in expected_fields.items():
            if source.get(field_name) != expected:
                errors.append(f"{record_id}: source ledger {field_name} disagrees with corpus provenance")
        if provenance.get("dataset") == "swe_care":
            source_path = source.get("source_path")
            if not isinstance(source_path, str) or not source_path.startswith("data/dev-"):
                errors.append(f"{record_id}: source ledger references a non-dev SWE-CARE split")


def validate_artifacts(
    *, corpus_path: Path = DEFAULT_CORPUS, sources_path: Path = DEFAULT_SOURCES,
    manifest_path: Path = DEFAULT_MANIFEST, repo_root: Path = ROOT,
) -> ArtifactValidationReport:
    report = ArtifactValidationReport()
    manifest = _read_json(manifest_path, "external corpus manifest", report.errors)
    records = _read_jsonl(corpus_path, "external corpus", report.errors)
    sources = _read_jsonl(sources_path, "external source ledger", report.errors)
    if manifest is None or records is None or sources is None:
        return report

    _check_digest(corpus_path, manifest.get("corpus_sha256"), "corpus_sha256", report.errors)
    _check_digest(sources_path, manifest.get("sources_sha256"), "sources_sha256", report.errors)
    _, lock = _validate_lock_descriptor(
        manifest, key="acquisition_lock", repo_root=repo_root, errors=report.errors,
    )
    _validate_lock_descriptor(
        manifest, key="runtime_requirements", repo_root=repo_root, errors=report.errors,
    )
    _validate_acquisition_lock(manifest, lock, report.errors)
    _validate_manifest_rules(manifest, report.errors)
    eval_manifest = _read_json(
        repo_root / "training/manifests/eval_v1_manifest.json",
        "Eval v1 manifest",
        report.errors,
    )
    if eval_manifest is not None:
        eval_digest = eval_manifest.get("records_sha256")
        if not isinstance(eval_digest, str) or HEX_SHA256.fullmatch(eval_digest) is None:
            report.errors.append("Eval v1 manifest records_sha256 is invalid")
        elif manifest.get("eval_manifest_records_sha256") != eval_digest:
            report.errors.append("external corpus manifest references a different Eval v1 manifest")
    runtime = manifest.get("build_runtime")
    if not isinstance(runtime, dict) or runtime.get("exact_package_versions_verified") is not True:
        report.errors.append("manifest build_runtime must confirm exact package versions")

    dataset_specs = manifest.get("datasets")
    if not isinstance(dataset_specs, dict) or set(dataset_specs) != EXPECTED_DATASETS:
        report.errors.append("manifest datasets must contain exactly the six External Corpus v1 datasets")
        dataset_specs = {}

    dataset_counts, split_counts, training_counts, records_by_id = _validate_records(
        records, manifest, report.errors,
    )
    _validate_sources(sources, records_by_id, report.errors)

    declared_records = _manifest_int(manifest.get("records"), "records", report.errors)
    if declared_records is not None and declared_records != len(records):
        report.errors.append(f"manifest records={declared_records}, actual={len(records)}")
    maximum_records = _manifest_int(manifest.get("maximum_records"), "maximum_records", report.errors, minimum=1)
    if maximum_records is not None and len(records) > maximum_records:
        report.errors.append(f"corpus has {len(records)} records, above maximum_records={maximum_records}")

    actual_accepted = {dataset: dataset_counts.get(dataset, 0) for dataset in sorted(EXPECTED_DATASETS)}
    declared_accepted = _manifest_count_map(manifest.get("accepted_counts"), "accepted_counts", report.errors)
    if declared_accepted is not None and declared_accepted != actual_accepted:
        report.errors.append(f"manifest accepted_counts={declared_accepted}, actual={actual_accepted}")
    for dataset, spec in dataset_specs.items():
        ceiling = spec.get("ceiling") if isinstance(spec, dict) else None
        ceiling = _manifest_int(ceiling, f"datasets.{dataset}.ceiling", report.errors)
        if ceiling is not None and dataset_counts.get(dataset, 0) > ceiling:
            report.errors.append(
                f"dataset {dataset} has {dataset_counts[dataset]} records, above ceiling={ceiling}"
            )

    actual_splits = dict(sorted((key, count) for key, count in split_counts.items() if key in ALLOWED_SPLITS))
    declared_splits = _manifest_count_map(manifest.get("split_counts"), "split_counts", report.errors)
    if declared_splits is not None and declared_splits != actual_splits:
        report.errors.append(f"manifest split_counts={declared_splits}, actual={actual_splits}")
    if records and set(actual_splits) != ALLOWED_SPLITS:
        report.errors.append("external corpus must contain both train and validation splits")

    actual_training = {split: training_counts.get(split, 0) for split in sorted(ALLOWED_SPLITS)}
    actual_training["total"] = sum(actual_training.values())
    declared_training = _manifest_count_map(
        manifest.get("training_example_counts"), "training_example_counts", report.errors,
    )
    if declared_training is not None and declared_training != actual_training:
        report.errors.append(
            f"manifest training_example_counts={declared_training}, actual={actual_training}"
        )
    declared_training_total = _manifest_int(
        manifest.get("training_examples"), "training_examples", report.errors,
    )
    if declared_training_total is not None and declared_training_total != actual_training["total"]:
        report.errors.append(
            f"manifest training_examples={declared_training_total}, actual={actual_training['total']}"
        )

    report.summary = {
        "records": len(records),
        "accepted_counts": actual_accepted,
        "split_counts": actual_splits,
        "training_example_counts": actual_training,
        "corpus_sha256": sha256_file(corpus_path),
        "sources_sha256": sha256_file(sources_path),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_artifacts(
        corpus_path=args.corpus,
        sources_path=args.sources,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
    )
    payload = {"valid": report.valid, **report.summary}
    if report.errors:
        payload["errors"] = report.errors
    print(json.dumps(payload, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
