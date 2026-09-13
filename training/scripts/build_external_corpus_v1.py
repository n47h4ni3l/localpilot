#!/usr/bin/env python3
"""Build the revision-pinned, leakage-checked External Corpus v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_corpus_v1 import (
    ACQUISITION_DATE,
    CODEACT_CONSTITUENTS,
    DATASETS,
    MAX_SEQUENCE_LENGTH,
    TRANSFORMATION_VERSION,
    Candidate,
    CandidatePool,
    Deduplicator,
    canonical_json,
    canonical_tool_contracts,
    canonical_tool_signature,
    compact_tools,
    candidate_record,
    classify_agentless,
    codeact_messages,
    global_content_rejection,
    grouped_split,
    jsonl_rows,
    messages_without_hidden_reasoning,
    native_tool_call,
    native_tools,
    normalize_text,
    opencode_judgement,
    openhands_trajectory,
    parse_json,
    raw_record_digest,
    score_components,
    sha256_file,
    sha256_text,
    simple_difficulty,
    source_record,
    swecare_messages,
    tool_trajectory,
    validate_calls,
    verify_agentless,
    xlam_trivial_one_call,
)
from acquire_external_corpus_v1 import ACQUISITION_LOCK, LOCK_PATH, verify_dataset
from external_corpus_runtime import build_runtime
from training_common import find_manifest_leaks, load_eval_manifest
from train_adapter import expand_training_examples, validate_tokenized_examples


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(r"E:\LocalPilot-Training-Data")
DEFAULT_CORPUS = ROOT / "training/datasets/external_corpus_v1.jsonl"
DEFAULT_SOURCES = ROOT / "training/sources/external_corpus_v1_sources.jsonl"
DEFAULT_MANIFEST = ROOT / "training/manifests/external_corpus_v1_manifest.json"
DEFAULT_REPORT = ROOT / "training/reports/external_corpus_v1_stats.json"
EVAL_MANIFEST = ROOT / "training/manifests/eval_v1_manifest.json"
TOKENIZER_ID = "openai/gpt-oss-20b"
TOKENIZER_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
TOKENIZER_CACHE = DEFAULT_SOURCE_ROOT / ".tokenizers/gpt-oss-20b"
RUNTIME_REQUIREMENTS = ROOT / "training/requirements-external-corpus-v1.txt"


class Stats:
    def __init__(self) -> None:
        self.considered = Counter()
        self.rejections: dict[str, Counter[str]] = defaultdict(Counter)
        self.rejected_ids: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.duplicate_clusters: dict[str, set[str]] = defaultdict(set)

    def reject(self, dataset: str, reason: str, identifier: Any, *, cluster: str | None = None) -> None:
        self.rejections[dataset][reason] += 1
        if cluster:
            self.duplicate_clusters[reason].add(cluster)
        values = self.rejected_ids[dataset][reason]
        if len(values) < 100:
            values.append(str(identifier))

    def merge_overflow(self, dataset: str, pool: CandidatePool) -> None:
        for reason, count in pool.overflow.items():
            self.rejections[dataset][reason] += count


def _stable_diversity(identifier: str) -> float:
    return 0.5 + int(sha256_text(identifier)[:8], 16) / 0xFFFFFFFF * 0.5


def _family_from_prompt(prompt: str) -> str:
    # Hash the complete normalized task. Truncating a boilerplate-heavy prompt
    # (or erasing all numeric distinctions) can collapse unrelated problems;
    # paraphrases are handled separately by global shingle deduplication.
    return sha256_text(normalize_text(prompt))


def _issue_identity(prompt: str) -> str:
    for pattern in (
        r"\[issue\](.*?)(?:\[/issue\]|###\s*(?:repository|repo)|\Z)",
        r"<issue_description>(.*?)(?:</issue_description>|\Z)",
        r"---\s*begin issue\s*---(.*?)---\s*end issue\s*---",
        r"###\s*github problem description\s*###(.*?)(?:###\s*(?:repository|repo)|\Z)",
    ):
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            return _family_from_prompt(match.group(1))
    return _family_from_prompt(prompt)


def _difficulty_score(text: str, base: float = 0.35) -> float:
    value = normalize_text(text)
    signals = (
        "debug", "incomplete", "boundary", "edge case", "state", "parser", "repository",
        "concurr", "security", "performance", "regression", "multi", "api", "data structure",
    )
    return min(1.0, base + 0.07 * sum(signal in value for signal in signals))


def _upstream_materially_overlength(row: dict[str, Any]) -> bool:
    """Reject obviously huge traces before retaining them; survivors still use the pinned tokenizer."""
    metadata = row.get("metadata")
    counts = metadata.get("all_turns_token_count") if isinstance(metadata, dict) else None
    try:
        return isinstance(counts, dict) and int(counts.get("all") or 0) > MAX_SEQUENCE_LENGTH * 4
    except (TypeError, ValueError):
        return False


def _generic_candidate(
    *, dataset: str, original_id: Any, source_path: str, messages: list[dict[str, str]],
    task_type: str, difficulty_score: float, family: str, category: str, subskill: str,
    verification: float, relevance: float, metadata: dict[str, Any], repository: str | None = None,
    tool_signature: str | None = None, raw_sha256: str | None = None, license_override: str | None = None,
) -> Candidate:
    identifier = str(original_id)
    return Candidate(
        dataset=dataset,
        original_id=identifier,
        source_path=source_path,
        messages=messages,
        task_type=task_type,
        difficulty=simple_difficulty(difficulty_score),
        family=family,
        category=re.sub(r"[^a-z0-9_:-]+", "_", normalize_text(category)).strip("_")[:60] or "general",
        subskill=subskill,
        scores=score_components(
            verification=verification,
            difficulty=difficulty_score,
            relevance=relevance,
            diversity=_stable_diversity(f"{dataset}:{identifier}:{category}"),
        ),
        metadata=metadata,
        repository=repository,
        tool_signature=tool_signature,
        raw_sha256=raw_sha256,
        license_override=license_override,
    )


def load_xlam(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "xlam_60k"
    path = root / "xlam-function-calling-60k/xlam_function_calling_60k.json"
    if sha256_file(path) != DATASETS[dataset]["expected_sha256"]:
        raise RuntimeError("xLAM source hash does not match the verified file")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("xLAM source must be a JSON array")
    limits = {f"{calls}:{generator}": size for calls, size in (("one", 1200), ("two", 2600), ("three_plus", 1600)) for generator in ("deepseek", "mixtral")}
    pool = CandidatePool(limits)
    for row_index, row in enumerate(rows):
        stats.considered[dataset] += 1
        identifier = row.get("id", row_index) if isinstance(row, dict) else row_index
        try:
            tools, calls = parse_json(row["tools"]), parse_json(row["answers"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            stats.reject(dataset, "malformed_tool_json", identifier)
            continue
        reason = validate_calls(tools, calls)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            stats.reject(dataset, "malformed_messages", identifier)
            continue
        call_count = len(calls)
        bucket_calls = "one" if call_count == 1 else "two" if call_count == 2 else "three_plus"
        if xlam_trivial_one_call(call_count, len(tools)):
            stats.reject(dataset, "trivial_one_call_without_distractors", identifier)
            continue
        generator = "deepseek" if row_index < 33659 else "mixtral"
        tool_signature = canonical_tool_signature(tools)
        contract_signatures = canonical_tool_contracts(tools)
        call_functions = [call.get("function", call) if isinstance(call, dict) else {} for call in calls]
        called_names = [str(call.get("name") or "") for call in call_functions]
        categories = sorted({
            re.sub(r"[^a-z0-9_:-]+", "_", normalize_text(name)).strip("_") or "unknown"
            for name in called_names
        })
        category = categories[0] if categories else "unknown"
        called_signatures = sorted({contract_signatures[name] for name in called_names if name in contract_signatures})
        if len(called_signatures) != len(set(called_names)):
            stats.reject(dataset, "unresolved_called_tool_contract", identifier)
            continue
        parsed_arguments = []
        for call in call_functions:
            try:
                parsed_arguments.append(parse_json(call.get("arguments", {})))
            except (TypeError, ValueError):
                parsed_arguments.append({})
        nested = sum(isinstance(value, (list, dict)) for arguments in parsed_arguments for value in arguments.values())
        required = 0
        for tool in tools:
            parameters = tool.get("parameters", {}) if isinstance(tool, dict) else {}
            required += len(parameters.get("required", [])) if isinstance(parameters, dict) else 0
        difficulty = min(1.0, 0.20 + 0.13 * call_count + 0.035 * min(len(tools), 12) + 0.08 * min(nested, 3) + 0.02 * min(required, 5))
        messages = [
            {"role": "system", "content": "Available tools:\n" + canonical_json(compact_tools(tools))},
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": canonical_json(calls)},
        ]
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        answer_key = sha256_text(normalize_text(query) + "\n" + tool_signature + "\n" + canonical_json(calls))
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path="xlam_function_calling_60k.json",
            messages=messages, task_type="tool_use", difficulty_score=difficulty,
            family=sha256_text(normalize_text(query) + "\n" + tool_signature), category=category,
            subskill=f"{bucket_calls}_call", verification=1.0, relevance=0.90 if call_count >= 2 else 0.68,
            tool_signature=tool_signature, raw_sha256=raw_record_digest(row),
            metadata={
                "generator_partition": generator, "call_count": call_count, "available_tool_count": len(tools),
                "api_category": category, "api_categories": categories,
                "called_tool_signatures": called_signatures,
                "native_tools": native_tools(tools),
                # The schema-safe corpus messages retain a textual tool list for
                # PR #90 validation.  Native training must receive the contracts
                # exactly once through the tokenizer's `tools` argument.
                "native_messages": [{"role": "user", "content": query.strip()}],
                "native_call_targets": [
                    {"role": "assistant", "content": "", "tool_calls": [native_tool_call(call, index)]}
                    for index, call in enumerate(calls)
                ],
                "problem_identity_sha256": _family_from_prompt(query),
                "dataset_duplicate_key_sha256": answer_key,
            },
        )
        pool.add(f"{bucket_calls}:{generator}", candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


def load_codeact(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "codeact_instruct"
    path = root / "CodeActInstruct/full_std.jsonl"
    if not path.exists():
        path = root / "CodeActInstruct/codeactinstruct/full_std.jsonl"
    pool = CandidatePool({"multi": 6000, "simple": 2400})
    for line_number, row, raw_digest in jsonl_rows(path):
        stats.considered[dataset] += 1
        identifier = row.get("id", line_number)
        parts = str(identifier).split("/")
        constituent = "/".join(parts[1:3]) if len(parts) >= 3 else ""
        constituent_provenance = CODEACT_CONSTITUENTS.get(constituent)
        if not constituent_provenance:
            stats.reject(dataset, "unresolved_constituent_license", identifier)
            continue
        license_name = constituent_provenance["license"]
        messages, traits, reason = codeact_messages(row)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        cycles = traits["action_observation_cycles"]
        if cycles < 1:
            stats.reject(dataset, "pure_explanatory_chat", identifier)
            continue
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        initial = next(item["content"] for item in messages if item["role"] == "user")
        action_sequence = [item["content"] for item in messages if item["role"] == "assistant" and item["content"].startswith("<execute>")]
        base_task = "/".join(parts[1:4]) if len(parts) >= 4 else str(identifier)
        family = sha256_text(base_task + "\n" + normalize_text(initial))
        duplicate_key = sha256_text(family + "\n" + canonical_json(action_sequence))
        correction = traits["error_recovered"]
        difficulty = min(1.0, 0.30 + 0.11 * cycles + (0.18 if correction else 0.0))
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path="codeactinstruct/full_std.jsonl",
            messages=messages, task_type="agentic_code_execution", difficulty_score=difficulty,
            family=family, category=constituent.replace("/", "_"), subskill="execute_observe_revise",
            verification=0.95 if traits["successful_result"] else 0.70, relevance=0.92 if correction else 0.78,
            raw_sha256=raw_digest, license_override=f"Apache-2.0 wrapper; constituent: {license_name}",
            metadata={
                **traits,
                "constituent": constituent,
                "constituent_license": license_name,
                "upstream_source_family": constituent_provenance["upstream_source_family"],
                "constituent_provenance": {"name": constituent, **constituent_provenance},
                "problem_identity_sha256": _family_from_prompt(initial),
                "dataset_duplicate_key_sha256": duplicate_key,
            },
        )
        pool.add("multi" if cycles >= 2 else "simple", candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


def _parquet_rows(paths: Iterable[Path], columns: list[str] | None = None) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the corpus extra: pip install -e .[corpus]") from exc
    for path in sorted(paths):
        file = pq.ParquetFile(path)
        row_index = 0
        for batch in file.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                yield path, row_index, row
                row_index += 1


def load_swecare(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "swe_care"
    # The test split may exist in the owner's raw directory, but this glob is intentionally dev-only.
    paths = list((root / "SWE-CARE").glob("dev-*.parquet")) + list((root / "SWE-CARE/data").glob("dev-*.parquet"))
    if not paths:
        raise RuntimeError("SWE-CARE dev split is missing")
    pool = CandidatePool({"hard": 5000, "medium": 3600, "easy": 1800})
    for path, row_index, row in _parquet_rows(paths):
        stats.considered[dataset] += 1
        identifier = row.get("instance_id") or f"{path.name}:{row_index}"
        pr_key = f"{row.get('repo')}#{row.get('pull_number')}"
        messages, traits, reason = swecare_messages(row)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        meta = row.get("metadata") or {}
        difficulty_name = normalize_text(meta.get("difficulty") or "medium")
        difficulty_name = {"low": "easy", "high": "hard"}.get(difficulty_name, difficulty_name)
        if difficulty_name not in {"easy", "medium", "hard"}:
            difficulty_name = "medium"
        difficulty = {"easy": 0.32, "medium": 0.67, "hard": 0.92}[difficulty_name]
        effort = int(meta.get("estimated_review_effort") or 0)
        domain = str(meta.get("problem_domain") or "general")
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path=f"data/{path.name}", messages=messages,
            task_type="code_review", difficulty_score=difficulty,
            family=sha256_text(pr_key), category=domain, subskill="actionable_code_review",
            verification=1.0, relevance=0.95 if effort >= 4 else 0.78,
            repository=str(row.get("repo") or "unknown"), raw_sha256=raw_record_digest(row),
            metadata={**traits, "pull_number": row.get("pull_number"), "source_difficulty": difficulty_name, "estimated_review_effort": effort, "problem_domain": domain, "problem_identity_sha256": sha256_text(pr_key), "dataset_duplicate_key_sha256": sha256_text(pr_key)},
        )
        pool.add(difficulty_name, candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


def load_opencode(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "open_code_instruct"
    paths = list((root / "OpenCodeInstruct/data").glob("train-*-of-00050.parquet"))
    names = {path.name for path in paths}
    expected = {f"train-{index:05d}-of-00050.parquet" for index in range(50)}
    if names != expected:
        missing = sorted(expected - names)
        raise RuntimeError(f"OpenCodeInstruct requires all 50 pinned shards; missing {missing[:5]}")
    limits = {
        "generic:evol-instruct": 12600,
        "generic:self-instruct": 8400,
        "algorithmic:evol-instruct": 5400,
        "algorithmic:self-instruct": 3600,
    }
    pool = CandidatePool(limits)
    columns = ["id", "input", "output", "domain", "generation_algorithm", "llm_judgement", "unit_tests", "tests_execution_status", "average_test_score"]
    for path, row_index, row in _parquet_rows(paths, columns):
        stats.considered[dataset] += 1
        identifier = row.get("id") or f"{path.name}:{row_index}"
        if row.get("average_test_score") != 1.0:
            stats.reject(dataset, "test_score_not_perfect", identifier)
            continue
        try:
            tests = parse_json(row.get("unit_tests"))
            statuses = parse_json(row.get("tests_execution_status"))
        except (TypeError, ValueError):
            stats.reject(dataset, "malformed_tests", identifier)
            continue
        serialized_tests = [canonical_json(test) for test in tests] if isinstance(tests, list) else []
        if (
            not isinstance(tests, list) or not isinstance(statuses, list) or len(tests) < 3
            or len(set(serialized_tests)) < 3 or any(len(normalize_text(test).split()) < 3 for test in serialized_tests)
            or len(tests) != len(statuses) or any(status != "pass" for status in statuses)
        ):
            stats.reject(dataset, "insufficient_verified_tests", identifier)
            continue
        judgement, reason = opencode_judgement(row.get("llm_judgement"), len(tests))
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        input_text, output_text = row.get("input"), row.get("output")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            stats.reject(dataset, "malformed_messages", identifier)
            continue
        messages = [{"role": "user", "content": input_text.strip()}, {"role": "assistant", "content": output_text.strip()}]
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        domain = str(row.get("domain") or "generic")
        generation = str(row.get("generation_algorithm") or "self-instruct")
        bucket = f"{domain}:{generation}"
        if bucket not in limits:
            stats.reject(dataset, "unsupported_generation_stratum", identifier)
            continue
        test_key = sha256_text(canonical_json(tests))
        family = _family_from_prompt(input_text)
        duplicate_key = sha256_text(family + "\n" + test_key)
        difficulty = min(1.0, _difficulty_score(input_text, 0.28) + min(len(tests), 10) * 0.035)
        realistic = any(term in normalize_text(input_text) for term in ("debug", "api", "class", "parse", "state", "implement", "incomplete"))
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path=f"data/{path.name}", messages=messages,
            task_type="code_generation", difficulty_score=difficulty, family=family,
            category=domain, subskill=generation, verification=judgement,
            relevance=0.92 if realistic else 0.72, raw_sha256=raw_record_digest(row),
            metadata={"domain": domain, "generation_algorithm": generation, "verified_test_count": len(tests), "average_test_score": 1.0, "problem_identity_sha256": family, "dataset_duplicate_key_sha256": duplicate_key},
        )
        pool.add(bucket, candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


def _loop_quality(row: dict[str, Any]) -> tuple[float, float]:
    processing = row.get("processing_info")
    if not isinstance(processing, dict):
        return 1.0, 0.3
    loop = processing.get("loop_detection")
    if not isinstance(loop, dict):
        return 1.0, 0.3
    try:
        # Despite its upstream name, suffix_repetition_ratio is a retained-token
        # diversity/pass score: higher is better and `threshold` is its floor.
        return float(loop.get("repetition_ratio") or 0.0), float(loop.get("threshold") or 0.3)
    except (TypeError, ValueError):
        return 0.0, 0.3


def load_nemotron_swe(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "nemotron_swe_v2"
    data_root = root / "Nemotron-SFT-SWE-v2/data"
    swe_path, agentless_path = data_root / "swe.jsonl", data_root / "agentless.jsonl"
    if not swe_path.exists() or not agentless_path.exists():
        raise RuntimeError("Nemotron-SFT-SWE-v2 requires data/swe.jsonl and data/agentless.jsonl")
    pool = CandidatePool({"openhands_swe": 18000, "repair": 10000, "test_generation": 9000, "localization": 7000})
    for line_number, row, raw_digest in jsonl_rows(swe_path):
        stats.considered[dataset] += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        identifier = metadata.get("uuid") or f"swe:{line_number}"
        if row.get("filter_reason") not in {None, "", "None"}:
            stats.reject(dataset, "upstream_filter_reason", identifier)
            continue
        if _upstream_materially_overlength(row):
            stats.reject(dataset, "upstream_materially_overlength", identifier)
            continue
        loop_quality, loop_threshold = _loop_quality(row)
        if loop_quality < loop_threshold:
            stats.reject(dataset, "loop_repetition_score", identifier)
            continue
        messages, traits, sequence_reason = openhands_trajectory(row)
        if sequence_reason:
            stats.reject(dataset, f"invalid_openhands_{sequence_reason}", identifier)
            continue
        if traits["action_count"] < 3:
            stats.reject(dataset, "insufficient_repo_actions", identifier)
            continue
        if traits["repeated_identical_action"]:
            stats.reject(dataset, "repeated_identical_action", identifier)
            continue
        if not traits["inspect_before_edit"]:
            stats.reject(dataset, "edit_before_inspection", identifier)
            continue
        if not traits["validation_after_edit"]:
            stats.reject(dataset, "no_validation", identifier)
            continue
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        final = next((item["content"] for item in reversed(messages) if item["role"] == "assistant"), "")
        if re.search(r"\b(?:unable|cannot|failed to|not completed|still fail|tests? (?:are )?failing)\b", final, re.IGNORECASE):
            stats.reject(dataset, "terminal_failure", identifier)
            continue
        first_user = next(item["content"] for item in messages if item["role"] == "user")
        # UUID/alt_id identify generated trajectories, not necessarily the
        # underlying repository issue.  Normalize the issue text so agentic
        # and Agentless variants are grouped together for dedup and splitting.
        family_source = _issue_identity(first_user)
        difficulty = min(1.0, 0.55 + 0.025 * min(traits["action_count"], 10) + (0.15 if traits["failed_changed_passed"] else 0.0))
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path="data/swe.jsonl", messages=messages,
            task_type="software_engineering_agent", difficulty_score=difficulty,
            family=family_source, category="openhands_swe", subskill="agentic_swe",
            verification=1.0 if traits["failed_changed_passed"] else 0.90, relevance=1.0,
            raw_sha256=raw_digest, metadata={
                **traits,
                "upstream_suffix_nonrepetition_ratio": loop_quality,
                "upstream_alt_id": metadata.get("alt_id"),
                "upstream_uuid": metadata.get("uuid"),
                "problem_identity_sha256": family_source,
            },
        )
        pool.add("openhands_swe", candidate)
    for line_number, row, raw_digest in jsonl_rows(agentless_path):
        stats.considered[dataset] += 1
        identifier = row.get("uuid") or f"agentless:{line_number}"
        messages = messages_without_hidden_reasoning(row.get("messages"))
        subskill = classify_agentless(messages)
        if not subskill:
            stats.reject(dataset, "unknown_agentless_subskill", identifier)
            continue
        if not verify_agentless(subskill, messages):
            stats.reject(dataset, "unverified_agentless_artifact", identifier)
            continue
        reason = global_content_rejection(messages)
        if reason:
            stats.reject(dataset, reason, identifier)
            continue
        prompt = next(item["content"] for item in messages if item["role"] == "user")
        difficulty = _difficulty_score(prompt, {"repair": 0.62, "test_generation": 0.55, "localization": 0.42}[subskill])
        candidate = _generic_candidate(
            dataset=dataset, original_id=identifier, source_path="data/agentless.jsonl", messages=messages,
            task_type=f"software_engineering_{subskill}", difficulty_score=difficulty,
            family=_issue_identity(prompt), category=subskill, subskill=subskill,
            verification=0.92, relevance=0.96 if subskill != "localization" else 0.82,
            raw_sha256=raw_digest, metadata={"agentless_subskill": subskill, "used_in": row.get("used_in", []), "problem_identity_sha256": _issue_identity(prompt)},
        )
        pool.add(subskill, candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


def load_nemotron_agentic(root: Path, stats: Stats) -> tuple[dict[str, list[Candidate]], CandidatePool]:
    dataset = "nemotron_agentic_v2"
    data_root = root / "Nemotron-SFT-Agentic-v2/data"
    paths = [("tool_calling", data_root / "tool_calling.jsonl"), ("search", data_root / "search.jsonl")]
    if any(not path.exists() for _, path in paths):
        raise RuntimeError("Nemotron-SFT-Agentic-v2 tool_calling/search sources are missing")
    pool = CandidatePool({"tool_simple": 3600, "tool_multi": 16000, "search": 1200})
    for subset, path in paths:
        for line_number, row, raw_digest in jsonl_rows(path):
            stats.considered[dataset] += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            identifier = metadata.get("uuid") or row.get("uuid") or f"{subset}:{line_number}"
            if row.get("filter_reason") not in {None, "", "None"}:
                stats.reject(dataset, "upstream_filter_reason", identifier)
                continue
            if _upstream_materially_overlength(row):
                stats.reject(dataset, "upstream_materially_overlength", identifier)
                continue
            source_family = str(metadata.get("source") or "unknown")
            if subset == "tool_calling" and (source_family == "api_bank" or source_family.startswith("custom_")):
                stats.reject(dataset, "customer_service_or_interactive_material", identifier)
                continue
            messages, traits, reason = tool_trajectory(row)
            if reason:
                stats.reject(dataset, reason, identifier)
                continue
            calls = traits["tool_call_count"]
            if calls < 1 or calls > 6:
                stats.reject(dataset, "tool_call_count_out_of_range", identifier)
                continue
            if calls >= 2 and traits["distinct_tool_count"] < 2:
                stats.reject(dataset, "insufficient_distinct_tools", identifier)
                continue
            reason = global_content_rejection(messages)
            if reason:
                stats.reject(dataset, reason, identifier)
                continue
            user_goal = next(item["content"] for item in messages if item["role"] == "user")
            tools = row["tools"]
            tool_signature = canonical_tool_signature(tools)
            family = sha256_text(normalize_text(user_goal) + "\n" + tool_signature + "\n" + canonical_json(traits["ordered_tool_names"]))
            difficulty = min(1.0, 0.20 + 0.11 * calls + 0.08 * traits["distinct_tool_count"] + (0.18 if traits["error_recovered"] else 0.0))
            candidate = _generic_candidate(
                dataset=dataset, original_id=identifier, source_path=f"data/{path.name}", messages=messages,
                task_type="search_tool_use" if subset == "search" else "multi_tool_use",
                difficulty_score=difficulty, family=family, category=source_family,
                subskill="search" if subset == "search" else ("one_tool" if calls == 1 else "dependent_tool_chain"),
                verification=0.98 if traits["error_recovered"] else 0.92,
                relevance=0.95 if subset == "tool_calling" and calls >= 2 else 0.68,
                tool_signature=tool_signature, raw_sha256=raw_digest,
                metadata={**traits, "upstream_source_family": source_family, "subset": subset, "problem_identity_sha256": _family_from_prompt(user_goal), "dataset_duplicate_key_sha256": family},
            )
            pool.add("search" if subset == "search" else ("tool_simple" if calls == 1 else "tool_multi"), candidate)
    pools = pool.finish()
    stats.merge_overflow(dataset, pool)
    return pools, pool


QUOTAS: dict[str, dict[str, int]] = {
    "nemotron_swe_v2": {"openhands_swe": 2500, "repair": 1400, "test_generation": 1200, "localization": 900},
    "open_code_instruct": {"generic:evol-instruct": 2520, "generic:self-instruct": 1680, "algorithmic:evol-instruct": 1080, "algorithmic:self-instruct": 720},
    "nemotron_agentic_v2": {"tool_multi": 2400, "tool_simple": 423, "search": 150},
    "xlam_60k": {
        "one:deepseek": 200, "one:mixtral": 200,
        "two:deepseek": 550, "two:mixtral": 550,
        "three_plus:deepseek": 250, "three_plus:mixtral": 250,
    },
    "codeact_instruct": {"multi": 1050, "simple": 450},
    "swe_care": {"hard": 750, "medium": 525, "easy": 225},
}


def _cap_reason(candidate: Candidate, accepted: list[Candidate], counters: dict[str, Counter[str]]) -> str | None:
    dataset = candidate.dataset
    if dataset == "xlam_60k":
        for signature in candidate.metadata.get("called_tool_signatures", []):
            if counters["called_tool_signature"][str(signature)] >= 10:
                return "api_signature_cap"
        # This provisional ceiling bounds memory; the exact 2% ceiling is
        # recomputed against the final xLAM tranche after global dedup.
        for category in candidate.metadata.get("api_categories", [candidate.category]):
            if counters["api_category"][str(category)] >= 40:
                return "api_category_cap"
    if dataset == "swe_care":
        source_difficulty = str(candidate.metadata.get("source_difficulty") or "").lower()
        hard_count = counters["swecare_difficulty"]["hard"]
        if source_difficulty == "medium" and counters["swecare_difficulty"]["medium"] >= math.floor(hard_count * 0.7):
            return "difficulty_mix_cap"
        if source_difficulty in {"easy", "low"} and counters["swecare_difficulty"]["easy"] >= math.floor(hard_count * 0.3):
            return "difficulty_mix_cap"
    if dataset == "nemotron_agentic_v2":
        family = str(candidate.metadata.get("upstream_source_family") or "unknown")
        if counters["source_family"][family] >= 600:
            return "tool_family_cap"
        subset = candidate.metadata.get("subset")
        if candidate.metadata.get("tool_call_count") == 1:
            allowed_simple = math.floor(counters["agentic_complexity"]["multi"] * 15 / 85)
            if counters["agentic_complexity"]["simple"] >= allowed_simple:
                return "simple_tool_case_mix_cap"
        if subset == "search":
            allowed_search = math.floor(counters["agentic_complexity"]["tool_calling"] * 5 / 95)
            if counters["agentic_complexity"]["search"] >= allowed_search:
                return "search_mix_cap"
    if dataset == "codeact_instruct":
        family = str(candidate.metadata.get("upstream_source_family") or "unknown")
        if counters["source_family"][family] >= 600:
            return "source_family_cap"
    if dataset == "swe_care" and candidate.metadata.get("estimated_review_effort", 0) < 4:
        allowed_low = min(450, math.floor(counters["swecare_effort"]["high"] * 3 / 7))
        if counters["swecare_effort"]["low"] >= allowed_low:
            return "review_effort_mix_cap"
    if dataset == "codeact_instruct" and candidate.metadata.get("action_observation_cycles", 0) < 2:
        allowed_simple = math.floor(counters["codeact_cycles"]["multi"] * 3 / 7)
        if counters["codeact_cycles"]["simple"] >= allowed_simple:
            return "simple_action_cycle_cap"
    return None


def _increment_caps(candidate: Candidate, counters: dict[str, Counter[str]]) -> None:
    if candidate.tool_signature:
        counters["tool_signature"][candidate.tool_signature] += 1
    counters["category"][candidate.category] += 1
    if candidate.dataset == "xlam_60k":
        for signature in candidate.metadata.get("called_tool_signatures", []):
            counters["called_tool_signature"][str(signature)] += 1
        for category in candidate.metadata.get("api_categories", [candidate.category]):
            counters["api_category"][str(category)] += 1
    if candidate.repository:
        counters["repository"][candidate.repository] += 1
    source_family = candidate.metadata.get("upstream_source_family") or candidate.metadata.get("constituent")
    if source_family:
        counters["source_family"][str(source_family)] += 1
    if candidate.dataset == "swe_care":
        counters["swecare_effort"]["high" if candidate.metadata.get("estimated_review_effort", 0) >= 4 else "low"] += 1
        source_difficulty = str(candidate.metadata.get("source_difficulty") or "").lower()
        counters["swecare_difficulty"]["easy" if source_difficulty in {"easy", "low"} else source_difficulty] += 1
    if candidate.dataset == "codeact_instruct":
        counters["codeact_cycles"]["multi" if candidate.metadata.get("action_observation_cycles", 0) >= 2 else "simple"] += 1
    if candidate.dataset == "nemotron_agentic_v2":
        subset = str(candidate.metadata.get("subset") or "tool_calling")
        counters["agentic_complexity"]["search" if subset == "search" else "tool_calling"] += 1
        counters["agentic_complexity"]["simple" if candidate.metadata.get("tool_call_count") == 1 else "multi"] += 1


def _global_source_family(candidate: Candidate) -> str | None:
    value: Any = None
    if candidate.dataset == "nemotron_agentic_v2":
        value = candidate.metadata.get("upstream_source_family")
    elif candidate.dataset == "codeact_instruct":
        value = candidate.metadata.get("upstream_source_family")
    elif candidate.dataset == "swe_care":
        value = candidate.repository
    elif candidate.dataset == "xlam_60k":
        value = candidate.metadata.get("api_category")
    if value in {None, "", "unknown"}:
        return None
    return f"{candidate.dataset}:{value}"


def _prune_global_source_families(
    values: list[tuple[Candidate, int]], stats: Stats,
) -> list[tuple[Candidate, int]]:
    current = values
    while current:
        cap = max(1, math.floor(len(current) * 0.03))
        counts: Counter[str] = Counter()
        kept: list[tuple[Candidate, int]] = []
        removed = 0
        for candidate, length in current:
            family = _global_source_family(candidate)
            if family and counts[family] >= cap:
                stats.reject(candidate.dataset, "global_source_family_cap", candidate.original_id)
                removed += 1
                continue
            if family:
                counts[family] += 1
            kept.append((candidate, length))
        current = kept
        if not removed:
            return current
    return current


def _prune_swecare_repositories(
    values: list[tuple[Candidate, int]], stats: Stats,
) -> list[tuple[Candidate, int]]:
    """Keep every SWE-CARE repository at or below 3% of the final tranche."""
    current = values
    while current:
        tranche_size = sum(candidate.dataset == "swe_care" for candidate, _ in current)
        if not tranche_size:
            return current
        cap = max(1, math.floor(tranche_size * 0.03))
        counts: Counter[str] = Counter()
        kept: list[tuple[Candidate, int]] = []
        removed = 0
        for candidate, length in current:
            repository = candidate.repository if candidate.dataset == "swe_care" else None
            if repository and counts[repository] >= cap:
                stats.reject(candidate.dataset, "source_repository_cap", candidate.original_id)
                removed += 1
                continue
            if repository:
                counts[repository] += 1
            kept.append((candidate, length))
        current = kept
        if not removed:
            return current
    return current


def _prune_xlam_api_categories(
    values: list[tuple[Candidate, int]], stats: Stats,
) -> list[tuple[Candidate, int]]:
    """Keep every called API-name category at or below 2% of final xLAM."""
    current = values
    while current:
        tranche_size = sum(candidate.dataset == "xlam_60k" for candidate, _ in current)
        if not tranche_size:
            return current
        cap = max(1, math.floor(tranche_size * 0.02))
        counts: Counter[str] = Counter()
        kept: list[tuple[Candidate, int]] = []
        removed = 0
        for candidate, length in current:
            categories = (
                [str(value) for value in candidate.metadata.get("api_categories", [])]
                if candidate.dataset == "xlam_60k" else []
            )
            if categories and any(counts[category] >= cap for category in categories):
                stats.reject(candidate.dataset, "api_category_final_cap", candidate.original_id)
                removed += 1
                continue
            for category in categories:
                counts[category] += 1
            kept.append((candidate, length))
        current = kept
        if not removed:
            return current
    return current


def _prune_final_distributions(
    values: list[tuple[Candidate, int]], stats: Stats,
) -> list[tuple[Candidate, int]]:
    by_dataset: dict[str, list[tuple[Candidate, int]]] = defaultdict(list)
    for value in values:
        by_dataset[value[0].dataset].append(value)
    remove: set[int] = set()

    def trim(candidates: list[tuple[Candidate, int]], allowed: int) -> None:
        for candidate, _ in candidates[max(0, allowed) :]:
            remove.add(id(candidate))

    agentic = by_dataset["nemotron_agentic_v2"]
    multi = [item for item in agentic if item[0].metadata.get("subset") == "tool_calling" and item[0].metadata.get("tool_call_count", 0) > 1]
    simple = [item for item in agentic if item[0].metadata.get("subset") == "tool_calling" and item[0].metadata.get("tool_call_count") == 1]
    search = [item for item in agentic if item[0].metadata.get("subset") == "search"]
    trim(simple, math.floor(len(multi) * 15 / 85))
    trim(search, math.floor((len(multi) + min(len(simple), math.floor(len(multi) * 15 / 85))) * 5 / 95))

    codeact = by_dataset["codeact_instruct"]
    codeact_multi = [item for item in codeact if item[0].metadata.get("action_observation_cycles", 0) >= 2]
    codeact_simple = [item for item in codeact if item[0].metadata.get("action_observation_cycles", 0) < 2]
    trim(codeact_simple, math.floor(len(codeact_multi) * 3 / 7))

    xlam = by_dataset["xlam_60k"]
    xlam_total = len(xlam)
    xlam_by_calls = {
        "one": [item for item in xlam if item[0].metadata.get("call_count") == 1],
        "two": [item for item in xlam if item[0].metadata.get("call_count") == 2],
        "three_plus": [item for item in xlam if item[0].metadata.get("call_count", 0) >= 3],
    }
    # Allow a small deterministic tolerance around the 20/55/25 target so
    # percentage rounding cannot collapse a small post-filter tranche.
    for label, ratio in (("one", 0.22), ("two", 0.58), ("three_plus", 0.28)):
        trim(xlam_by_calls[label], math.ceil(xlam_total * ratio))
        deepseek = [item for item in xlam_by_calls[label] if item[0].metadata.get("generator_partition") == "deepseek"]
        mixtral = [item for item in xlam_by_calls[label] if item[0].metadata.get("generator_partition") == "mixtral"]
        # Rows are quality ordered; balance by trimming only the majority
        # partition, retaining the best members in each.
        if len(deepseek) > len(mixtral):
            trim(deepseek, len(mixtral))
        elif len(mixtral) > len(deepseek):
            trim(mixtral, len(deepseek))

    opencode = by_dataset["open_code_instruct"]
    generic = [item for item in opencode if item[0].metadata.get("domain") == "generic"]
    algorithmic = [item for item in opencode if item[0].metadata.get("domain") == "algorithmic"]
    evol = [item for item in opencode if item[0].metadata.get("generation_algorithm") == "evol-instruct"]
    self_instruct = [item for item in opencode if item[0].metadata.get("generation_algorithm") == "self-instruct"]
    trim(algorithmic, math.floor(len(generic) * 3 / 7))
    trim(self_instruct, math.floor(len(evol) * 2 / 3))

    swecare = by_dataset["swe_care"]
    hard = [item for item in swecare if item[0].metadata.get("source_difficulty") == "hard"]
    medium = [item for item in swecare if item[0].metadata.get("source_difficulty") == "medium"]
    easy = [item for item in swecare if item[0].metadata.get("source_difficulty") == "easy"]
    high_effort = [item for item in swecare if item[0].metadata.get("estimated_review_effort", 0) >= 4]
    low_effort = [item for item in swecare if item[0].metadata.get("estimated_review_effort", 0) < 4]
    trim(medium, math.floor(len(hard) * 0.7))
    trim(easy, math.floor(len(hard) * 0.3))
    trim(low_effort, math.floor(len(high_effort) * 3 / 7))

    kept = []
    for candidate, length in values:
        if id(candidate) in remove:
            stats.reject(candidate.dataset, "post_dedup_distribution_cap", candidate.original_id)
        else:
            kept.append((candidate, length))
    return kept


def _apply_final_constraints(
    values: list[tuple[Candidate, int]], stats: Stats,
) -> list[tuple[Candidate, int]]:
    """Re-apply interacting percentage constraints until a fixed point."""
    current = values
    while True:
        before = len(current)
        current = _prune_global_source_families(current, stats)
        current = _prune_final_distributions(current, stats)
        current = _prune_swecare_repositories(current, stats)
        current = _prune_xlam_api_categories(current, stats)
        if len(current) == before:
            return current


def select(
    pools_by_dataset: dict[str, dict[str, list[Candidate]]], tokenizer: Any,
    eval_manifest: dict[str, Any], stats: Stats,
) -> list[tuple[Candidate, int]]:
    provisional: list[tuple[Candidate, int]] = []
    local_dedup: dict[tuple[str, str], Deduplicator] = defaultdict(Deduplicator)
    held_out_ids = {str(item.get("id")) for item in eval_manifest.get("records", []) if item.get("id")}
    held_out_id_hashes = {str(item.get("id_sha256")) for item in eval_manifest.get("records", []) if item.get("id_sha256")}
    for dataset in DATASETS:
        accepted: list[Candidate] = []
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        for bucket, quota in QUOTAS[dataset].items():
            bucket_count = 0
            for candidate in pools_by_dataset.get(dataset, {}).get(bucket, []):
                if bucket_count >= quota:
                    stats.reject(dataset, "bucket_ceiling", candidate.original_id)
                    continue
                reason = _cap_reason(candidate, accepted, counters)
                if reason:
                    stats.reject(dataset, reason, candidate.original_id)
                    continue
                reason, cluster = local_dedup[(dataset, bucket)].match(candidate)
                if reason:
                    stats.reject(dataset, reason, candidate.original_id, cluster=f"local:{dataset}:{bucket}:{cluster}")
                    continue
                if candidate.original_id in held_out_ids or sha256_text(candidate.original_id) in held_out_id_hashes:
                    stats.reject(dataset, "eval_v1_id_leakage", candidate.original_id)
                    continue
                try:
                    probe = candidate_record(candidate, "train", 0)
                    examples = expand_training_examples([probe])
                    token_metrics = validate_tokenized_examples(tokenizer, examples, 1_000_000_000)
                    length = token_metrics["maximum"]
                    candidate.metadata["training_example_count"] = token_metrics["training_examples"]
                except Exception:
                    stats.reject(dataset, "tokenizer_error", candidate.original_id)
                    continue
                if length > MAX_SEQUENCE_LENGTH:
                    stats.reject(dataset, "actual_tokenizer_overlength", candidate.original_id)
                    continue
                if length < 8:
                    stats.reject(dataset, "empty_or_too_short", candidate.original_id)
                    continue
                probe = candidate_record(candidate, "train", length)
                leaks = find_manifest_leaks([probe], eval_manifest)
                if leaks:
                    stats.reject(dataset, "eval_v1_leakage", candidate.original_id)
                    continue
                local_dedup[(dataset, bucket)].add(candidate)
                _increment_caps(candidate, counters)
                accepted.append(candidate)
                provisional.append((candidate, length))
                bucket_count += 1
    # Cross-dataset duplicate retention is quality-first and may reduce, never inflate, ceilings.
    dedup = Deduplicator(include_dataset_keys=True)
    final: list[tuple[Candidate, int]] = []
    for candidate, length in sorted(provisional, key=lambda item: item[0].rank_key):
        reason, cluster = dedup.match(candidate)
        if reason:
            stats.reject(
                candidate.dataset, f"cross_dataset_{reason}", candidate.original_id,
                cluster=f"global:{cluster}",
            )
            continue
        dedup.add(candidate)
        final.append((candidate, length))
    return _apply_final_constraints(final, stats)


def _distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(field)) for record in records).items()))


def _training_example_summary(records: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    """Return frozen per-split and total expanded-example counts."""
    by_split = {
        split: sum(
            int(record["metadata"].get("training_example_count", 1))
            for record in records if record["split"] == split
        )
        for split in sorted({str(record["split"]) for record in records})
    }
    total = sum(by_split.values())
    return {**by_split, "total": total}, total


def _temporary_sibling(path: Path) -> Path:
    """Reserve a same-directory path for an atomic artifact replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
    )
    handle.close()
    return Path(handle.name)


def _write_json_lf(path: Path, value: Any) -> None:
    """Write a checkout-stable JSON artifact with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl_lf(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Stream deterministic JSONL without platform newline translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _source_inventory(root: Path, *, show_progress: bool = False) -> list[dict[str, Any]]:
    """Verify every approved local byte against the immutable acquisition lock."""
    inventory: list[dict[str, Any]] = []
    for dataset in DATASETS:
        if show_progress:
            print(canonical_json({"stage": "verify_source", "dataset": dataset}), file=sys.stderr, flush=True)
        locked_source = ACQUISITION_LOCK["datasets"][dataset]
        verified = verify_dataset(dataset, root, require_all=True)
        local_directory = root / locked_source["local_directory"]
        for entry in locked_source["files"]:
            local_path = str(entry["local_path"])
            identity = verified["files"][local_path]
            path = local_directory / Path(*local_path.split("/"))
            item: dict[str, Any] = {
                "dataset": dataset,
                "repo_id": locked_source["repo_id"],
                "revision": locked_source["revision"],
                "remote_path": entry["remote_path"],
                "path": path.relative_to(root).as_posix(),
                "bytes": identity["size"],
            }
            if "sha256" in identity:
                item["sha256"] = identity["sha256"]
                item["identity"] = "sha256"
            else:
                item["git_oid"] = identity["git_oid"]
                item["identity"] = "git-blob-sha1"
            inventory.append(item)
    return sorted(inventory, key=lambda item: (item["dataset"], item["remote_path"]))


def load_tokenizer(cache: Path, allow_download: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the corpus extra: pip install -e .[corpus]") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        cache_dir=str(cache),
        local_files_only=not allow_download,
        trust_remote_code=False,
        token=os.environ.get("HF_TOKEN"),
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned gpt-oss tokenizer has no native chat template")
    return tokenizer


def build(
    *, source_root: Path, corpus_path: Path, sources_path: Path, manifest_path: Path,
    report_path: Path, tokenizer: Any, include_source_hashes: bool = True,
    runtime_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_info = runtime_info or build_runtime(require_exact=False)
    if not RUNTIME_REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing corpus runtime lock: {RUNTIME_REQUIREMENTS}")
    # Verify the immutable acquisition lock before parsing any source content
    # or replacing a previously valid corpus artifact.
    source_inventory = _source_inventory(source_root, show_progress=True) if include_source_hashes else []
    stats = Stats()
    loaders = {
        "nemotron_swe_v2": load_nemotron_swe,
        "open_code_instruct": load_opencode,
        "nemotron_agentic_v2": load_nemotron_agentic,
        "xlam_60k": load_xlam,
        "codeact_instruct": load_codeact,
        "swe_care": load_swecare,
    }
    pools: dict[str, dict[str, list[Candidate]]] = {}
    for dataset in DATASETS:
        if include_source_hashes:
            print(canonical_json({"stage": "load", "dataset": dataset}), file=sys.stderr, flush=True)
        pools[dataset], _ = loaders[dataset](source_root, stats)
    if include_source_hashes:
        print(canonical_json({"stage": "select"}), file=sys.stderr, flush=True)
    eval_manifest = load_eval_manifest(EVAL_MANIFEST)
    selected = select(pools, tokenizer, eval_manifest, stats)
    split_map = grouped_split(candidate for candidate, _ in selected)
    records = [candidate_record(candidate, split_map[candidate.family], length) for candidate, length in selected]
    paired = sorted(zip(records, selected), key=lambda item: item[0]["id"])
    records = [item[0] for item in paired]
    selected = [item[1] for item in paired]
    source_rows = [source_record(record, candidate) for record, (candidate, _) in zip(records, selected, strict=True)]
    accepted_counter = Counter(candidate.dataset for candidate, _ in selected)
    accepted_counts = {dataset: accepted_counter.get(dataset, 0) for dataset in DATASETS}
    accounting: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        considered = stats.considered[dataset]
        accepted = accepted_counts.get(dataset, 0)
        rejected = sum(stats.rejections[dataset].values())
        if considered != accepted + rejected:
            raise RuntimeError(
                f"Selection accounting mismatch for {dataset}: considered={considered}, accepted={accepted}, rejected={rejected}"
            )
        accounting[dataset] = {"considered": considered, "accepted": accepted, "rejected": rejected}
    split_counts = _distribution(records, "split")
    training_example_counts, training_examples = _training_example_summary(records)
    token_counts = [record["metadata"]["token_count"] for record in records]
    family_counts = Counter(candidate.family for candidate, _ in selected)
    locked_file_count = sum(
        len(source["files"]) for source in ACQUISITION_LOCK["datasets"].values()
    )

    destinations = [corpus_path, sources_path, manifest_path, report_path]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise RuntimeError("Corpus, source ledger, manifest and report paths must be distinct")
    staged: dict[Path, Path] = {}
    try:
        # Reserve each path inside the protected region so an allocation failure
        # cannot strand siblings that were created earlier in the sequence.
        for destination in destinations:
            staged[destination] = _temporary_sibling(destination)
        _write_jsonl_lf(staged[corpus_path], records)
        _write_jsonl_lf(staged[sources_path], source_rows)
        corpus_digest = sha256_file(staged[corpus_path])
        sources_digest = sha256_file(staged[sources_path])

        manifest = {
            "schema_version": 1,
            "artifact_type": "external_corpus_manifest",
            "name": "External Corpus v1",
            "transformation_version": TRANSFORMATION_VERSION,
            "acquisition_date": ACQUISITION_DATE,
            "deterministic_seed": 3407,
            "maximum_records": 20000,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "tokenizer": {"repo_id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "chat_template": "native"},
            "build_runtime": runtime_info,
            "runtime_requirements": {
                "path": RUNTIME_REQUIREMENTS.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(RUNTIME_REQUIREMENTS),
            },
            "acquisition_lock": {
                "path": LOCK_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(LOCK_PATH),
                "format_version": ACQUISITION_LOCK["format_version"],
                "file_count": locked_file_count,
            },
            "eval_manifest_records_sha256": eval_manifest["records_sha256"],
            "records": len(records),
            "training_examples": training_examples,
            "accepted_counts": accepted_counts,
            "split_counts": split_counts,
            "training_example_counts": training_example_counts,
            "corpus_sha256": corpus_digest,
            "sources_sha256": sources_digest,
            "datasets": {name: spec for name, spec in DATASETS.items()},
            "source_files": source_inventory,
            "rules": {
                "quality_tier": "B",
                "exact_duplicate": "normalized prompt+target SHA-256",
                "near_duplicate": "global 5-token shingle Jaccard >= 0.85",
                "code_duplicate": "global canonical token signature Jaccard >= 0.90",
                "sensitive_credentials": "reject provider-shaped credential and private-key patterns",
                "split": "90/10 deterministic family-grouped train/validation",
                "held_out_eval_created": False,
                "swe_care_test_split_used": False,
                "merged_patch_in_training_content": False,
            },
        }
        report = {
            "artifact_type": "external_corpus_statistics",
            "transformation_version": TRANSFORMATION_VERSION,
            "considered": dict(sorted(stats.considered.items())),
            "accepted": accepted_counts,
            "accounting": accounting,
            "rejections": {dataset: dict(sorted(reasons.items())) for dataset, reasons in sorted(stats.rejections.items())},
            "rejected_id_samples": {
                dataset: {reason: values for reason, values in sorted(reasons.items())}
                for dataset, reasons in sorted(stats.rejected_ids.items())
            },
            "duplicate_cluster_counts": {reason: len(clusters) for reason, clusters in sorted(stats.duplicate_clusters.items())},
            "duplicate_rejection_counts": {
                reason: sum(reasons.get(reason, 0) for reasons in stats.rejections.values())
                for reason in sorted(stats.duplicate_clusters)
            },
            "leakage_rejections": sum(
                reasons.get("eval_v1_leakage", 0) + reasons.get("eval_v1_id_leakage", 0)
                for reasons in stats.rejections.values()
            ),
            "source_stats": _distribution(records, "source"),
            "category_stats": dict(sorted(Counter(f"{candidate.dataset}:{candidate.category}" for candidate, _ in selected).items())),
            "xlam_api_category_stats": dict(sorted(Counter(
                category
                for candidate, _ in selected if candidate.dataset == "xlam_60k"
                for category in candidate.metadata.get("api_categories", [])
            ).items())),
            "upstream_source_family_stats": dict(sorted(Counter(
                f"{candidate.dataset}:{candidate.metadata['upstream_source_family']}"
                for candidate, _ in selected if candidate.metadata.get("upstream_source_family")
            ).items())),
            "subskill_stats": dict(sorted(Counter(f"{candidate.dataset}:{candidate.subskill}" for candidate, _ in selected).items())),
            "source_repository_stats": dict(sorted(Counter(candidate.repository for candidate, _ in selected if candidate.repository).items())),
            "license_stats": _distribution(records, "license"),
            "task_stats": _distribution(records, "task_type"),
            "split_stats": split_counts,
            "training_example_stats": {
                "total": training_examples,
                "by_split": {key: value for key, value in training_example_counts.items() if key != "total"},
            },
            "family_stats": {"families": len(family_counts), "maximum_family_size": max(family_counts.values(), default=0)},
            "tokenizer_lengths": {
                "minimum": min(token_counts, default=0), "maximum": max(token_counts, default=0),
                "mean": round(sum(token_counts) / len(token_counts), 3) if token_counts else 0,
                "p50": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
                "p95": sorted(token_counts)[min(len(token_counts) - 1, math.floor(len(token_counts) * 0.95))] if token_counts else 0,
            },
            "corpus_sha256": corpus_digest,
            "sources_sha256": sources_digest,
        }
        _write_json_lf(staged[report_path], report)
        _write_json_lf(staged[manifest_path], manifest)
        # The manifest is the commit marker and is replaced last. If a process
        # is interrupted between replacements, every consumer fails closed on
        # its digest checks instead of accepting a partially updated build.
        for destination in (corpus_path, sources_path, report_path, manifest_path):
            os.replace(staged[destination], destination)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return {"manifest": manifest, "report": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tokenizer-cache", type=Path, default=TOKENIZER_CACHE)
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    args = parser.parse_args(argv)
    runtime_info = build_runtime(require_exact=True)
    tokenizer = load_tokenizer(args.tokenizer_cache, args.allow_tokenizer_download)
    result = build(
        source_root=args.source_root.resolve(), corpus_path=args.corpus.resolve(),
        sources_path=args.sources.resolve(), manifest_path=args.manifest.resolve(),
        report_path=args.report.resolve(), tokenizer=tokenizer, runtime_info=runtime_info,
    )
    manifest = result["manifest"]
    print(canonical_json({"records": manifest["records"], "accepted_counts": manifest["accepted_counts"], "split_counts": manifest["split_counts"], "corpus_sha256": manifest["corpus_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
