#!/usr/bin/env python3
"""Screen a built external corpus for suspicious semantic overlap with Eval v1.

This is a deterministic, offline audit.  It never sends held-out material to a
model and its report contains identifiers, aggregate counts, and similarity
scores only.  A non-empty report is a review/rejection gate, not permission to
copy held-out content into a repair prompt.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import content_text, load_jsonl, normalize_text, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "training/datasets/external_corpus_v1.jsonl"
DEFAULT_EVAL_ROOT = ROOT / "training/evals"
DEFAULT_REPORT = ROOT / "training/reports/external_corpus_v1_semantic_leakage.json"

TOKEN_RE = re.compile(r"[a-z0-9_./:+#-]+", re.IGNORECASE)
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "do", "does", "for", "from", "had", "has", "have", "how",
        "i", "if", "in", "into", "is", "it", "its", "may", "not", "of",
        "on", "or", "our", "should", "so", "that", "the", "their", "then",
        "this", "to", "use", "using", "was", "we", "what", "when", "where",
        "which", "with", "would", "you", "your",
    }
)


def _message_text(record: dict[str, Any]) -> str:
    """Return only material that can become a training target or prompt."""
    parts = [content_text(record)]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        native_messages = metadata.get("native_messages")
        if isinstance(native_messages, list):
            for message in native_messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function")
                        if isinstance(function, dict):
                            parts.extend(str(function.get(key) or "") for key in ("name", "arguments"))
        native_targets = metadata.get("native_call_targets")
        if isinstance(native_targets, list):
            parts.append(json.dumps(native_targets, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(normalize_text(value))
        if len(token) > 1 and token not in STOP_WORDS
    )


def _ngrams(tokens: tuple[str, ...], width: int) -> set[tuple[str, ...]]:
    if len(tokens) < width:
        return set()
    return {tokens[index : index + width] for index in range(len(tokens) - width + 1)}


def _similarity_from_tokens(
    left: str,
    right: str,
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> dict[str, float | int]:
    left_set, right_set = set(left_tokens), set(right_tokens)
    shared = left_set & right_set
    union = left_set | right_set
    smaller = min(len(left_set), len(right_set))
    left_bigrams, right_bigrams = _ngrams(left_tokens, 2), _ngrams(right_tokens, 2)
    bigram_union = left_bigrams | right_bigrams
    token_jaccard = len(shared) / len(union) if union else 0.0
    shorter_coverage = len(shared) / smaller if smaller else 0.0
    bigram_jaccard = (
        len(left_bigrams & right_bigrams) / len(bigram_union)
        if bigram_union else 0.0
    )
    # SequenceMatcher is the expensive signal.  Only compute it after a loose
    # token prefilter, and bound unusually large traces deterministically.
    sequence_ratio = 0.0
    if len(shared) >= 6 and (
        token_jaccard >= 0.15 or shorter_coverage >= 0.25 or bigram_jaccard >= 0.08
    ):
        left_norm = normalize_text(left)[:12_000]
        right_norm = normalize_text(right)[:12_000]
        sequence_ratio = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    return {
        "shared_significant_tokens": len(shared),
        "token_jaccard": round(token_jaccard, 6),
        "shorter_token_coverage": round(shorter_coverage, 6),
        "bigram_jaccard": round(bigram_jaccard, 6),
        "sequence_ratio": round(sequence_ratio, 6),
    }


def similarity_metrics(left: str, right: str) -> dict[str, float | int]:
    return _similarity_from_tokens(left, right, _tokens(left), _tokens(right))


def suspicion_reasons(metrics: dict[str, float | int]) -> list[str]:
    shared = int(metrics["shared_significant_tokens"])
    reasons: list[str] = []
    if shared >= 8 and float(metrics["token_jaccard"]) >= 0.34:
        reasons.append("high_token_jaccard")
    if shared >= 8 and float(metrics["shorter_token_coverage"]) >= 0.62:
        reasons.append("high_shorter_side_coverage")
    if shared >= 6 and float(metrics["bigram_jaccard"]) >= 0.22:
        reasons.append("high_bigram_jaccard")
    if shared >= 6 and float(metrics["sequence_ratio"]) >= 0.58:
        reasons.append("high_sequence_similarity")
    return reasons


def screen_records(
    corpus_rows: Iterable[dict[str, Any]], eval_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    eval_material = [
        (str(row.get("id") or "unknown"), text, _tokens(text))
        for row in eval_rows
        for text in [_message_text(row)]
    ]
    findings: list[dict[str, Any]] = []
    for record in corpus_rows:
        record_id = str(record.get("id") or "unknown")
        material = _message_text(record)
        provenance = record.get("provenance")
        upstream_id = (
            provenance.get("original_id") or provenance.get("upstream_id")
            if isinstance(provenance, dict) else None
        )
        material_tokens = _tokens(material)
        for eval_id, eval_text, eval_tokens in eval_material:
            metrics = _similarity_from_tokens(material, eval_text, material_tokens, eval_tokens)
            reasons = suspicion_reasons(metrics)
            if not reasons:
                continue
            findings.append(
                {
                    "record_id": record_id,
                    "source": str(record.get("source") or "unknown"),
                    "upstream_id": str(upstream_id or "unknown"),
                    "eval_id": eval_id,
                    "reasons": reasons,
                    "metrics": metrics,
                }
            )
    return sorted(findings, key=lambda item: (item["record_id"], item["eval_id"]))


def load_eval_rows(eval_root: Path) -> list[dict[str, Any]]:
    paths = sorted(eval_root.glob("*/eval_v1_seed.jsonl"))
    if not paths:
        raise RuntimeError(f"No Eval v1 seed files found below {eval_root}")
    rows = [row for path in paths for row in load_jsonl(path)]
    ids = [str(row.get("id") or "") for row in rows]
    if not ids or any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Eval v1 IDs must be present and unique")
    return rows


def audit(corpus_path: Path, eval_root: Path, report_path: Path) -> dict[str, Any]:
    corpus_rows = load_jsonl(corpus_path)
    eval_rows = load_eval_rows(eval_root)
    findings = screen_records(corpus_rows, eval_rows)
    report = {
        "artifact_type": "external_corpus_semantic_leakage_audit",
        "contains_plaintext_eval_content": False,
        "contains_plaintext_corpus_content": False,
        "corpus_records": len(corpus_rows),
        "eval_records": len(eval_rows),
        "finding_count": len(findings),
        "reason_counts": dict(sorted(Counter(reason for item in findings for reason in item["reasons"]).items())),
        "findings": findings,
    }
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-findings", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.corpus.resolve(), args.eval_root.resolve(), args.report.resolve())
    print(json.dumps({"finding_count": report["finding_count"], "report": str(args.report)}, sort_keys=True))
    return 0 if args.allow_findings or not report["findings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
