"""Unit tests for the fail-closed micro-continuation corpus review boundary.

No training, model weights, Ollama, private heldouts or reviewer approvals.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_nestra_selfdev_micro.py"
SPEC = importlib.util.spec_from_file_location("nestra_selfdev_micro_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _approvals(records, candidate_sha):
    return {
        "candidate_sha256": candidate_sha,
        "approvals": [{
            "id": r["id"],
            "approved": True,
            "reviewer": "Independent evaluator",
            "rationale": "Compared the answer and its stated limits to immutable source behavior.",
            "source_commit": builder.PROVENANCE_COMMIT,
            "source_path": r["provenance"]["files"][0],
        } for r in records],
    }


def test_staging_is_frozen_not_trainable_and_has_correct_distribution():
    records, digest = builder.read_candidates()
    assert len(records) == 76 and len(digest) == 64
    assert sum(x["split"] == "train" for x in records) == 64
    assert sum(x["split"] == "validation" for x in records) == 12
    assert all(x["quality_tier"] == "D" for x in records)
    assert all(x["verification_status"] == "review_required" for x in records)
    assert all(x["metadata"]["candidate_only"] for x in records)
    # A local experiment may already have approved artifacts; the candidate
    # file itself must remain explicitly non-trainable regardless.


def test_reject_incomplete_review_and_changed_candidate_hash():
    records, digest = builder.read_candidates()
    approvals = _approvals(records, digest)
    with pytest.raises(ValueError, match="incomplete"):
        builder.verify_review(records, digest, {**approvals, "approvals": approvals["approvals"][:-1]}, verify_git=False)
    with pytest.raises(ValueError, match="frozen candidate bytes"):
        builder.verify_review(records, digest, {**approvals, "candidate_sha256": "0" * 64}, verify_git=False)


def test_reject_source_mismatch_and_missing_rationale():
    records, digest = builder.read_candidates()
    data = _approvals(records, digest)
    data["approvals"][0]["source_path"] = "not/source/README.md"
    with pytest.raises(ValueError, match="reviewed source"):
        builder.verify_review(records, digest, data, verify_git=False)
    data = _approvals(records, digest)
    data["approvals"][0]["rationale"] = "fine"
    with pytest.raises(ValueError, match="approval or reasoning"):
        builder.verify_review(records, digest, data, verify_git=False)


def test_reviewed_data_but_not_original_candidates_can_become_tier_c():
    records, digest = builder.read_candidates()
    converted = builder.verify_review(records, digest, _approvals(records, digest), verify_git=False)
    assert len(converted) == 76
    assert all(r["quality_tier"] == "C" and
               r["verification_status"] == "synthetic_validated" for r in converted)
    assert all(r["provenance"]["review"]["reviewer"] == "Independent evaluator"
               for r in converted)
    assert all(r["metadata"]["candidate_sha256"] == digest for r in converted)
    assert all(r["quality_tier"] == "D" for r in records)  # No source mutation.


def test_changed_source_cannot_keep_a_valid_candidate_digest(tmp_path):
    records, digest = builder.read_candidates()
    modified = copy.deepcopy(records)
    modified[0]["messages"][1]["content"] += " EXTRA"
    staged = tmp_path / "staged.jsonl"
    staged.write_text("".join(json.dumps(r) + "\n" for r in modified), encoding="utf-8")
    new_records, new_digest = builder.read_candidates(staged)
    assert digest != new_digest
    approvals = _approvals(records, digest)
    with pytest.raises(ValueError, match="frozen candidate bytes"):
        builder.verify_review(new_records, new_digest, approvals, verify_git=False)


def test_reviewed_training_pairs_have_one_target_per_record():
    records, digest = builder.read_candidates()
    reviewed = builder.verify_review(records, digest, _approvals(records, digest), verify_git=False)
    counts = {split: len(builder.expand_training_examples(reviewed, split))
              for split in ("train", "validation")}
    assert counts == {"train": 64, "validation": 12}
