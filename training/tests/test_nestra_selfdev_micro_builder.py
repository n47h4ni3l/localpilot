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
            "source_commit": (
                builder.UPSTREAM_COMMIT if r["id"].startswith("nestra-selfdev-micro-external")
                else builder.PROVENANCE_COMMIT
            ),
            "source_path": (
                builder.UPSTREAM_PATH if r["id"].startswith("nestra-selfdev-micro-external")
                else r["provenance"]["files"][0]
            ),
        } for r in records],
    }


def test_staging_is_frozen_not_trainable_and_has_correct_distribution():
    records, digest = builder.read_candidates()
    assert len(records) == 80 and len(digest) == 64
    assert sum(x["split"] == "train" for x in records) == 68
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
    assert len(converted) == 80
    assert sum(r["quality_tier"] == "C" and
               r["verification_status"] == "synthetic_validated" for r in converted) == 76
    assert sum(r["quality_tier"] == "B" and
               r["verification_status"] == "source_verified" for r in converted) == 4
    assert all(r["provenance"]["review"]["reviewer"] == "Independent evaluator"
               for r in converted)
    assert all(r["metadata"]["candidate_sha256"] == digest for r in converted)
    assert all(r["quality_tier"] == "D" for r in records)  # No source mutation.


def test_changed_source_cannot_keep_a_valid_candidate_digest(tmp_path):
    records, digest = builder.read_candidates()
    # The override path is the local 64/12 file, not the combined 80 rows.
    modified = copy.deepcopy(records[:76])
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
    assert counts == {"train": 68, "validation": 12}



def test_external_self_instruct_seeds_are_pinned_and_nontrainable():
    source = SCRIPT.parents[1] / "sources" / "nestra_p1_selfdev_external_candidates.jsonl"
    records = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(records) == 4
    assert {r["provenance"]["source_id"] for r in records} == {
        "seed_task_92", "seed_task_70", "seed_task_37", "seed_task_19",
    }
    for record in records:
        assert record["split"] == "train"
        assert record["quality_tier"] == "D"
        assert record["verification_status"] == "review_required"
        assert record["metadata"]["candidate_only"] is True
        assert record["license"] == "Apache-2.0"
        assert record["provenance"]["upstream_commit"] == (
            "0b26ccaa415992100fa32df62d41b994cf928e23"
        )
        assert record["provenance"]["upstream_path"] == "data/seed_tasks.jsonl"
    original, _ = builder.read_candidates()
    assert len(original) == 80
    assert sum(r["id"].startswith("nestra-selfdev-micro-external")
               for r in original) == 4



def test_external_file_is_required_for_build(tmp_path):
    empty = tmp_path / "missing.jsonl"
    with pytest.raises(OSError):
        builder.read_candidates(external_path=empty)


def test_reject_missing_or_duplicate_external_record(tmp_path):
    external = builder.EXTERNAL_STAGED.read_text(encoding="utf-8").splitlines()
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("\n".join(external[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required 64 local"):
        builder.read_candidates(external_path=incomplete)
    duplicated = tmp_path / "duplicated.jsonl"
    duplicated.write_text("\n".join(external[:3] + [external[0]]) + "\n",
                          encoding="utf-8")
    with pytest.raises(ValueError, match="required Self-Instruct seed ID"):
        builder.read_candidates(external_path=duplicated)


def test_external_tampering_invalidates_review_signature(tmp_path):
    records, digest = builder.read_candidates()
    changed = tmp_path / "external.jsonl"
    lines = builder.EXTERNAL_STAGED.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["messages"][1]["content"] += " unverified"
    lines[0] = json.dumps(row)
    changed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    edited, changed_digest = builder.read_candidates(external_path=changed)
    assert changed_digest != digest
    with pytest.raises(ValueError, match="frozen candidate bytes"):
        builder.verify_review(edited, changed_digest, _approvals(records, digest),
                              verify_git=False)


def test_reject_external_claimed_license_and_wrong_review_commit(tmp_path):
    rows = builder.EXTERNAL_STAGED.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["license"] = "unknown"
    rows[0] = json.dumps(first)
    tampered = tmp_path / "external-license.jsonl"
    tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned licensed external"):
        builder.read_candidates(external_path=tampered)
    records, digest = builder.read_candidates()
    approval = _approvals(records, digest)
    external = next(a for a in approval["approvals"]
                    if a["id"].startswith("nestra-selfdev-micro-external"))
    external["source_commit"] = builder.PROVENANCE_COMMIT
    with pytest.raises(ValueError, match="reviewed source"):
        builder.verify_review(records, digest, approval, verify_git=False)
