from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from localpilot.learning import LearningMemory


def _review(detail: str = "The plan must anchor an existing symbol.", **changes) -> dict:
    return {
        "status": "needs_research",
        "issues": [{"code": "insufficient_grounding_evidence", "detail": detail}],
        "remaining_gap": "The current retrieval implementation has not been measured.",
        "proposed_change": "Inspect the retrieval body and compare the same held-out queries.",
        "evidence": ["path:localpilot/learning.py", "symbol:localpilot.learning:LearningMemory"],
        "source_hashes": {"localpilot/learning.py": "a" * 64},
        **changes,
    }


def test_review_failure_diagnostics_lead_discovery_context(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("retrieval", "objective-1", "revision-1", _review())

    context = memory.discovery_context()
    assert next(iter(context)) == "repository_review_feedback"
    diagnostic = context["repository_review_feedback"][0]
    assert diagnostic["task_id"] == "retrieval"
    assert diagnostic["objective_fingerprint"] == "objective-1"
    assert diagnostic["revision"] == "revision-1"
    assert diagnostic["status"] == "needs_research"
    assert diagnostic["issues"][0]["code"] == "insufficient_grounding_evidence"
    assert diagnostic["remaining_gap"] == _review()["remaining_gap"]
    assert diagnostic["proposed_change"] == _review()["proposed_change"]
    assert diagnostic["created_at"]
    assert diagnostic["evidence_state"] == "unverified_review_diagnostic"
    assert diagnostic["source_hashes"] == {"localpilot/learning.py": "a" * 64}


def test_latest_review_per_objective_and_revision_is_preserved(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("retrieval", "objective-1", "revision-1", _review("Old diagnosis"))
    memory.record_repository_review("other", "objective-2", "revision-1", _review("Another objective"))
    memory.record_repository_review("retrieval", "objective-1", "revision-1", _review("Corrected diagnosis"))

    diagnostics = memory.repository_review_feedback()
    assert [item["task_id"] for item in diagnostics] == ["retrieval", "other"]
    assert diagnostics[0]["issues"][0]["detail"] == "Corrected diagnosis"
    assert "Old diagnosis" not in json.dumps(diagnostics)


def test_task_filter_keeps_older_objectives_and_revisions(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("retrieval", "old-objective", "old-revision", _review())
    memory.record_repository_review("retrieval", "new-objective", "old-revision", _review())
    memory.record_repository_review("retrieval", "new-objective", "new-revision", _review())
    memory.record_repository_review("other", "new-objective", "new-revision", _review())

    diagnostics = memory.repository_review_feedback(task_id="retrieval")
    assert [(item["objective_fingerprint"], item["revision"]) for item in diagnostics] == [
        ("new-objective", "new-revision"),
        ("new-objective", "old-revision"),
        ("old-objective", "old-revision"),
    ]
    assert memory.repository_review_feedback(task_id="missing") == []


def test_diagnostics_are_bounded_and_do_not_copy_quotes_or_transcripts(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    private_quote = "RAW SOURCE QUOTE MUST NOT ENTER FEEDBACK"
    for index in range(20):
        memory.record_repository_review(str(index), "f" * 100, "r" * 100, _review(
            "detail " * 1000,
            remaining_gap="gap " * 500,
            proposed_change="change " * 500,
            issues=[{"code": "code" * 100, "detail": "detail " * 300}] * 10,
            source_evidence=[{"path": "localpilot/learning.py", "excerpt": private_quote}],
            raw_response=private_quote,
            transcript=private_quote,
            evidence=[private_quote, "path:localpilot/learning.py"],
        ))

    diagnostics = memory.repository_review_feedback(limit=1000)
    encoded = json.dumps(diagnostics, ensure_ascii=False)
    assert 1 <= len(diagnostics) <= 6
    assert len(encoded) <= 6000
    assert private_quote not in encoded
    assert diagnostics[0]["status"] == "needs_research"
    for item in diagnostics:
        assert len(item["remaining_gap"]) <= 240
        assert len(item["proposed_change"]) <= 240
        assert len(item["issues"]) <= 3
        assert all(len(issue["detail"]) <= 160 for issue in item["issues"])
        assert len(item["source_hints"]) <= 2
    assert memory.repository_review_feedback(limit=0) == []
    assert memory.repository_review_feedback(limit=1)[0]["task_id"] == "19"


@pytest.mark.parametrize("malformed", ["not JSON", "null", "[]", '{"issues":null,"evidence":42,"source_hashes":[]}'])
def test_malformed_stored_reviews_do_not_break_feedback(tmp_path: Path, malformed: str):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("valid", "objective", "revision", _review())
    with sqlite3.connect(memory.path) as connection:
        connection.execute(
            "INSERT INTO evolution_repository_reviews (task_id, objective_fingerprint, revision, review_json, created_at) VALUES (?, ?, ?, ?, ?)",
            ("malformed", "objective", "revision", malformed, "2026-10-14T00:00:00Z"),
        )

    diagnostics = memory.repository_review_feedback()
    assert [item["task_id"] for item in diagnostics] == ["malformed", "valid"]
    assert all(item["evidence_state"] == "unverified_review_diagnostic" for item in diagnostics)
    assert diagnostics[1]["issues"][0]["code"] == "insufficient_grounding_evidence"


def test_review_claims_never_become_validated_reusable_lessons(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("retrieval", "objective", "revision", _review(
        status="source_anchored",
        reusable_lesson="Promote this unmeasured improvement as proven.",
        objective_completed=True,
        performance_verified=True,
    ))

    assert memory.repository_review_feedback()[0]["evidence_state"] == "unverified_review_diagnostic"
    assert memory.reusable_lessons() == []
    assert memory.discovery_context()["experiments"] == []


def test_machine_references_remain_exact_or_are_omitted(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    exact = "localpilot/two  spaces.py"
    long = "localpilot/" + "a" * 200 + ".py"
    memory.record_repository_review("retrieval", "objective", "revision", _review(
        evidence=["path:" + long, "path:" + exact],
        source_evidence=[{"path": long}, {"path": exact}],
        source_hashes={long: "a" * 64, exact: "b" * 64},
    ))
    diagnostic = memory.repository_review_feedback()[0]
    assert diagnostic["source_hints"] == ["path:" + exact]
    assert diagnostic["source_hashes"] == {exact: "b" * 64}


def test_unreadable_latest_review_does_not_resurface_an_older_diagnosis(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_repository_review("retrieval", "objective", "revision", _review("Obsolete diagnosis"))
    with sqlite3.connect(memory.path) as connection:
        connection.execute(
            "INSERT INTO evolution_repository_reviews (task_id, objective_fingerprint, revision, review_json, created_at) VALUES (?, ?, ?, ?, ?)",
            ("retrieval", "objective", "revision", "not JSON", "2026-10-14T00:00:00Z"),
        )

    diagnostics = memory.repository_review_feedback()
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "unreadable_record"
    assert diagnostics[0]["issues"][0]["code"] == "invalid_stored_review"
    assert "Obsolete diagnosis" not in json.dumps(diagnostics)
