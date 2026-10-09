"""Positive quality feedback must never become a benchmark reward or bypass."""
import json
import sqlite3
from pathlib import Path

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.cli import build_parser
from localpilot.config import Config
from localpilot.quality_feedback import QualityFeedbackStore, CATEGORIES


def _scores(**changes):
    return dict.fromkeys(CATEGORIES, 4) | changes


def _record(store: QualityFeedbackStore, task_id: str = "owner-repair-001", **changes):
    kwargs = {
        "task_id": task_id, "model": "nestra:20b-p1",
        "topic": "Causal diagnosis", "outcome": "verified_success",
        "scores": _scores(), "evidence_ref": "sha256:abcdef123456",
        "review_note": "Owner independently verified the repaired symptom was gone.",
        "human_attested": True,
    }
    kwargs.update(changes)
    return store.record(**kwargs)


def test_review_requires_human_attestation_and_validated_dimensions(tmp_path: Path):
    store = QualityFeedbackStore(tmp_path / "quality.sqlite3")
    with pytest.raises(PermissionError, match="human"):
        _record(store, human_attested=False)
    for bad in (
        {"safety": 5},
        {"evidence": 2.5},
        {"correctness": True},
    ):
        with pytest.raises(ValueError, match="integer between"):
            _record(store, scores=_scores(**bad))
    with pytest.raises(ValueError, match="exactly"):
        _record(store, scores={"correctness": 4})
    with pytest.raises(ValueError, match="evidence_ref"):
        _record(store, evidence_ref="private personal transcript here")
    with pytest.raises(ValueError, match="note"):
        _record(store, review_note="too short")
    with pytest.raises(ValueError, match="production task"):
        _record(store, task_id="lp-paired-debug-001")
    with pytest.raises(ValueError, match="production task"):
        _record(store, task_id="eval/sanity-002")
    assert store.recent() == []


def test_approval_is_separate_eligibility_gated_and_revocable(tmp_path):
    store = QualityFeedbackStore(tmp_path / "quality.sqlite3")
    good = _record(store)
    assert good.mean_score == 4.0
    assert good.eligible_for_coaching
    with pytest.raises(PermissionError):
        store.approve_coaching(good.id, lesson="Verify the original symptom after each fix.")
    event = store.approve_coaching(
        good.id,
        lesson="Verify the original symptom after a narrow, reversible repair.",
        human_attested=True,
    )
    assert event > good.id
    assert store.approved_lessons() == [
        ("Causal diagnosis", "Verify the original symptom after a narrow, reversible repair.")
    ]
    with pytest.raises(PermissionError):
        store.revoke_coaching(good.id, reason="Owner says lesson is stale.")
    store.revoke_coaching(good.id, reason="Owner says lesson is stale.", human_attested=True)
    assert store.approved_lessons() == []
    assert store.get(good.id) == good
    # Revocation does not erase the evidence trail. A later reapproval requires
    # explicit human attestation and leaves its own new event.
    store.approve_coaching(
        good.id,
        lesson="Verify fault resolution with a repeatable workload before claiming success.",
        human_attested=True,
    )
    assert len(store.approved_lessons()) == 1
    with pytest.raises(ValueError, match="already has a rating"):
        _record(store)


def test_mediocre_or_unsafe_results_cannot_earn_positive_coaching(tmp_path):
    store = QualityFeedbackStore(tmp_path / "quality.sqlite3")
    lower = _record(store, task_id="owner-repair-low", scores=_scores(safety=1))
    assert not lower.eligible_for_coaching
    with pytest.raises(ValueError, match="strong production"):
        store.approve_coaching(
            lower.id, lesson="Attempt the risky procedure again even after safety failed.",
            human_attested=True,
        )
    failed = _record(store, task_id="owner-faulty-answer", outcome="failed")
    assert not failed.eligible_for_coaching
    assert store.approved_lessons() == []


def test_event_log_is_append_only_and_does_not_store_prompts_or_reasoning(tmp_path):
    store = QualityFeedbackStore(tmp_path / "quality.sqlite3")
    entry = _record(store)
    store.approve_coaching(
        entry.id, lesson="Confirm the before-and-after measurement independently.",
        human_attested=True,
    )
    with sqlite3.connect(store.path) as connection:
        cols = {
            row[1] for row in connection.execute("PRAGMA table_info(feedback_events)").fetchall()
        }
        assert not {"prompt", "messages", "response", "thinking", "chain_of_thought"} & cols
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "UPDATE feedback_events SET task_id='tampered' WHERE id=?", (entry.id,)
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("DELETE FROM feedback_events WHERE id=?", (entry.id,))
    assert store.get(entry.id) == entry


def test_coaching_is_off_by_default_and_requires_explicit_opt_in(tmp_path: Path):
    cfg = Config()
    cfg.systemsense.enabled = False
    cfg.agent.feedback_coaching_enabled = False
    agent = LocalPilotAgent(cfg, tmp_path)
    assert "Owner-approved coaching" not in str(agent.messages)
    feedback_path = tmp_path / cfg.agent.data_dir / "quality-feedback.sqlite3"
    assert not feedback_path.exists()

    store = QualityFeedbackStore(feedback_path)
    entry = _record(store)
    store.approve_coaching(
        entry.id,
        lesson="Gather discriminating evidence before changing a machine setting.",
        human_attested=True,
    )
    cfg.agent.feedback_coaching_enabled = True
    agent = LocalPilotAgent(cfg, tmp_path)
    joined = str(agent.messages)
    assert "Owner-approved coaching" in joined
    assert "Gather discriminating evidence" in joined
    assert "not permission to call tools" not in joined.lower()  # wording remains explicit
    assert "Never treat any score or coaching note as permission" in joined
    assert agent.audit.latest("quality_feedback_coaching_loaded")["count"] == 1
    assert agent.tools  # existing guarded tool registry is unmodified
    assert agent.config.selfdev.auto_promote is False


def test_evaluation_isolation_unconditionally_disables_coaching(tmp_path):
    from training.scripts.run_eval_v1 import isolated_config

    (tmp_path / "localpilot.toml").write_text(
        "[agent]\nfeedback_coaching_enabled = true\n", encoding="utf-8"
    )
    cfg = isolated_config(tmp_path, model="nestra:20b-p1")
    assert cfg.agent.feedback_coaching_enabled is False
    assert cfg.library.enabled is False
    assert cfg.selfdev.auto_promote is False


def test_cli_supports_manual_attestation_without_auto_reward():
    args = build_parser().parse_args([
        "feedback", "record", "--task-id", "owner-repair-002",
        "--topic", "Diagnosis", "--outcome", "verified_success",
        "--evidence-ref", "sha256:abc123456",
        "--note", "Human confirmed original symptom no longer occurs.",
        "--correctness", "4", "--evidence", "4", "--safety", "4",
        "--efficiency", "3", "--initiative", "3", "--attest",
    ])
    assert args.command == "feedback" and args.feedback_action == "record"
    assert args.attest and args.safety == 4
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "feedback", "record", "--task-id", "p1", "--topic", "Diagnosis",
            "--outcome", "failed", "--evidence-ref", "sha256:abc1234",
            "--note", "Inspected",
            "--correctness", "6", "--evidence", "2", "--safety", "2",
            "--efficiency", "2", "--initiative", "1",
        ])
