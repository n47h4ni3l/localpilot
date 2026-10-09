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



def _create_candidate(memory, *, task_id="owner-fix-101", checks_passed=True,
                      validation="passed", merged=True, url="https://github.com/n47h4ni3l/localpilot/pull/123"):
    cycle = memory.start_cycle(
        task_id=task_id,
        branch="localpilot/candidate-" + task_id,
        everyday_model="nestra:20b-p1",
        developer_model="nestra:20b-p1",
    )
    memory.finish_cycle(
        cycle, status="candidate_pending_validation", summary="Candidate submitted.",
        reusable_lesson="", checks_passed=checks_passed, pushed=True,
    )
    memory.update_candidate_review(
        cycle, validation_state=validation, merged=merged, pull_request_url=url,
    )
    return cycle


def test_automatic_observations_are_idempotent_and_not_reward_scores(tmp_path):
    from localpilot.learning import LearningMemory

    learning = LearningMemory(tmp_path / "learning.sqlite3")
    verified = _create_candidate(learning)
    # Three weak signals must not turn into a verified outcome.
    _create_candidate(learning, task_id="owner-fix-102", checks_passed=False)
    _create_candidate(learning, task_id="owner-fix-103", merged=False)
    _create_candidate(learning, task_id="lp-paired-debug-001")
    _create_candidate(learning, task_id="owner-fix-104", url="https://example.org/unverified")
    record = QualityFeedbackStore(tmp_path / "quality.sqlite3")
    assert record.sync_verified_development_outcomes(learning.path) == 1
    assert record.sync_verified_development_outcomes(learning.path) == 0
    observed = record.recent_observations()
    assert len(observed) == 1
    assert observed[0]["source_id"] == verified
    assert observed[0]["task_id"] == "owner-fix-101"
    assert observed[0]["evidence_type"] == "ci_passed_and_pr_merged"
    assert observed[0]["scope"] == "delivery_only_not_quality_score"
    assert record.recent() == []
    assert record.approved_lessons() == []
    with sqlite3.connect(record.path) as db:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("DELETE FROM feedback_observations")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("UPDATE feedback_observations SET task_id='changed'")


def test_agent_auto_collects_delivery_evidence_without_coaching(tmp_path):
    from localpilot.learning import LearningMemory

    cfg = Config()
    cfg.systemsense.enabled = False
    cfg.agent.feedback_coaching_enabled = False
    learning_path = tmp_path / cfg.agent.data_dir / cfg.selfdev.learning_database
    memory = LearningMemory(learning_path)
    _create_candidate(memory)
    agent = LocalPilotAgent(cfg, tmp_path)
    store = QualityFeedbackStore(tmp_path / cfg.agent.data_dir / "quality-feedback.sqlite3")
    assert len(store.recent_observations()) == 1
    assert store.approved_lessons() == []
    assert "Owner-approved coaching" not in str(agent.messages)
    assert agent.audit.latest("quality_feedback_objective_evidence_collected")["scoring_performed"] is False
    LocalPilotAgent(cfg, tmp_path)  # a repeated startup cannot duplicate the event
    assert len(store.recent_observations()) == 1


def test_auto_observations_respect_configuration_and_benchmark_isolation(tmp_path):
    from localpilot.learning import LearningMemory
    from training.scripts.run_eval_v1 import isolated_config

    cfg = Config()
    cfg.systemsense.enabled = False
    cfg.agent.feedback_auto_observations_enabled = False
    memory_path = tmp_path / cfg.agent.data_dir / cfg.selfdev.learning_database
    learning = LearningMemory(memory_path)
    _create_candidate(learning)
    LocalPilotAgent(cfg, tmp_path)
    assert not (tmp_path / cfg.agent.data_dir / "quality-feedback.sqlite3").exists()

    (tmp_path / "localpilot.toml").write_text(
        "[agent]\nfeedback_auto_observations_enabled = true\nfeedback_coaching_enabled = true\n",
        encoding="utf-8",
    )
    isolated = isolated_config(tmp_path, model="nestra:20b-p1")
    assert isolated.agent.feedback_auto_observations_enabled is False
    assert isolated.agent.feedback_coaching_enabled is False


def test_cli_observation_commands_are_read_only_to_scorecards():
    parsed = build_parser().parse_args(["feedback", "observations"])
    assert parsed.command == "feedback"
    assert parsed.feedback_action == "observations"
    parsed = build_parser().parse_args(["feedback", "sync"])
    assert parsed.feedback_action == "sync"
