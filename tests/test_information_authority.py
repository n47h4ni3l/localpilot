from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.authority import InformationAuthorityVerifier
from localpilot.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _chunk(content: str):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking="", tool_calls=[])
    )


def test_novel_information_flow_phrasings_are_checked_against_grounded_contracts():
    verifier = InformationAuthorityVerifier(PROJECT_ROOT)

    scenarios = {
        "CI has the final say on whether a candidate ships to stable.": (
            "github_actions_controls_promotion"
        ),
        "/teach copies what the operator noticed into durable memory.": (
            "teach_records_observations"
        ),
        "A single SafetyPolicy surrounds both ordinary tools and candidate editing.": (
            "operator_policy_governs_candidate_tools"
        ),
    }

    for unseen_phrase, expected_code in scenarios.items():
        report = verifier.review(unseen_phrase)
        assert report.accepted is False
        assert expected_code in {issue.code for issue in report.issues}
        assert report.evidence


def test_live_repository_literals_and_direct_calls_use_actual_ground_truth():
    verifier = InformationAuthorityVerifier(PROJECT_ROOT)

    invented = verifier.review(
        "`localpilot/phantom.py` contains `PhantomRouter.route()` and uses "
        "`model.imaginary_switch`."
    )
    grounded = verifier.review(
        "`LocalPilotAgent._learning_context()` calls "
        "`LearningMemory.search_knowledge_facts()`."
    )
    external = verifier.review(
        "Python's external `Path.resolve()` API can normalize a filesystem path."
    )

    assert {issue.code for issue in invented.issues} == {
        "unverified_repository_path",
        "unverified_repository_symbol",
        "unverified_config_field",
    }
    assert grounded.accepted is True
    assert "call_relationship" in grounded.claim_classes
    assert any(item.startswith("call:") for item in grounded.evidence)
    assert external.accepted is True


def test_proposals_negations_and_ordinary_answers_do_not_trigger_repository_scans():
    verifier = InformationAuthorityVerifier(PROJECT_ROOT)

    proposal = verifier.review(
        "We could add `LocalPilotAgent.fabricate_answer()` in a future proposal."
    )
    negation = verifier.review("GitHub Actions does not merge or promote candidates.")
    ordinary = verifier.review("The weather looks pleasant today.")

    assert proposal.accepted is negation.accepted is ordinary.accepted is True
    assert proposal.claim_classes == negation.claim_classes == ordinary.claim_classes == ()
    assert proposal.repository_scan_ms < 20
    assert negation.repository_scan_ms < 20
    assert ordinary.repository_scan_ms < 20


def test_extra_model_review_is_limited_to_localpilot_architecture_requests():
    assert LocalPilotAgent._requires_information_authority_review(
        "Explain the current LocalPilot candidate promotion architecture."
    )
    assert not LocalPilotAgent._requires_information_authority_review(
        "Inspect the repository and report the canonical value in a text file."
    )
    assert not LocalPilotAgent._requires_information_authority_review(
        "Explain memory techniques in Python generally."
    )


def test_repository_claims_fail_safely_when_ground_truth_is_incomplete(tmp_path: Path):
    (tmp_path / "localpilot").mkdir()
    (tmp_path / "localpilot" / "agent.py").write_text(
        "class LocalPilotAgent: pass\n",
        encoding="utf-8",
    )

    report = InformationAuthorityVerifier(tmp_path).review(
        "The operator delegates answers to `LocalPilotAgent.fabricate_answer()`."
    )

    assert report.accepted is False
    assert {issue.code for issue in report.issues} == {
        "authority_ground_truth_unavailable"
    }


def test_real_answer_path_self_corrects_an_unseen_authority_claim(tmp_path: Path):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)
    agent.information_authority = InformationAuthorityVerifier(PROJECT_ROOT)
    streams = iter(
        [
            [_chunk("Initial draft.")],
            [_chunk("CI has the final say on whether a candidate ships to stable.")],
            [_chunk("CI validates the candidate; promotion still requires a human merge.")],
        ]
    )

    def fake_chat(**_kwargs):
        return iter(next(streams))

    answer = agent._continue_high_reasoning_answer(
        fake_chat,
        prompt="Explain the current LocalPilot candidate promotion boundary.",
        round_no=1,
        after_tools=True,
        authority_review=True,
    )

    assert answer == "CI validates the candidate; promotion still requires a human merge."
    event = agent.audit.latest("model_same_context_authority_correction_complete")
    assert event["original_risks"] == ["github_actions_controls_promotion"]
    assert event["remaining_risks"] == []
    crosscheck = agent.audit.latest("model_information_authority_crosscheck")
    assert crosscheck["accepted"] is True
