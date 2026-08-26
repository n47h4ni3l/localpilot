from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.authority import InformationAuthorityVerifier, TurnEvidenceVerifier
from localpilot.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _chunk(content: str):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking="", tool_calls=[])
    )


def _call(name: str, arguments: dict[str, object]):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


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


def test_postvalidation_preserves_a_grounded_draft_without_a_rewrite(tmp_path: Path):
    agent = LocalPilotAgent(Config(), tmp_path)
    streams = []
    draft = (
        "I think the most useful next step is to inspect the contradiction directly, "
        "because another option menu would only hand the decision back to you."
    )

    def fake_chat(**kwargs):
        streams.append(kwargs)
        return iter([_chunk(draft)])

    answer = agent._continue_high_reasoning_answer(
        fake_chat,
        prompt="What do you think is the most useful thing to do next, and why?",
        round_no=1,
        after_tools=False,
        authority_review=True,
    )

    assert answer == draft
    assert len(streams) == 1
    event = agent.audit.latest("model_same_context_postvalidation_complete")
    assert event["accepted"] is True
    assert event["prose_rewritten"] is False


def test_live_state_postcondition_requires_claim_specific_evidence():
    verifier = TurnEvidenceVerifier()

    rejected = verifier.review(
        "Disk space is healthy, the Balanced power plan is active, and there are no known critical bugs.",
        successful_tools=frozenset({"get_system_summary"}),
    )
    scoped = verifier.review(
        "CPU and memory look normal in the system summary. Disk and power-plan state were not checked, "
        "so I cannot say there are no critical bugs.",
        successful_tools=frozenset({"get_system_summary"}),
    )
    guidance = verifier.review(
        "Open File Explorer to check disk usage, then open Power Options to inspect the active power plan."
    )

    assert {issue.code for issue in rejected.issues} == {
        "storage_state_without_storage_evidence",
        "power_plan_without_power_evidence",
        "unsupported_blanket_health_claim",
    }
    assert scoped.accepted is True
    assert guidance.accepted is True


def test_quick_health_check_corrects_only_claims_missing_specific_observations(
    tmp_path: Path, monkeypatch
):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    streams = iter(
        [
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        thinking="",
                        tool_calls=[_call("get_system_summary", {})],
                    )
                )
            ],
            [_chunk("The system summary is ready.")],
            [
                _chunk(
                    "Memory use looks normal in the system summary. Disk space is healthy, "
                    "the Balanced power plan is active, and there are no known critical bugs."
                )
            ],
            [
                _chunk(
                    "Memory use looks normal in the system summary. Disk usage and the active "
                    "power plan were not checked, and I cannot say there are no critical bugs."
                )
            ],
        ]
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask("Give me a quick health check.")

    assert "Memory use looks normal" in answer
    assert "were not checked" in answer
    correction = agent.audit.latest("model_same_context_authority_correction_complete")
    assert correction["original_risks"] == [
        "storage_state_without_storage_evidence",
        "power_plan_without_power_evidence",
        "unsupported_blanket_health_claim",
    ]
    assert correction["accepted"] is True
    assert calls[-2]["think"] == "high"
    assert calls[-1]["think"] == "low"
