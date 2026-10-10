"""A/B regression coverage for Nestra guide-first assistance.

These tests use a mocked final draft, not a claim that a local P1 inference
comparison has been run. Tool authorisation is never changed by scaffold_mode.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.agent_tools import _tool_result_success
from localpilot.config import Config, load_config
from localpilot.guide_scaffold import GUIDE_FIRST_INSTRUCTIONS


@pytest.mark.parametrize("value", ["guide_first", "strict"])
def test_scaffold_modes_can_be_selected_from_config(value, tmp_path: Path):
    config_path = tmp_path / "localpilot.toml"
    config_path.write_text(f'[agent]\nscaffold_mode = "{value}"\n', encoding="utf-8")
    assert load_config(config_path).agent.scaffold_mode == value


def test_unknown_scaffold_mode_is_rejected(tmp_path: Path):
    config_path = tmp_path / "localpilot.toml"
    config_path.write_text('[agent]\nscaffold_mode = "unrestricted"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="scaffold_mode"):
        load_config(config_path)


def test_historical_evaluators_remain_strict(tmp_path: Path):
    from training.scripts.run_eval_v1 import isolated_config

    (tmp_path / "localpilot.toml").write_text(
        '[agent]\nscaffold_mode = "guide_first"\n', encoding="utf-8"
    )
    config = isolated_config(tmp_path, model="nestra:20b-p1")
    assert config.agent.scaffold_mode == "strict"


def _agent(tmp_path: Path, mode: str = "guide_first") -> LocalPilotAgent:
    config = Config()
    config.agent.scaffold_mode = mode
    config.systemsense.enabled = False
    return LocalPilotAgent(config, tmp_path)


def test_guide_preserves_good_first_draft_without_correction_or_withholding(tmp_path):
    agent = _agent(tmp_path)
    draft = "Monday's green build is historical evidence; it does not prove main is green today."
    delivered = agent._continue_high_reasoning_answer(
        None, prompt="A CI build was green on Monday. Is main green right now?",
        round_no=1, after_tools=False, draft_content=draft,
    )
    assert delivered == draft
    assert "[LocalPilot withheld" not in delivered
    assert agent.audit.latest("model_guide_first_answer_delivered")["answer_review_performed"] is False
    assert "GUIDE-FIRST ASSISTANCE" in str(agent.messages)
    assert "authorization" in GUIDE_FIRST_INSTRUCTIONS.lower()
    assert "When discussing LocalPilot's own implementation" not in str(agent.messages)
    assert "answer validator" not in str(agent.messages).lower().split("guide-first assistance:")[0]


def test_missing_live_evidence_is_scoped_but_draft_is_not_lost(tmp_path):
    agent = _agent(tmp_path)
    draft = "Yesterday's value was Balanced. A new machine read is required for today's value."
    delivered = agent._continue_high_reasoning_answer(
        None,
        prompt="What is the active Windows power plan today?",
        round_no=1,
        after_tools=True,
        draft_content=draft,
        missing_evidence=frozenset({"Windows/PC state"}),
    )
    assert delivered == draft
    assert "Verification note:" not in delivered
    assert "[LocalPilot withheld" not in delivered
    assert agent.audit.latest("model_guide_first_answer_delivered")["answer_review_performed"] is False


def test_sourced_numeric_citation_issues_are_advice_not_automatic_refusal(tmp_path):
    agent = _agent(tmp_path)
    draft = "A repository constant is 500 (localpilot/tools/repository.py:99), but that line is unverified."
    delivered = agent._continue_high_reasoning_answer(
        None,
        prompt="Explain the source code and cite the exact constant.",
        round_no=1, after_tools=True, draft_content=draft,
        recovery_messages=[{
            "role": "tool", "tool_name": "read_repository_file",
            "content": "Repository file: localpilot/tools/repository.py lines 40-40\n"
                       "40: max_results = min(max_results, 100)",
        }],
        successful_tools=frozenset({"read_repository_file"}),
    )
    assert delivered == draft
    assert "Verification note:" not in delivered
    assert agent.audit.latest("model_guide_first_answer_delivered")["draft_preserved_byte_for_byte"] is True


def test_guide_preserves_owner_supplied_hypothetical_without_live_tools(tmp_path):
    agent = _agent(tmp_path)
    draft = "If inherited tensors don't match, lineage validation fails despite fluent generation."
    delivered = agent._continue_high_reasoning_answer(
        None,
        prompt="Suppose inherited tensors differ. Can a fluent response validate lineage?",
        round_no=1, after_tools=False, draft_content=draft,
    )
    assert delivered == draft
    assert "missing evidence" not in delivered.lower()


def test_guide_lacks_any_action_authority_and_respects_no_web(tmp_path):
    strict = _agent(tmp_path / "strict", "strict")
    guide = _agent(tmp_path / "guide", "guide_first")
    assert type(strict.policy) is type(guide.policy)
    assert strict.policy.auto_allow_read_only == guide.policy.auto_allow_read_only
    assert strict.policy.auto_allow_reversible == guide.policy.auto_allow_reversible
    assert strict.policy.require_confirmation_for_destructive == guide.policy.require_confirmation_for_destructive
    assert "search_public_web" in guide._forbidden_tools(
        "Please don't use the internet for this answer."
    )
    assert "fetch_public_https" in guide._forbidden_tools(
        "Please don't use the internet for this answer."
    )


def test_tool_failure_marker_inside_read_source_is_not_tool_error():
    valid_read = (
        "Repository file: localpilot/tools/repository.py lines 1-2\n"
        "1: message = 'No matches found.'\n"
        "2: note = 'Tool error: is a quoted example'"
    )
    assert _tool_result_success(valid_read)
    assert not _tool_result_success("Tool error: permission denied")
    assert not _tool_result_success(
        "Private GitHub pull request (read-only).\nGitHub read failed: authentication required"
    )
    assert not _tool_result_success(
        "Public web search: query\nNo bounded HTTPS results were found."
    )
    assert not _tool_result_success("Repository search: 'missing'\nNo matches found.")
    assert _tool_result_success(
        "Repository search: 'No matches found.'\n"
        "localpilot/test_data.py:22: message = 'No matches found.'"
    )


def test_guide_is_method_coaching_not_a_postprocessor():
    assert "research" in GUIDE_FIRST_INSTRUCTIONS.lower()
    assert "owner's authorization" in GUIDE_FIRST_INSTRUCTIONS
    from localpilot import guide_scaffold
    assert not hasattr(guide_scaffold, "guide_annotate_answer")


def test_guide_exact_draft_preservation_even_with_unsupported_fact(tmp_path):
    agent = _agent(tmp_path)
    draft = "  I think LIMIT = 900 (not source-verified).  \n"
    delivered = agent._continue_high_reasoning_answer(
        None, prompt="Inspect the repository and explain LIMIT.",
        round_no=1, after_tools=True,
        draft_content=draft,
        missing_evidence=frozenset({"trusted repository"}),
    )
    assert delivered == draft
    assert agent.audit.latest("model_guide_first_answer_delivered")["answer_review_performed"] is False
