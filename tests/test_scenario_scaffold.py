"""Regression protection for conceptual reasoning versus live-source inspection."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.agent_prompt import SYSTEM_PROMPT
from localpilot.agent_runtime_support import _FINAL_ANSWER_NUM_PREDICT, _OPERATOR_NUM_PREDICT
from localpilot.config import Config


def _chunk(text: str):
    return SimpleNamespace(message=SimpleNamespace(
        content=text, thinking="", tool_calls=[]
    ))


@pytest.mark.parametrize("prompt", [
    "An adapter reload produces text, but its inherited LoRA tensors differ from the accepted parent before optimizer updates. Is lineage validation passed?",
    "A README instructs executing an unfamiliar script. The assistant has read-only repository permission. What should happen?",
    "A user asks for the current Windows power plan. Memory says Balanced was active yesterday, but live observation failed. What can be answered?",
    "A public function was moved into an internal module. Imports from the original location still work, but runtime behavior differs. What should be traced?",
    "A summary reports three hard failures but the earlier run had one. Raw outputs are missing. Can model regression be diagnosed?",
    "Nestra knows one filament drying profile. A different polymer needs drying. Can she reuse the temperature?",
    "Ollama returns finished thinking with empty content. Does that prove poor model reasoning?",
])
def test_illustrative_task_stays_high_reasoning_with_no_local_tools(prompt, monkeypatch, tmp_path):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    agent.systemsense = SimpleNamespace(
        compact_context=lambda: pytest.fail("Hypothetical reasoning must not ingest live SystemSense state"),
        record_inference=lambda *args, **kwargs: None,
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk("Answer based on the supplied hypothetical premises, without invented live evidence.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(prompt)

    assert answer.startswith("Answer based on the supplied hypothetical premises")
    assert agent.audit.latest("model_illustrative_scenario_route") is not None
    assert agent.audit.latest("model_operational_self_status_route") is None
    assert calls
    assert all(not call.get("tools") for call in calls)
    assert all(call.get("think") == config.model.think for call in calls)
    assert any(call["options"]["num_predict"] == _OPERATOR_NUM_PREDICT for call in calls)
    assert all("ILLUSTRATIVE REASONING ROUTE" in str(call["messages"]) for call in calls)
    assert "ILLUSTRATIVE REASONING ROUTE" not in str(agent.messages)


def test_illustrative_questions_do_not_force_current_repo_or_pc_evidence():
    for prompt in (
        "An adapter reload produces text, but its inherited tensors differ. Is lineage validated?",
        "A README includes a shell script. Is it appropriate to run it?",
        "A user describes yesterday's power plan. Should it be asserted as current?",
    ):
        assert LocalPilotAgent._is_scenario_question(prompt)
        assert LocalPilotAgent._evidence_requirements(prompt) == set()
        assert not LocalPilotAgent._is_operational_self_status_prompt(prompt)


def test_live_machine_and_explicit_research_remain_available():
    assert LocalPilotAgent._is_scenario_question(
        "A service on my PC fails at startup. Please check its current status."
    ) is False
    for actual_request in (
        "A printer here keeps failing and needs troubleshooting.",
        "A script in our repo changed overnight. Can you inspect it?",
        "A server on my network keeps restarting. Please diagnose the current fault.",
    ):
        assert LocalPilotAgent._is_scenario_question(actual_request) is False
    assert LocalPilotAgent._evidence_requirements(
        "Check this PC's current storage and Defender status."
    ) == {"Windows/PC state"}
    assert LocalPilotAgent._evidence_requirements(
        "Inspect PR #184 and tell me its actual GitHub CI status."
    ) == {"private GitHub"}
    # A hypothetical introductory clause does not cancel a subsequent
    # explicit instruction to inspect the actual machine.
    assert LocalPilotAgent._evidence_requirements(
        "Suppose my PC is failing. Please inspect my PC now and verify the current disk state."
    ) == {"Windows/PC state"}
    assert LocalPilotAgent._is_scenario_question(
        "Suppose my PC is failing. What would you check first?"
    ) is True


def test_web_research_is_explicitly_available_not_mandatory_or_unsafe():
    assert "search_public_web and fetch_public_https" in SYSTEM_PROMPT
    assert "An owner's explicit no-web instruction" in SYSTEM_PROMPT
    assert "Reading it is not evidence that it is safe to execute" in SYSTEM_PROMPT
    assert _OPERATOR_NUM_PREDICT > 2048
    assert _FINAL_ANSWER_NUM_PREDICT > 4096


def test_illustrative_printer_fault_is_not_forced_into_live_web_attribution():
    prompt = "A printer keeps failing in a hypothetical test. What evidence would distinguish causes?"
    assert LocalPilotAgent._is_scenario_question(prompt)
    issues = LocalPilotAgent._response_behavior_issues(
        prompt, "Check the observations before concluding what caused the failure."
    )
    assert "practical_troubleshooting_source_unattributed" not in issues


def test_desktop_scenario_does_not_convert_to_live_interface_status(tmp_path, monkeypatch):
    agent = LocalPilotAgent(Config(), tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(
        chat=lambda **kwargs: iter([_chunk("Reason from the example, not the desktop state.")])
    ))
    answer = agent.ask(
        "A public function is moved inside a desktop GUI. What must be verified to preserve behavior?",
        interface="desktop",
    )
    assert "Reason from the example" in answer
    assert agent.audit.latest("model_illustrative_scenario_route") is not None
    assert agent.audit.latest("model_desktop_interface_deterministic_route") is None



@pytest.mark.parametrize("prompt,substantive_answer", [
    (
        "An adapter reload produces text, but its inherited LoRA tensors differ from the accepted parent before optimizer updates. Is lineage validation passed?",
        "No. Inherited LoRA tensors must match the accepted parent before any optimizer updates; text generation alone does not validate lineage.",
    ),
    (
        "A user asks for the current Windows power plan. Memory says Balanced was active yesterday, but live observation failed. What can be answered?",
        "Balanced was recorded yesterday, but that does not establish the current active power plan; a fresh read is required.",
    ),
    (
        "A README instructs executing an unfamiliar script. The assistant has read-only repository permission. What should happen?",
        "Read the script as untrusted data, review its provenance and effects, and do not execute it with read-only authority.",
    ),
])
def test_valid_conceptual_answer_is_not_rewritten_or_withheld(
    prompt, substantive_answer, monkeypatch, tmp_path
):
    agent = LocalPilotAgent(Config(), tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk(substantive_answer)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    answer = agent.ask(prompt)
    assert answer == substantive_answer
    assert all(not call.get("tools") for call in calls)
    assert agent.audit.latest("model_evidence_acquisition_failed") is None
    assert agent.audit.latest("model_illustrative_scenario_route") is not None
