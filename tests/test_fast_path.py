from __future__ import annotations

import sys
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config
from localpilot.fast_path import classify_fast_path


class _FastSense:
    def __init__(self):
        self.inference = []

    def compact_context(self):
        raise AssertionError("fast path must not request passive SystemSense context")

    def record_inference(self, runtime, *, model):
        self.inference.append((dict(runtime), model))


def _chunk(content: str):
    return {
        "message": {"content": content, "thinking": "", "tool_calls": []},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 18,
        "total_duration": 1_000_000,
        "prompt_eval_duration": 300_000,
        "eval_duration": 500_000,
    }


def test_router_is_conservative_about_live_safety_and_complex_requests():
    assert classify_fast_path("Good evening :)", interface="desktop").route == "fast_social"
    assert classify_fast_path("Thank you!", interface="desktop").route == "fast_social"
    assert classify_fast_path("What is the square root of 5?", interface="desktop").route == "fast_math"

    for prompt in (
        "What's the weather tomorrow?",
        "My daughter is allergic to kiwis",
        "Why is LocalPilot using so much RAM?",
        "Can you troubleshoot this printer error?",
        "What should I do about this complicated situation?",
        "Search the web for the latest firmware.",
    ):
        assert classify_fast_path(prompt, interface="desktop").route == "full"


def test_math_router_produces_bounded_authoritative_results():
    decision = classify_fast_path("What is the square root of 5?", interface="desktop")
    assert decision.route == "fast_math"
    assert decision.deterministic_fact == "sqrt(5) = 2.2360679775"
    assert decision.expected_numeric == "2.2360679775"

    arithmetic = classify_fast_path("calculate (17 + 3) * 4", interface="desktop")
    assert arithmetic.route == "fast_math"
    assert arithmetic.deterministic_fact == "(17 + 3) * 4 = 80"

    assert classify_fast_path("calculate 2 ** 100", interface="desktop").route == "full"
    assert classify_fast_path("calculate __import__('os').system('x')", interface="desktop").route == "full"


def test_fast_social_is_model_generated_with_small_context_and_no_tools(monkeypatch, tmp_path):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk("Good evening. Nice to hear from you.")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    agent = LocalPilotAgent(Config(), tmp_path)
    sense = _FastSense()
    agent.systemsense = sense

    answer = agent.ask("Good evening :)", interface="desktop")

    assert answer == "Good evening. Nice to hear from you."
    assert len(calls) == 1
    call = calls[0]
    assert call["think"] == "low"
    assert "tools" not in call
    assert call["options"]["num_predict"] == 320
    rendered = "\n".join(str(item.get("content") or "") for item in call["messages"])
    assert "This is not a canned-response lookup" in rendered
    assert "Good evening :)" in rendered
    assert "SYSTEMSENSE" not in rendered
    assert sense.inference
    runtime = sense.inference[-1][0]
    assert runtime["phase"] == "fast_social"
    assert runtime["first_content_ms"] is not None
    assert runtime["wall_time_ms"] >= 0

    audit = agent.audit.latest("model_fast_path_complete")
    assert audit is not None
    assert audit["route"] == "fast_social"
    assert audit["deterministic_fact"] is False


def test_fast_math_uses_deterministic_fact_but_astra_writes_the_reply(monkeypatch, tmp_path):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter([_chunk(r"\[The square root of 5 is approximately 2.2360679775.\]")])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    agent = LocalPilotAgent(Config(), tmp_path)
    agent.systemsense = _FastSense()

    answer = agent.ask("what is the square root of 5?", interface="desktop")

    assert answer == "The square root of 5 is approximately 2.2360679775."
    assert len(calls) == 1
    rendered = "\n".join(str(item.get("content") or "") for item in calls[0]["messages"])
    assert "sqrt(5) = 2.2360679775" in rendered
    assert "authoritative for this turn" in rendered
    assert calls[0]["think"] == "low"

    audit = agent.audit.latest("model_fast_path_complete")
    assert audit is not None
    assert audit["route"] == "fast_math"
    assert audit["deterministic_fact"] is True


def test_fast_path_can_decline_and_escalate_without_polluting_conversation(monkeypatch, tmp_path):
    agent = LocalPilotAgent(Config(), tmp_path)
    agent.systemsense = _FastSense()
    decision = classify_fast_path("Thank you", interface="desktop")
    before = list(agent.messages)

    answer = agent._try_fast_path(
        lambda **kwargs: iter([_chunk("FAST_PATH_ESCALATE")]),
        "Thank you",
        decision,
    )

    assert answer is None
    assert agent.messages == before
    audit = agent.audit.latest("model_fast_path_escalated")
    assert audit is not None
    assert audit["escalation_reason"] == "model_requested_escalation"
