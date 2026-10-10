"""Protocol recovery must preserve guide answers, including at the tool ceiling."""
import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config
from localpilot.safety import RiskLevel, ToolSpec


def chunk(*, content="", thinking="", tool_calls=(), reason="stop", tokens=100):
    return {
        "message": {"content": content, "thinking": thinking, "tool_calls": list(tool_calls)},
        "done": True, "done_reason": reason, "prompt_eval_count": 1000,
        "eval_count": tokens,
    }


def agent_at(tmp_path):
    config = Config()
    config.agent.scaffold_mode = "guide_first"
    config.agent.research_soft_tool_rounds = config.agent.research_hard_tool_rounds = 1
    config.agent.feedback_coaching_enabled = config.agent.feedback_auto_observations_enabled = False
    config.systemsense.enabled = config.library.enabled = config.mentor.enabled = False
    config.model.memory_embeddings_enabled = False
    return LocalPilotAgent(config, tmp_path)


@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.parametrize("draft", [
    "  Hello! How can I help?  \n",
    "DECLINE: I choose not to answer.\n",
    "  My unsupported claim remains my own answer.\n",
])
def test_high_reasoning_empty_final_continuation_preserves_exact_guide_draft(
    tmp_path, monkeypatch, after_tool, draft,
):
    agent = agent_at(tmp_path)
    executions, requests, snapshots = [], [], []

    def inspect_record():
        executions.append(True)
        return "A bounded source record."

    agent.tools = {"inspect_record": ToolSpec(
        "inspect_record", "Read a record", RiskLevel.READ_ONLY, inspect_record,
    )}
    streams = []
    if after_tool:
        streams.append(chunk(tool_calls=[{"function": {
            "name": "inspect_record", "arguments": {},
        }}]))
    streams.extend([
        chunk(thinking="operator reasoning", reason="length", tokens=3072),
        chunk(thinking="private final reasoning", reason="length", tokens=6144),
        chunk(content=draft),
    ])
    responses = iter(streams)

    def chat(**request):
        requests.append(request)
        snapshots.append([dict(m) for m in request["messages"]])
        return iter([next(responses)])

    def forbidden_review(*args, **kwargs):
        pytest.fail("Guide recovery examined the model's answer")

    monkeypatch.setattr(agent.information_authority, "review", forbidden_review)
    monkeypatch.setattr(agent.turn_evidence, "review", forbidden_review)
    monkeypatch.setattr(agent, "_response_behavior_issues", forbidden_review)
    original_reset_classifier = agent._looks_like_generic_reset

    def reset_classifier(text):
        assert text != draft, "Guide continuation text was classified before delivery"
        return original_reset_classifier(text)

    monkeypatch.setattr(agent, "_looks_like_generic_reset", reset_classifier)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    assert agent.ask("Analyze this supplied record; a source read is optional.") == draft
    assert len(requests) == 3 + int(after_tool)
    assert len(executions) == int(after_tool)
    assert requests[-2]["think"] == "high"
    assert requests[-1]["think"] is False  # Existing single rendering continuation.
    assert requests[-1]["options"]["num_predict"] == 2048
    assert "tools" not in requests[-2] and "tools" not in requests[-1]
    assert "private final reasoning" in str(snapshots[-1])
    assert "private final reasoning" not in str(agent.messages)
    assert agent.messages[-1] == {"role": "assistant", "content": draft}
    assert agent.audit.latest("model_guide_first_answer_delivered")["draft_preserved_byte_for_byte"]
    assert "private final reasoning" not in agent.audit.path.read_text(encoding="utf-8")


def test_guide_second_reasoning_limit_stays_bounded_and_reports_no_final(tmp_path, monkeypatch):
    agent = agent_at(tmp_path)
    responses = iter([
        chunk(thinking="operator reasoning", reason="length", tokens=3072),
        chunk(thinking="private final reasoning", reason="length", tokens=6144),
        chunk(thinking="continuation reasoning", reason="length", tokens=2048),
    ])
    requests = []

    def chat(**request):
        requests.append(request)
        return iter([next(responses)])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    answer = agent.ask("Reason about this supplied record.")
    assert "single bounded same-context answer continuation" in answer
    assert len(requests) == 3
    assert "private final reasoning" not in str(agent.messages)
    assert "continuation reasoning" not in str(agent.messages)
