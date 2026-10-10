"""Recover GPT-OSS transport names without adding answer or tool authority.

All network/model messages are fake. Inferences aren't graded or rewritten.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.agent_tools import (
    _advertised_tool_names, _recover_readonly_tool_call, _tool_call_parts,
)
from localpilot.config import Config
from localpilot.safety import RiskLevel, ToolSpec


def _agent(tmp_path, mode="guide_first"):
    cfg = Config()
    cfg.agent.scaffold_mode = mode
    cfg.systemsense.enabled = False
    return LocalPilotAgent(cfg, tmp_path)


def _call(name, args=None):
    return SimpleNamespace(function=SimpleNamespace(
        name=name, arguments=dict(args or {}),
    ))


def _chunk(content="", tool_calls=None):
    return SimpleNamespace(message=SimpleNamespace(
        content=content, thinking="", tool_calls=list(tool_calls or [])
    ), done=True, done_reason="stop")


@pytest.mark.parametrize(("requested", "expected_reason"), [
    ("read_repository_file<|channel|>commentary", "gpt_oss_channel_suffix"),
    (" read_repository_file ", "surrounding_whitespace"),
    ("READ_REPOSITORY_FILE", "case_only"),
])
def test_repairs_only_unambiguous_advertised_read_only_names(requested, expected_reason):
    args = {"path": "README.md", "start_line": 1, "end_line": 3}
    original = _call(requested, args)
    repaired, reason = _recover_readonly_tool_call(
        original, exposed=frozenset({"read_repository_file"}),
        recoverable=frozenset({"read_repository_file"}),
    )
    assert reason == expected_reason
    assert _tool_call_parts(repaired) == ("read_repository_file", args)
    assert original.function.name == requested
    assert original.function.arguments == args


@pytest.mark.parametrize("requested", [
    "read_repository_file<|channel|>analysis",
    "read_repository_file<|channel|>commentary<|channel|>commentary",
    "read_repository_file<|channel|>commentaryx",
    "read_repository_fil",
    "read_repository_file_more",
    "repo_browser.search",
    "read_repository_file;open_windows_app",
])
def test_does_not_guess_or_interpret_unrecognized_name(requested):
    original = {"function": {"name": requested, "arguments": {"path": "README.md"}}}
    recovered, reason = _recover_readonly_tool_call(
        original, exposed=frozenset({"read_repository_file", "search_repository"}),
        recoverable=frozenset({"read_repository_file", "search_repository"}),
    )
    assert reason is None
    assert recovered is original


def test_observed_repository_namespace_alias_requires_exact_argument_signature():
    allowed = frozenset({"search_repository"})
    original = _call("repo_browser.search", {"query": "candidate", "max_results": 3})
    repaired, reason = _recover_readonly_tool_call(
        original, exposed=allowed, recoverable=allowed,
    )
    assert reason == "known_readonly_alias"
    assert _tool_call_parts(repaired) == (
        "search_repository", {"query": "candidate", "max_results": 3},
    )
    for arguments in ({"pattern": "candidate"}, {"query": "", "path": "."},
                      {"query": "candidate", "command": "delete"}):
        bad = _call("repo_browser.search", arguments)
        unchanged, reason = _recover_readonly_tool_call(
            bad, exposed=allowed, recoverable=allowed,
        )
        assert unchanged is bad and reason is None


def test_explicit_owner_no_web_request_cannot_be_bypassed(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    seen = []
    def fake_chat(**kwargs):
        seen.append(kwargs.get("tools") or [])
        return iter([_chunk(content="I can answer without internet.")])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    assert agent.ask("Explain the concept but do not use the web.") == (
        "I can answer without internet."
    )
    assert seen and not any(
        getattr(fn, "__name__", None) in {"search_public_web", "fetch_public_https"}
        for fn in seen[0]
    )
    assert agent.audit.latest("model_tool_name_transport_recovered") is None


def test_external_mentor_is_never_recovered_even_when_enabled(tmp_path):
    cfg = Config()
    cfg.agent.scaffold_mode = "guide_first"
    cfg.mentor.enabled = True
    cfg.systemsense.enabled = False
    agent = LocalPilotAgent(cfg, tmp_path)
    offered = agent._functions()
    assert "consult_external_mentor" in _advertised_tool_names(offered)
    raw = _call("consult_external_mentor<|channel|>commentary", {
        "question": "What approach should I consider?", "impasse": "I am stuck.",
    })
    result = agent._stream_chat_message(
        lambda **_: iter([_chunk(tool_calls=[raw])]),
        think="high", tools=offered, phase="operator"
    )
    assert _tool_call_parts(result["tool_calls"][0])[0] == raw.function.name
    assert agent.audit.latest("model_tool_name_transport_recovered") is None


def test_missing_or_unadvertised_tools_never_recover():
    original = _call("read_repository_file<|channel|>commentary")
    for exposed, recoverable in [
        (frozenset(), frozenset({"read_repository_file"})),
        (frozenset({"read_repository_file"}), frozenset()),
        (frozenset({"get_repository_status"}), frozenset({"read_repository_file"})),
    ]:
        recovered, reason = _recover_readonly_tool_call(
            original, exposed=exposed, recoverable=recoverable,
        )
        assert recovered is original and reason is None


def test_mutating_or_expensive_mentor_tools_are_never_recovered(tmp_path):
    agent = _agent(tmp_path)
    exposed = _advertised_tool_names(agent._functions())
    assert "read_repository_file" in exposed
    assert "open_windows_app" not in exposed  # By default, not authorized for automatic use.
    raw = _call("open_windows_app<|channel|>commentary", {"app": "notepad"})
    repaired, reason = _recover_readonly_tool_call(
        raw, exposed=frozenset({"open_windows_app"}),
        recoverable=frozenset({"read_repository_file"}),
    )
    assert repaired is raw and reason is None


def test_stream_recovers_name_and_preserves_arguments_audit_and_original(tmp_path):
    agent = _agent(tmp_path)
    request = _call("read_repository_file<|channel|>commentary", {"path": "README.md"})
    def fake_chat(**kwargs):
        assert any(getattr(fn, "__name__", None) == "read_repository_file"
                   for fn in kwargs["tools"])
        yield _chunk(tool_calls=[request])
    result = agent._stream_chat_message(
        fake_chat, think="high", tools=agent._functions(), phase="operator", turn_no=1
    )
    assert _tool_call_parts(result["tool_calls"][0]) == (
        "read_repository_file", {"path": "README.md"},
    )
    assert request.function.name == "read_repository_file<|channel|>commentary"
    event = agent.audit.latest("model_tool_name_transport_recovered")
    assert event["resolved_name"] == "read_repository_file"
    assert event["recovery_reason"] == "gpt_oss_channel_suffix"
    assert "arguments" not in event and "args" not in event


def test_strict_comparison_control_is_untouched(tmp_path):
    agent = _agent(tmp_path, "strict")
    request = _call("read_repository_file<|channel|>commentary", {"path": "README.md"})
    result = agent._stream_chat_message(
        lambda **_: iter([_chunk(tool_calls=[request])]),
        think="high", tools=agent._functions(), phase="operator", turn_no=1
    )
    assert _tool_call_parts(result["tool_calls"][0])[0] == request.function.name
    assert agent.audit.latest("model_tool_name_transport_recovered") is None


def test_model_without_advertised_tools_cannot_recover(tmp_path):
    agent = _agent(tmp_path)
    request = _call("read_repository_file<|channel|>commentary")
    result = agent._stream_chat_message(
        lambda **kwargs: iter([_chunk(tool_calls=[request])]),
        think="high", tools=None, phase="final"
    )
    assert _tool_call_parts(result["tool_calls"][0])[0] == request.function.name
    assert agent.audit.latest("model_tool_name_transport_recovered") is None


def test_dict_tool_call_preserves_other_wire_metadata_and_parameters():
    original = {
        "type": "function", "id": "tool-17",
        "function": {"name": "search_repository<|channel|>commentary",
                     "arguments": {"query": "self-development", "max_results": 2}},
    }
    recovered, reason = _recover_readonly_tool_call(
        original, exposed=frozenset({"search_repository"}),
        recoverable=frozenset({"search_repository"}),
    )
    assert reason == "gpt_oss_channel_suffix"
    assert recovered["id"] == original["id"]
    assert recovered["function"]["arguments"] == original["function"]["arguments"]
    assert original["function"]["name"] != recovered["function"]["name"]


def test_end_to_end_recovered_tool_executes_original_registered_function(
        tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    invocations = []
    def read_repository_file(path: str, start_line: int = 1, end_line: int = 10) -> str:
        invocations.append((path, start_line, end_line))
        return "Repository file: README.md lines 1-1\n1: verified text"
    agent.tools["read_repository_file"] = ToolSpec(
        "read_repository_file", "Safe stub read", RiskLevel.READ_ONLY,
        read_repository_file,
    )
    streams = iter([
        [_chunk(tool_calls=[_call("read_repository_file<|channel|>commentary", {
            "path": "README.md", "start_line": 1, "end_line": 1,
        })])],
        [_chunk(content="I read the registered source and formed my own conclusion.")],
    ])
    offered = []
    def fake_chat(**kwargs):
        offered.append(kwargs.get("tools"))
        return iter(next(streams))
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    reply = agent.ask("Inspect README.md and explain its meaning.")
    assert reply == "I read the registered source and formed my own conclusion."
    assert invocations == [("README.md", 1, 1)]
    assert all(any(getattr(fn, "__name__", "") == "read_repository_file"
                   for fn in tools or []) for tools in offered)
    assert agent.audit.latest("model_tool_name_transport_recovered") is not None
    assert agent.audit.latest("model_guide_first_answer_delivered")[
        "draft_preserved_byte_for_byte"
    ] is True
