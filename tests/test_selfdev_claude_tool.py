from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from localpilot.agent_tools import _tool_result_success
from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.selfdev import CandidateTools
from localpilot.selfdev_claude_tool import candidate_claude_code_tools


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=Test", "-c", "user.email=test@invalid", "commit", "-qm", "base")):
        subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)
    config = Config()
    config.agent.data_dir = "data"
    config.model.name = config.selfdev.developer_model = config.selfdev.implementation_model = "nestra:p1.2"
    config.selfdev.implementation_backend = "local_tools"
    developer = SelfDeveloper(config, root)
    monkeypatch.setattr(developer, "_check_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(developer.governor, "prepare_ollama_handoff", lambda *_args: {"attempted": False, "success": False})
    tools = CandidateTools(workspace)
    task = {"id": "tool", "title": "Improve module", "acceptance": ["correctness"],
            "evaluation": {"metric": "tests", "baseline": "old", "measurement_method": "CI"}}
    kwargs = dict(chat=lambda **_kwargs: None, model="nestra:p1.2", tools=tools,
                  task=task, branch="localpilot/candidate-tool", cycle_id=1, force=False,
                  grounding_plan={"referenced_paths": ["module.py"], "new_runtime_paths": ["new.py"]})
    return developer, tools, kwargs


def test_native_tool_delegates_verified_scope_and_keeps_original_contract(tmp_path, monkeypatch):
    developer, tools, kwargs = _fixture(tmp_path, monkeypatch)
    seen = {}

    def invoke(**request):
        seen.update(request)
        (tools.workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        return json.dumps({"summary": "reviewed proposal", "evaluation_evidence": {"result": "pending_ci"}})

    monkeypatch.setattr(developer, "_run_claude_code_implementation", invoke)
    fn, = candidate_claude_code_tools(developer, **kwargs)
    assert fn.__name__ == "claude_code"
    result = fn("Improve the calculation", ["module.py"])
    assert "reviewed proposal" in result
    assert seen["task"] is kwargs["task"]
    assert seen["grounding_plan"] == {"referenced_paths": ["module.py"], "new_runtime_paths": []}
    assert (tools.workspace / "module.py") in tools.files_written
    assert not (developer.root / "module.py").exists()


@pytest.mark.parametrize("paths", [["other.py"], ["../escape.py"], ["module.py", "other.py"], []])
def test_native_tool_never_broadens_grounded_scope(tmp_path, monkeypatch, paths):
    developer, _tools, kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(developer, "_run_claude_code_implementation", lambda **_kwargs: pytest.fail("must not launch"))
    fn, = candidate_claude_code_tools(developer, **kwargs)
    with pytest.raises(PermissionError):
        fn("Do work", paths)


def test_native_tool_preserves_partial_failure_and_blocks_bad_write(tmp_path, monkeypatch):
    developer, tools, kwargs = _fixture(tmp_path, monkeypatch)

    def invoke(**_request):
        (tools.workspace / "other.py").write_text("OTHER = 2\n", encoding="utf-8")
        raise RuntimeError("stopped CLI")

    monkeypatch.setattr(developer, "_run_claude_code_implementation", invoke)
    fn, = candidate_claude_code_tools(developer, **kwargs)
    with pytest.raises(PermissionError, match="outside"):
        fn("Do work")
    assert tools.failed_write_attempts
    assert (tools.workspace / "other.py").read_text() == "OTHER = 2\n"


def test_native_tool_checks_model_and_resources_before_launch(tmp_path, monkeypatch):
    developer, _tools, kwargs = _fixture(tmp_path, monkeypatch)
    fn, = candidate_claude_code_tools(developer, **kwargs)
    developer.config.selfdev.implementation_model = "wrong:model"
    with pytest.raises(ValueError, match="same accepted"):
        fn("Do work")
    developer.config.selfdev.implementation_model = "nestra:p1.2"
    monkeypatch.setattr(developer, "_check_resources", lambda *_args: (_ for _ in ()).throw(RuntimeError("memory pressure")))
    with pytest.raises(RuntimeError, match="memory pressure"):
        fn("Do work")


def test_native_handoff_is_serialized_and_resource_gate_still_precedes_backend(tmp_path, monkeypatch):
    developer, _tools, kwargs = _fixture(tmp_path, monkeypatch)
    order = []

    def reclaim(*_args):
        assert (developer.data_dir / "claude-code/active.json").exists()
        order.append("reclaim")
        return {"attempted": True, "success": False, "reason": "access denied"}

    def blocked(*_args):
        order.append("guard")
        raise RuntimeError("memory pressure")

    monkeypatch.setattr(developer.governor, "prepare_ollama_handoff", reclaim)
    monkeypatch.setattr(developer, "_check_resources", blocked)
    monkeypatch.setattr(developer, "_run_claude_code_implementation", lambda **_kwargs: pytest.fail("must not load model"))
    fn, = candidate_claude_code_tools(developer, **kwargs)
    with pytest.raises(RuntimeError, match="memory pressure"):
        fn("Do work")
    assert order == ["reclaim", "guard"]


def test_valid_partial_repair_is_registered_even_if_cli_stops(tmp_path, monkeypatch):
    developer, tools, kwargs = _fixture(tmp_path, monkeypatch)
    path = tools.workspace / "module.py"
    tools.files_written.add(path)

    def invoke(**_request):
        path.write_text("VALUE = 2000\n", encoding="utf-8")
        raise RuntimeError("CLI stopped after a useful edit")

    monkeypatch.setattr(developer, "_run_claude_code_implementation", invoke)
    fn, = candidate_claude_code_tools(developer, **kwargs)
    with pytest.raises(RuntimeError, match="CLI stopped"):
        fn("Repair module.py", ["module.py"])
    assert tools.write_count == 1 and path in tools.files_written
    assert not tools.failed_write_attempts


def test_repair_tool_only_receives_existing_candidate_owned_paths(tmp_path, monkeypatch):
    developer, tools, kwargs = _fixture(tmp_path, monkeypatch)
    kwargs["grounding_plan"] = None
    monkeypatch.setattr(developer, "_claude_repair_scope", lambda *_args: ("module.py", "other.py"))
    tools.protected_paths.add("other.py")
    captured = {}
    monkeypatch.setattr(developer, "_run_claude_code_implementation", lambda **request: captured.update(request) or "reviewed")
    fn, = candidate_claude_code_tools(developer, **kwargs)
    assert fn("Fix recorded CI failure") == "reviewed"
    assert captured["grounding_plan"]["referenced_paths"] == ["module.py"]


def test_native_tool_not_offered_under_fixed_backend_or_empty_scope(tmp_path, monkeypatch):
    developer, _tools, kwargs = _fixture(tmp_path, monkeypatch)
    developer.config.selfdev.implementation_backend = "claude_code"
    assert not candidate_claude_code_tools(developer, **kwargs)
    developer.config.selfdev.implementation_backend = "local_tools"
    kwargs["grounding_plan"] = {}
    assert not candidate_claude_code_tools(developer, **kwargs)


def test_native_model_can_choose_tool_and_continue_own_turn(tmp_path, monkeypatch):
    developer, tools, kwargs = _fixture(tmp_path, monkeypatch)
    calls = []

    def invoke(**request):
        calls.append(request)
        (tools.workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        return '{"summary":"Candidate reviewed; CI pending"}'

    monkeypatch.setattr(developer, "_run_claude_code_implementation", invoke)
    responses = iter([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "claude_code", "arguments": {"task": "Improve the calculation", "allowed_paths": ["module.py"]}}}]},
        {"role": "assistant", "content": "Nestra reviewed the tool result; CI is pending."},
    ])
    monkeypatch.setattr(developer, "_developer_chat", lambda *_args, **_kwargs: next(responses))
    messages = []
    result = developer._tool_stage(
        chat=kwargs["chat"], model=kwargs["model"], messages=messages,
        functions=candidate_claude_code_tools(developer, **kwargs), rounds=2,
        force=False, branch=kwargs["branch"], stage="implementation",
    )
    assert "Nestra reviewed" in result and len(calls) == 1
    assert any(isinstance(row, dict) and row.get("role") == "tool" and row.get("tool_name") == "claude_code" for row in messages)
    assert tools.workspace / "module.py" in tools.files_written


def test_optional_scope_failure_preserves_native_tools_and_protected_references(tmp_path, monkeypatch):
    developer, _tools, kwargs = _fixture(tmp_path, monkeypatch)
    kwargs["grounding_plan"]["referenced_paths"].append(".github/workflows/tests.yml")
    fn, = candidate_claude_code_tools(developer, **kwargs)
    with pytest.raises(PermissionError):
        fn("Change workflow", [".github/workflows/tests.yml"])
    kwargs["grounding_plan"] = None
    monkeypatch.setattr(developer, "_claude_repair_scope", lambda *_args: (_ for _ in ()).throw(RuntimeError("missing base")))
    assert candidate_claude_code_tools(developer, **kwargs) == []


@pytest.mark.parametrize("status,ok", [("completed", True), ("cli_error", False), ("resource_pressure", False), ("malformed_output", False)])
def test_claude_tool_failures_are_not_reported_as_successful_ui_steps(status, ok):
    result = {"backend": "claude_code", "status": status, "summary": "evidence"}
    assert _tool_result_success(result) is ok
    assert _tool_result_success(json.dumps(result)) is ok
