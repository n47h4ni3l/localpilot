from pathlib import Path

import pytest

from localpilot.selfdev import (
    CandidateTools,
    apply_change_plan,
    choose_next_task,
    parse_change_plan,
    select_developer_model,
)


def test_qwen_developer_model_is_used_only_when_available():
    assert select_developer_model("qwen2.5:32b", "gpt-oss:20b", {"qwen2.5:32b"}) == "qwen2.5:32b"
    assert select_developer_model("qwen2.5:32b", "gpt-oss:20b", {"other"}) == "gpt-oss:20b"


def test_structured_fallback_applies_only_through_candidate_tools(tmp_path: Path, monkeypatch):
    raw = """{
      "summary": "add module",
      "reusable_lesson": "write a complete small file",
      "changes": [{"path": "localpilot/new_module.py", "content": "VALUE = 1\\n", "reason": "needed"}]
    }"""
    plan = parse_change_plan(raw)
    tools = CandidateTools(tmp_path)
    calls: list[tuple[str, str]] = []
    original = tools.write_project_file

    def tracked(path: str, content: str) -> str:
        calls.append((path, content))
        return original(path, content)

    monkeypatch.setattr(tools, "write_project_file", tracked)
    apply_change_plan(plan, tools)
    assert calls == [("localpilot/new_module.py", "VALUE = 1\n")]
    assert (tmp_path / "localpilot" / "new_module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_structured_fallback_still_cannot_escape(tmp_path: Path):
    raw = '{"changes":[{"path":"../stable.py","content":"bad","reason":"escape"}]}'
    with pytest.raises(ValueError, match="escapes"):
        apply_change_plan(parse_change_plan(raw), CandidateTools(tmp_path))


def test_backlog_does_not_advance_without_validated_merge():
    tasks = [{"id": "first", "status": "todo"}, {"id": "second", "status": "todo"}]
    assert choose_next_task(tasks, set())["id"] == "first"
    assert choose_next_task(tasks, set(), {"first"}) is None
    assert choose_next_task(tasks, {"first"})["id"] == "second"

