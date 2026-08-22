from pathlib import Path

import pytest

from localpilot.selfdev import (
    CandidateTools,
    apply_change_plan,
    choose_next_task,
    parse_change_plan,
    select_resource_aware_developer_model,
    select_developer_model,
)


def test_qwen_developer_model_is_used_only_when_available():
    assert select_developer_model("qwen2.5:32b", "gpt-oss:20b", {"qwen2.5:32b"}) == "qwen2.5:32b"
    assert select_developer_model("qwen2.5:32b", "gpt-oss:20b", {"other"}) == "gpt-oss:20b"


def test_model_selection_preserves_background_memory_ceiling():
    gib = 1024**3
    selection = select_resource_aware_developer_model(
        "qwen2.5:32b",
        "gpt-oss:20b",
        ["qwen2.5:14b"],
        {
            "qwen2.5:32b": 19 * gib,
            "gpt-oss:20b": 13 * gib,
            "qwen2.5:14b": 9 * gib,
        },
        total_memory_bytes=32 * gib,
        available_memory_bytes=20 * gib,
        max_memory_percent=82,
        overhead_bytes=1 * gib,
    )
    assert selection.model == "gpt-oss:20b"
    assert selection.projected_memory_percent == pytest.approx(81.25)
    assert "qwen2.5:32b would project memory" in selection.reason


def test_model_selection_defers_when_no_configured_model_fits():
    gib = 1024**3
    selection = select_resource_aware_developer_model(
        "large",
        "medium",
        [],
        {"large": 19 * gib, "medium": 13 * gib},
        total_memory_bytes=32 * gib,
        available_memory_bytes=13 * gib,
        max_memory_percent=82,
        overhead_bytes=1 * gib,
    )
    assert selection.model is None
    assert "No installed developer model fits" in selection.reason


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


def test_backlog_advances_only_after_validated_merge_or_terminal_rejection():
    tasks = [{"id": "first", "status": "todo"}, {"id": "second", "status": "todo"}]
    assert choose_next_task(tasks, set())["id"] == "first"
    assert choose_next_task(tasks, set(), {"first"}) is None
    assert choose_next_task(tasks, {"first"})["id"] == "second"
    assert choose_next_task(tasks, set(), set(), {"first"})["id"] == "second"
    assert choose_next_task(tasks, set(), set(), {"first", "second"}) is None

