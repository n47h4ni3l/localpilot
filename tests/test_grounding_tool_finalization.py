from __future__ import annotations

from localpilot.selfdev_model_selection import (
    _prepare_grounding_finalization,
    developer_chat,
)


def _grounding_messages(tool_turns: int) -> list[dict]:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are LocalPilot's pre-implementation repository-grounding planner. "
                "Return one strict JSON object with a change_plan object."
            ),
        },
        {"role": "user", "content": "Produce the grounded repository change-plan manifest now."},
    ]
    for index in range(tool_turns):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_project_file",
                            "arguments": {"relative_path": f"module_{index}.py"},
                        }
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_name": "read_project_file",
                "content": f"evidence {index}",
            }
        )
    return messages


def test_grounding_fourth_turn_is_reserved_for_schema_bound_json_synthesis():
    original = _grounding_messages(3)
    kwargs = {
        "messages": original,
        "tools": ["read_project_file"],
        "options": {"temperature": 0.4},
        "think": "medium",
    }

    _prepare_grounding_finalization(kwargs)

    assert "tools" not in kwargs
    assert isinstance(kwargs["format"], dict)
    assert kwargs["format"]["type"] == "object"
    change_plan = kwargs["format"]["properties"]["change_plan"]
    assert set(change_plan["required"]) == {
        "referenced_symbols",
        "referenced_config_fields",
        "referenced_paths",
        "required_test_contracts",
        "integration_points",
        "expected_call_relationships",
        "planned_subsystems",
        "new_runtime_paths",
    }
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["think"] is False
    assert kwargs["messages"] is not original
    assert len(original) + 1 == len(kwargs["messages"])
    final = kwargs["messages"][-1]
    assert final["role"] == "user"
    assert "Do not call any more tools" in final["content"]
    assert "strict JSON object" in final["content"]


def test_grounding_keeps_tools_during_first_three_inspection_turns():
    kwargs = {"messages": _grounding_messages(2), "tools": ["read_project_file"]}

    _prepare_grounding_finalization(kwargs)

    assert kwargs["tools"] == ["read_project_file"]
    assert "format" not in kwargs


def test_static_repair_fallback_forces_json_without_tool_mutation():
    messages = [
        {
            "role": "system",
            "content": (
                "Return one strict JSON object with summary, reusable_lesson, and a non-empty changes list. "
                "Each change requires path, complete replacement content, and reason."
            ),
        },
        {"role": "user", "content": "Produce the concrete static-repair plan now."},
    ]
    kwargs = {
        "messages": messages,
        "options": {"temperature": 0.7},
        "think": "medium",
    }

    _prepare_grounding_finalization(kwargs)

    assert kwargs["messages"] is messages
    assert kwargs["format"] == "json"
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["think"] is False
    assert "tools" not in kwargs


def test_non_grounding_tool_loops_are_unchanged():
    messages = [
        {"role": "system", "content": "You are a research-stage developer."},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_project_file", "arguments": {}}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_project_file", "arguments": {}}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_project_file", "arguments": {}}}]},
    ]
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": "done"}}

    developer_chat(
        fake_chat,
        request_think=False,
        model="gpt-oss:20b",
        messages=messages,
        tools=["read_project_file"],
    )

    assert captured["tools"] == ["read_project_file"]
    assert "format" not in captured