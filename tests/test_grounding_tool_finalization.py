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


def test_grounding_fourth_turn_is_reserved_for_json_synthesis():
    original = _grounding_messages(3)
    kwargs = {"messages": original, "tools": ["read_project_file"]}

    _prepare_grounding_finalization(kwargs)

    assert "tools" not in kwargs
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
