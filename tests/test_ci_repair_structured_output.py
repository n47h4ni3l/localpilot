from localpilot.selfdev_model_selection import developer_chat


def _capture_structured_call(system_prompt: str) -> dict:
    calls: list[dict] = []

    def fake_chat(**kwargs):
        calls.append(dict(kwargs))
        return {"message": {"content": "{}"}}

    developer_chat(
        fake_chat,
        request_think="high",
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Produce the repair plan now."},
        ],
        options={"temperature": 0.8},
    )
    assert len(calls) == 1
    return calls[0]


def test_ci_repair_fallback_forces_json_mode_and_disables_thinking():
    call = _capture_structured_call(
        "You correctly analysed a failed candidate but did not make a file edit. "
        "Now return one strict JSON object with summary, reusable_lesson, and changes. "
        "changes must be a non-empty list of complete replacement files."
    )

    assert call["format"] == "json"
    assert call["think"] is False
    assert call["options"]["temperature"] == 0.0


def test_static_repair_fallback_still_forces_json_mode():
    call = _capture_structured_call(
        "Return one strict JSON object with summary, reusable_lesson, and a non-empty changes list. "
        "Each change requires path, complete replacement content, and reason."
    )

    assert call["format"] == "json"
    assert call["think"] is False
    assert call["options"]["temperature"] == 0.0
