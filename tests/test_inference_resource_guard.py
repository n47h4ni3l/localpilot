from types import SimpleNamespace

import pytest

import localpilot.selfdev_model_selection as model_selection
from localpilot.selfdev_results import CyclePaused


_GIB = 1024**3


def _streaming_chat(**_kwargs):
    return iter([{"message": {"content": "ok", "tool_calls": []}}])


def test_admitted_inference_may_cross_background_memory_ceiling(monkeypatch):
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=83.4, available=5 * _GIB),
    )

    def admission_guard():
        raise CyclePaused("memory 83% > 82%")

    response = model_selection.developer_chat(
        _streaming_chat,
        request_think=False,
        stream_guard=admission_guard,
        model="gpt-oss:20b",
        messages=[],
    )

    assert response.message["content"] == "ok"


@pytest.mark.parametrize(
    ("percent", "available_gib"),
    [
        (94.0, 6.0),
        (90.0, 1.5),
    ],
)
def test_inference_still_stops_at_emergency_memory_boundary(
    monkeypatch,
    percent,
    available_gib,
):
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=percent, available=available_gib * _GIB),
    )

    def admission_guard():
        raise CyclePaused("memory 83% > 82%")

    with pytest.raises(CyclePaused, match="memory emergency during inference") as exc_info:
        model_selection.developer_chat(
            _streaming_chat,
            request_think=False,
            stream_guard=admission_guard,
            model="gpt-oss:20b",
            messages=[],
        )

    message = str(exc_info.value)
    assert f"{percent:.1f}% used" in message
    assert "hard stop at 94.0% or below 2.0 GiB available" in message


def test_foreground_preemption_is_never_suppressed(monkeypatch):
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: pytest.fail("foreground pauses must not be reclassified as memory pauses"),
    )

    def foreground_guard():
        raise CyclePaused("1 active foreground chat turn(s)")

    with pytest.raises(CyclePaused, match="active foreground chat turn"):
        model_selection.developer_chat(
            _streaming_chat,
            request_think=False,
            stream_guard=foreground_guard,
            model="gpt-oss:20b",
            messages=[],
        )


def test_combined_memory_and_foreground_pause_is_never_suppressed(monkeypatch):
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: pytest.fail("combined pauses must remain fail-closed"),
    )

    def combined_guard():
        raise CyclePaused("1 active foreground chat turn(s); memory 83% > 82%")

    with pytest.raises(CyclePaused, match="active foreground chat turn"):
        model_selection.developer_chat(
            _streaming_chat,
            request_think=False,
            stream_guard=combined_guard,
            model="gpt-oss:20b",
            messages=[],
        )
