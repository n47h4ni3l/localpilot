from __future__ import annotations

from types import SimpleNamespace

import localpilot.selfdev_model_selection as model_selection
from localpilot.selfdev_results import CyclePaused


_GIB = 1024**3


def _streaming_chat(**_kwargs):
    return iter([{"message": {"content": "ok", "tool_calls": []}}])


def test_new_selfdev_model_is_unloaded_after_suppressed_admission_pressure(monkeypatch):
    unloaded = []
    monkeypatch.setattr(model_selection, "running_ollama_models", lambda: set())
    monkeypatch.setattr(model_selection, "_unload_ollama_model", unloaded.append)
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=82.3, available=5 * _GIB),
    )

    def guard():
        raise CyclePaused("memory 82.3% > 82.0%")

    response = model_selection.developer_chat(
        _streaming_chat,
        request_think=False,
        keep_alive="30m",
        stream_guard=guard,
        model="gpt-oss:20b",
        messages=[],
    )

    assert response.message["content"] == "ok"
    assert unloaded == ["gpt-oss:20b"]


def test_preexisting_resident_model_is_not_unloaded(monkeypatch):
    unloaded = []
    monkeypatch.setattr(
        model_selection,
        "running_ollama_models",
        lambda: {"gpt-oss:20b"},
    )
    monkeypatch.setattr(model_selection, "_unload_ollama_model", unloaded.append)
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=82.3, available=5 * _GIB),
    )

    def guard():
        raise CyclePaused("memory 82.3% > 82.0%")

    model_selection.developer_chat(
        _streaming_chat,
        request_think=False,
        keep_alive="30m",
        stream_guard=guard,
        model="gpt-oss:20b",
        messages=[],
    )

    assert unloaded == []


def test_model_is_kept_resident_when_background_gate_never_trips(monkeypatch):
    unloaded = []
    monkeypatch.setattr(model_selection, "running_ollama_models", lambda: set())
    monkeypatch.setattr(model_selection, "_unload_ollama_model", unloaded.append)

    model_selection.developer_chat(
        _streaming_chat,
        request_think=False,
        keep_alive="30m",
        stream_guard=lambda: None,
        model="gpt-oss:20b",
        messages=[],
    )

    assert unloaded == []


def test_zero_keep_alive_does_not_issue_redundant_unload(monkeypatch):
    unloaded = []
    monkeypatch.setattr(model_selection, "running_ollama_models", lambda: set())
    monkeypatch.setattr(model_selection, "_unload_ollama_model", unloaded.append)
    monkeypatch.setattr(
        model_selection.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=82.3, available=5 * _GIB),
    )

    def guard():
        raise CyclePaused("memory 82.3% > 82.0%")

    model_selection.developer_chat(
        _streaming_chat,
        request_think=False,
        keep_alive=0,
        stream_guard=guard,
        model="gpt-oss:20b",
        messages=[],
    )

    assert unloaded == []
