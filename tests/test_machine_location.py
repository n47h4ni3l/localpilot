from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.machine_location import LocationSnapshot, MachineLocation
from localpilot.config import Config
from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot import webview_app


def _snapshot() -> LocationSnapshot:
    return LocationSnapshot(
        latitude=-34.812345,
        longitude=138.612345,
        accuracy_m=37.4,
        updated_at=datetime.now(UTC).isoformat(),
    )


def test_location_defaults_off_and_does_not_capture(tmp_path, monkeypatch):
    location = MachineLocation(tmp_path)
    calls = []

    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(lambda: calls.append(True) or _snapshot()),
    )

    status = location.public_status()
    assert status["enabled"] is False
    assert status["available"] is False
    assert calls == []

    location.refresh()
    assert calls == []


def test_enabling_location_captures_exact_locally_but_exposes_only_coarse_context(
    tmp_path, monkeypatch
):
    location = MachineLocation(tmp_path)
    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(_snapshot),
    )

    status = location.set_enabled(True)
    assert status["enabled"] is True
    assert status["available"] is True
    assert status["source"] == "Windows Location Service"
    assert status["accuracyM"] == 37.4
    assert "latitude" not in status
    assert "longitude" not in status

    raw = json.loads(location.path.read_text(encoding="utf-8"))
    assert raw["latitude"] == -34.812345
    assert raw["longitude"] == 138.612345

    context = location.coarse_model_context(refresh_if_stale=False)
    assert context is not None
    assert context["approximate_latitude"] == -34.81
    assert context["approximate_longitude"] == 138.61
    assert "latitude" not in {key for key in context if key in {"latitude", "longitude"}}
    assert "Exact coordinates remain" in context["privacy"]


def test_turning_location_off_deletes_retained_precise_coordinates(tmp_path, monkeypatch):
    location = MachineLocation(tmp_path)
    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(_snapshot),
    )

    location.set_enabled(True)
    location.set_enabled(False)

    state = location.read()
    assert state["enabled"] is False
    assert state["latitude"] is None
    assert state["longitude"] is None
    assert state["updated_at"] is None


def test_location_prompt_detection_is_bounded_to_location_dependent_requests():
    assert MachineLocation.prompt_needs_location("What's the weather tomorrow?")
    assert MachineLocation.prompt_needs_location("Find coffee near me")
    assert MachineLocation.prompt_needs_location("Where am I?")
    assert MachineLocation.prompt_needs_location("What time is sunset?")
    assert not MachineLocation.prompt_needs_location("Good evening")
    assert not MachineLocation.prompt_needs_location("What is the square root of 5?")


def test_window_bridge_location_toggle_uses_private_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(_snapshot),
    )

    class FakeWindow:
        width = webview_app.EXPANDED_SIZE[0]
        height = webview_app.EXPANDED_SIZE[1]
        x = 0
        y = 0

    bridge = webview_app.WindowBridge(FakeWindow(), tmp_path, None)

    before = bridge.get_location_settings()
    assert before["enabled"] is False

    enabled = bridge.set_location_enabled(True)
    assert enabled["enabled"] is True
    assert enabled["available"] is True
    assert "latitude" not in enabled
    assert "longitude" not in enabled

    refreshed = bridge.refresh_location()
    assert refreshed["available"] is True

    disabled = bridge.set_location_enabled(False)
    assert disabled["enabled"] is False
    assert bridge._location.read()["latitude"] is None


def test_location_settings_ui_never_renders_precise_coordinates():
    root = Path(__file__).resolve().parents[1]
    index = (root / "localpilot" / "webview" / "index.html").read_text(encoding="utf-8")
    script = (root / "localpilot" / "webview" / "settings-location.js").read_text(
        encoding="utf-8"
    )

    assert 'id="toggle-location"' in index
    assert 'id="location-status"' in index
    assert 'id="location-source"' in index
    assert 'id="location-refresh"' in index
    assert 'src="settings-location.js"' in index
    assert 'href="settings-location.css"' in index
    assert "Exact coordinates stay local" in index

    assert "get_location_settings" in script
    assert "set_location_enabled" in script
    assert "refresh_location" in script
    assert "latitude" not in script
    assert "longitude" not in script


def test_registry_exposes_only_coarse_machine_location(tmp_path, monkeypatch):
    config = Config()
    location = MachineLocation(tmp_path / config.agent.data_dir)
    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(_snapshot),
    )
    location.set_enabled(True)

    tools = registry(tmp_path, config=config, machine_location=location)
    spec = tools["get_machine_location"]
    assert spec.risk is RiskLevel.READ_ONLY

    result = spec.fn()
    assert result["enabled"] is True
    assert result["available"] is True
    assert result["approximate_latitude"] == -34.81
    assert result["approximate_longitude"] == 138.61
    assert "latitude" not in {key for key in result if key in {"latitude", "longitude"}}
    assert "Exact machine coordinates remain local" in result["privacy"]



def _chunk(*, content: str = "", thinking: str = "", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=list(tool_calls or []),
        )
    )


def _call(name: str, arguments: dict | None = None):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=dict(arguments or {}))
    )


def test_enabled_machine_location_drives_here_weather_research(tmp_path, monkeypatch):
    config = Config()
    config.agent.research_soft_tool_rounds = 4
    config.agent.research_hard_tool_rounds = 4
    monkeypatch.setattr(
        MachineLocation,
        "_capture_windows_location",
        staticmethod(_snapshot),
    )
    agent = LocalPilotAgent(config, tmp_path)
    agent.governor = SimpleNamespace(
        sample=lambda interval: SimpleNamespace(background_allowed=False),
        apply_process_priority=lambda idle: None,
    )
    agent.machine_location.set_enabled(True)

    search_spec = agent.tools["search_public_web"]
    fetch_spec = agent.tools["fetch_public_https"]
    agent.tools["search_public_web"] = SimpleNamespace(
        risk=search_spec.risk,
        fn=lambda **_kwargs: (
            "Public web search: weather tomorrow near -34.81, 138.61\n"
            "1. Forecast - https://weather.example/forecast"
        ),
    )
    agent.tools["fetch_public_https"] = SimpleNamespace(
        risk=fetch_spec.risk,
        fn=lambda **kwargs: (
            f"Public HTTPS source: {kwargs['url']}\n"
            "Tomorrow will be mild with a high of 24 C and a chance of showers."
        ),
    )

    calls = []
    streams = iter(
        [
            [[_chunk(tool_calls=[_call(
                "search_public_web",
                {"query": "weather tomorrow near -34.81 138.61"},
            )])]],
            [[_chunk(tool_calls=[_call(
                "fetch_public_https",
                {"url": "https://weather.example/forecast"},
            )])]],
            [[_chunk(content="I have the location and fresh forecast evidence.")]],
            [[_chunk(content=(
                "Tomorrow should be mild, around 24 °C at the top end, with a chance "
                "of showers. You do not need to give me your suburb while PC location is enabled."
            ))]],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams)[0])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    answer = agent.ask(
        "Good afternoon. What's the weather going to be like here tomorrow?",
        interface="desktop",
    )

    assert "24 °C" in answer
    assert "give me your suburb" in answer
    assert len(calls) == 4

    first_context = "\n".join(
        str(message.get("content") or "") for message in calls[0]["messages"]
    )
    assert "MACHINE LOCATION CONTEXT" in first_context
    assert "Do not ask the owner for a city, suburb, postcode, or ZIP" in first_context
    assert "authoritative machine-location evidence for this turn" in first_context
    assert '"approximate_latitude": -34.81' in first_context
    assert '"approximate_longitude": 138.61' in first_context

    evidence_events = agent.audit.recent("tool_result", limit=20)
    assert not any(event.get("tool") == "get_machine_location" for event in evidence_events)
    assert any(
        event.get("tool") == "search_public_web"
        and event.get("evidence_source") == "public web discovery"
        and event.get("ok") is True
        for event in evidence_events
    )
    assert any(
        event.get("tool") == "fetch_public_https"
        and event.get("evidence_source") == "public HTTPS"
        and event.get("ok") is True
        for event in evidence_events
    )

    assert "MACHINE LOCATION CONTEXT" not in str(agent.messages)
    assert "-34.81" not in str(agent.messages)
    assert "138.61" not in str(agent.messages)
