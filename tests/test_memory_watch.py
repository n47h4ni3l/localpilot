from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config, SystemSenseConfig
from localpilot.memory_watch import (
    is_memory_watch_process_investigation_request,
    is_memory_watch_report_request,
    parse_memory_watch_request,
)
from localpilot.systemsense_watch import (
    is_systemsense_process_investigation_request,
    is_systemsense_watch_report_request,
    parse_systemsense_watch_request,
)
from localpilot.runtime_worker import RuntimeWorker
from localpilot.safety import RiskLevel
from localpilot.systemsense import SystemSense
from localpilot.tools import registry


class _DynamicCollector:
    def __init__(self):
        self.calls = 0

    def collect(self):
        self.calls += 1
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cpu": {
                "percent": 8.0,
                "logical_cpus": 16,
                "physical_cpus": 8,
                "frequency_mhz": 3400.0,
                "max_frequency_mhz": 4500.0,
            },
            "memory": {
                "percent": 72.0 if self.calls == 1 else 84.0,
                "total_gb": 32.0,
                "available_gb": 8.96 if self.calls == 1 else 5.12,
                "swap_percent": 0.0,
            },
            "storage": {"io": {"read_mb_s": 0.0, "write_mb_s": 0.0}, "volumes": []},
            "network": {"send_mbps": 0.0, "receive_mbps": 0.0},
            "battery": None,
            "top_processes": [
                {"pid": 10, "name": "busy.exe", "cpu_percent": 40.0, "ram_mb": 120.0},
            ],
            "top_memory_processes": [
                {
                    "pid": 20,
                    "name": "pythonw.exe",
                    "cpu_percent": 2.0,
                    "ram_mb": 4300.0,
                    "executable": r"C:\LocalPilot\.venv\Scripts\pythonw.exe",
                    "command_line": r"C:\LocalPilot\.venv\Scripts\pythonw.exe -m localpilot.runtime_worker",
                    "parent_pid": 8,
                    "parent_name": "localpilot.exe",
                    "parent_executable": r"C:\LocalPilot\.venv\Scripts\localpilot.exe",
                    "started_at_epoch": 1000.0,
                    "username": r"PC\owner",
                },
                {
                    "pid": 30,
                    "name": "browser.exe",
                    "cpu_percent": 5.0,
                    "ram_mb": 900.0,
                    "executable": r"C:\Browser\browser.exe",
                    "command_line": r"C:\Browser\browser.exe --profile default",
                    "parent_pid": 4,
                    "parent_name": "explorer.exe",
                    "parent_executable": r"C:\Windows\explorer.exe",
                    "started_at_epoch": 900.0,
                    "username": r"PC\owner",
                },
            ],
        }


class _PerformanceCollector:
    def collect(self):
        return {
            "source": "test",
            "available": True,
            "gpu": {"engine_utilization_percent": 5.0},
            "processor": {},
            "power_plan": {},
            "thermal_zones": [],
            "errors": [],
        }


class _SensorCollector:
    def collect(self):
        return {"available": False, "source": "test", "sensors": []}


class _InventoryCollector:
    def collect(self):
        return {"available": True, "devices": []}


def _sense(tmp_path: Path) -> SystemSense:
    config = SystemSenseConfig()
    config.memory_watch_sample_interval_seconds = 15.0
    return SystemSense(
        config,
        tmp_path,
        psutil_collector=_DynamicCollector(),
        performance_collector=_PerformanceCollector(),
        sensor_collector=_SensorCollector(),
        inventory_collector=_InventoryCollector(),
    )


def test_memory_watch_parser_recognizes_real_owner_wording_and_bounds_duration():
    now = datetime(2026, 9, 23, 8, 15, tzinfo=timezone(timedelta(hours=9, minutes=30)))
    intent = parse_memory_watch_request(
        "Something keeps using a substantial amount of RAM periodically. "
        "Can you please monitor that today?",
        now=now,
    )
    assert intent is not None
    assert intent.label == "today"
    assert intent.expires_at.hour == 23
    assert intent.expires_at.minute == 59

    eight_hours = parse_memory_watch_request("Please monitor RAM for a while", now=now)
    assert eight_hours is not None
    assert eight_hours.label == "8 hours"

    assert parse_memory_watch_request("Why is my RAM high?", now=now) is None


def test_memory_watch_report_prompt_detection_is_conservative():
    assert is_memory_watch_report_request("What did the RAM monitor find?")
    assert is_memory_watch_report_request("Show me the memory watch results")
    assert not is_memory_watch_report_request("Why is my RAM high right now?")
    assert not is_memory_watch_report_request("Monitor RAM today")


def test_memory_watch_process_investigation_requires_identity_and_web_evidence():
    prompt = "Can you investigate what that pythonw.exe process is and why it used so much RAM?"

    assert is_memory_watch_process_investigation_request(prompt) is True
    assert LocalPilotAgent._evidence_requirements(prompt) == {
        "SystemSense watch",
        "process identity",
        "public web discovery",
        "public HTTPS",
    }

    no_web = (
        "Investigate what that pythonw.exe process is and why it used so much RAM, "
        "but don't use the web."
    )
    assert LocalPilotAgent._evidence_requirements(no_web) == {
        "SystemSense watch",
        "process identity",
    }


def test_systemsense_memory_watch_persists_process_ram_history(tmp_path, monkeypatch):
    sense = _sense(tmp_path)
    expiry = datetime.now(timezone.utc) + timedelta(hours=2)
    watch = sense.start_memory_watch(expires_at=expiry, label="2 hours")

    monotonic_values = iter([100.0, 116.0])
    monkeypatch.setattr("localpilot.systemsense.time.monotonic", lambda: next(monotonic_values))

    sense.collect_dynamic()
    sense.collect_dynamic()

    report = sense.memory_watch_report()
    assert report["available"] is True
    assert report["watch"]["watch_id"] == watch["watch_id"]
    assert report["watch"]["status"] == "active"
    assert report["samples"] == 2
    assert report["peak_system_memory_percent"] == 84.0
    assert report["minimum_available_gb"] == 5.12
    assert report["top_memory_consumers"][0]["name"] == "pythonw.exe"
    assert report["top_memory_consumers"][0]["pid"] == 20
    assert report["top_memory_consumers"][0]["peak_ram_mb"] == 4300.0
    assert report["top_memory_consumers"][0]["executable"].endswith("pythonw.exe")
    assert "-m localpilot.runtime_worker" in report["top_memory_consumers"][0]["command_line"]
    assert report["top_memory_consumers"][0]["parent_name"] == "localpilot.exe"
    assert report["top_memory_consumers"][0]["started_at_epoch"] == 1000.0
    assert report["top_memory_consumers"][1]["name"] == "browser.exe"

    # The watch/report lives in SQLite and survives a fresh SystemSense object.
    restored = _sense(tmp_path)
    restored_report = restored.memory_watch_report(watch_id=watch["watch_id"])
    assert restored_report["samples"] == 2
    assert restored_report["top_memory_consumers"][0]["name"] == "pythonw.exe"


def test_memory_watch_sampling_is_independent_of_cpu_ranking(tmp_path, monkeypatch):
    sense = _sense(tmp_path)
    sense.start_memory_watch(
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        label="1 hour",
    )
    monkeypatch.setattr("localpilot.systemsense.time.monotonic", lambda: 100.0)
    sense.collect_dynamic()

    report = sense.memory_watch_report()
    names = [row["name"] for row in report["top_memory_consumers"]]
    assert "pythonw.exe" in names
    assert "busy.exe" not in names


class _FakeAgent:
    def __init__(self):
        self.ask_calls = []
        self.ack_calls = []
        self.messages = []

    def ask(self, prompt, *, interface="direct"):
        self.ask_calls.append((prompt, interface))
        return "heavy model answer"

    def acknowledge_systemsense_watch(self, prompt, watch):
        self.ack_calls.append((prompt, dict(watch)))
        return "I’ll keep a real SystemSense RAM watch running today and record the top memory consumers."


class _FakeSense:
    def __init__(self):
        self.calls = []

    def start_watch(self, *, profile, expires_at, label):
        self.calls.append((profile, expires_at, label))
        return {
            "watch_id": 4,
            "created_at": "2026-09-23T00:00:00+00:00",
            "expires_at": expires_at.isoformat(),
            "status": "active",
            "label": label,
            "profile": profile,
            "sample_interval_seconds": 15.0,
            "retention_days": 7,
        }


def test_runtime_worker_registers_memory_watch_without_heavy_reasoning(tmp_path):
    worker = RuntimeWorker.__new__(RuntimeWorker)
    worker.root = tmp_path
    worker.config = Config()
    worker._agents = {}
    worker._write_lock = threading.Lock()
    worker._active_request_id = None
    worker._active_session_id = None
    worker.systemsense = _FakeSense()
    agent = _FakeAgent()
    worker._agent = lambda _session_id, _history: agent
    messages = []
    worker._write = messages.append

    prompt = (
        "Something keeps using a substantial amount of RAM periodically. "
        "Can you please monitor that today?"
    )
    worker.handle(
        {
            "kind": "ask",
            "request_id": "req-watch",
            "session_id": "session-1",
            "prompt": prompt,
            "history": [],
        }
    )

    assert len(worker.systemsense.calls) == 1
    assert worker.systemsense.calls[0][0] == "memory"
    assert agent.ask_calls == []
    assert agent.ack_calls and agent.ack_calls[0][0] == prompt
    result = next(message for message in messages if message.get("kind") == "result")
    assert "RAM watch" in result["answer"]


def test_agent_memory_watch_ack_is_model_generated_but_bounded(tmp_path, monkeypatch):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(
            [
                {
                    "message": {
                        "content": (
                            "I’ve started the RAM watch. I’ll record the top memory consumers "
                            "every 15 seconds until tonight, and we can review what actually spiked later."
                        ),
                        "thinking": "",
                        "tool_calls": [],
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 80,
                    "eval_count": 32,
                }
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    agent = LocalPilotAgent(Config(), tmp_path)
    watch = {
        "watch_id": 1,
        "expires_at": "2026-09-23T14:29:59+00:00",
        "status": "active",
        "sample_interval_seconds": 15.0,
    }

    watch["profile"] = "memory"
    answer = agent.acknowledge_systemsense_watch("Please monitor RAM today", watch)

    assert "record the top memory consumers" in answer
    assert len(calls) == 1
    assert calls[0]["think"] is False
    assert "tools" not in calls[0]
    assert calls[0]["options"]["num_predict"] == 192
    context = "\n".join(str(message.get("content") or "") for message in calls[0]["messages"])
    assert "application has already" in context
    assert "authoritative application state" in context


def test_memory_watch_process_identity_uses_historical_instance_and_provenance(
    tmp_path, monkeypatch
):
    sense = _sense(tmp_path)
    watch = sense.start_memory_watch(
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        label="1 hour",
    )
    monkeypatch.setattr("localpilot.systemsense.time.monotonic", lambda: 100.0)

    def fake_artifact_id(process):
        return sense.store.upsert_executable_artifact(
            {
                "available": True,
                "path": process["executable"],
                "size_bytes": 123,
                "modified_ns": 456,
                "company_name": "Python Software Foundation",
                "product_name": "Python",
                "signature_status": "Valid",
                "sha256": "ABC",
            }
        )

    monkeypatch.setattr(sense, "_artifact_id_for_process", fake_artifact_id)
    sense.collect_dynamic()

    historical = sense.memory_watch_process_identity(
        pid=20,
        watch_id=watch["watch_id"],
    )
    assert historical["available"] is True
    assert historical["pid"] == 20
    assert historical["name"] == "pythonw.exe"
    assert historical["executable"].endswith("pythonw.exe")
    assert historical["parent_name"] == "localpilot.exe"
    assert historical["peak_ram_mb"] == 4300.0

    import localpilot.tools.systemsense as systemsense_tools

    monkeypatch.setattr(
        systemsense_tools,
        "inspect_process_identity",
        lambda pid: __import__("json").dumps(
            {
                "available": True,
                "pid": pid,
                "running": True,
                "started_at_epoch": 1000.0,
                "name": "pythonw.exe",
                "executable": r"C:\LocalPilot\.venv\Scripts\pythonw.exe",
            }
        ),
    )
    monkeypatch.setattr(
        systemsense_tools,
        "inspect_process_launch_context",
        lambda *args, **kwargs: __import__("json").dumps(
            {"available": True, "services": [], "scheduled_tasks": [], "startup_items": []}
        ),
    )

    tools = registry(tmp_path, config=Config(), systemsense=sense)
    payload = __import__("json").loads(
        tools["inspect_memory_watch_process"].fn(
            pid=20,
            watch_id=watch["watch_id"],
        )
    )

    assert payload["available"] is True
    assert payload["company_name"] == "Python Software Foundation"
    assert payload["signature_status"] == "Valid"
    assert payload["sha256"] == "ABC"
    assert payload["current_process"]["available"] is True
    assert "observation time" in payload["identity_note"]


def test_registry_exposes_memory_watch_report_as_read_only_evidence(tmp_path):
    config = Config()
    sense = _sense(tmp_path / config.agent.data_dir)
    tools = registry(tmp_path, config=config, systemsense=sense)

    assert tools["get_systemsense_watch_report"].risk is RiskLevel.READ_ONLY
    assert tools["inspect_systemsense_watch_process"].risk is RiskLevel.READ_ONLY
    assert tools["get_memory_watch_report"].risk is RiskLevel.READ_ONLY
    assert tools["inspect_memory_watch_process"].risk is RiskLevel.READ_ONLY
    assert tools["inspect_process_identity"].risk is RiskLevel.READ_ONLY
    assert LocalPilotAgent._evidence_requirements(
        "What did the RAM monitor find?"
    ) == {"SystemSense watch"}



def test_generic_systemsense_watch_parser_covers_major_resource_profiles():
    now = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
    cases = {
        "Please monitor CPU for 2 hours": "cpu",
        "Watch the GPU and VRAM today": "gpu",
        "Track disk activity for 30 minutes": "storage",
        "Monitor network traffic today": "network",
        "Watch the computer for freezes today": "system",
    }
    for prompt, profile in cases.items():
        intent = parse_systemsense_watch_request(prompt, now=now)
        assert intent is not None
        assert intent.profile == profile

    combined = parse_systemsense_watch_request(
        "Monitor GPU and RAM today",
        now=now,
    )
    assert combined is not None
    assert combined.profile == "system"

    assert is_systemsense_watch_report_request("What did the SystemSense watch find?")
    assert is_systemsense_process_investigation_request(
        "Investigate what that worker.exe process was doing during the GPU watch"
    )
