from __future__ import annotations

from localpilot.agent import LocalPilotAgent
from localpilot.systemsense_diagnosis import (
    build_system_diagnosis_prompt,
    collect_system_diagnosis_evidence,
    normalize_diagnosis_scope,
)


class _FakeSense:
    def __init__(self):
        self.inventory_collected = 0

    def collect_dynamic(self):
        return {
            "captured_at": "2026-09-18T04:00:00+00:00",
            "base": {
                "cpu": {"percent": 12.0},
                "memory": {"percent": 32.0},
                "top_processes": [{"name": "msedgewebview2.exe", "cpu_percent": 38.0}],
            },
            "performance": {"gpu": {"engine_utilization_percent": 8.0}},
            "raw_sensors": {
                "source": "LibreHardwareMonitorLib",
                "sensors": [
                    {
                        "HardwareName": "AMD Radeon RX 9070",
                        "SensorType": "Temperature",
                        "Name": "GPU Hot Spot",
                        "Value": 48.0,
                    }
                ],
            },
            # These presentation/normalization layers deliberately exist in the
            # fixture so the test proves they are not forwarded to diagnosis.
            "sensors": {"max_temperature_c": 999.0},
            "derived": {"max_temperature_c": 999.0},
        }

    def history(self, *, metric, hours, limit):
        return {
            "metric": metric,
            "hours": hours,
            "samples": 1,
            "items": [{"captured_at": "2026-09-18T03:59:00+00:00", "value": 10.0}],
        }

    def collect_inventory(self):
        self.inventory_collected += 1
        return {"available": True}

    def raw(self, *, category, limit):
        assert category == "inventory"
        return {
            "captured_at": "2026-09-18T03:58:00+00:00",
            "sections": {"devices": [{"Name": "Example device", "ConfigManagerErrorCode": 0}]},
        }

    def correlations(self, *, limit):
        return {
            "window_days": 7,
            "correlations": [],
            "warning": "Correlation is observational and does not establish causality.",
        }

    def summary(self, *args, **kwargs):
        raise AssertionError("presentation summary must never be used for diagnosis")


class _Audit:
    def __init__(self):
        self.events = []

    def write(self, event, **payload):
        self.events.append((event, payload))


def test_diagnosis_scope_aliases_are_bounded():
    assert normalize_diagnosis_scope("") == "signals"
    assert normalize_diagnosis_scope("temperature") == "thermal"
    assert normalize_diagnosis_scope("VRAM") == "gpu"


def test_diagnosis_evidence_uses_raw_current_truth_not_presentation():
    sense = _FakeSense()
    evidence = collect_system_diagnosis_evidence(sense, scope="signals")

    assert evidence["scope"] == "signals"
    current = evidence["current_truth"]
    assert set(current) == {
        "captured_at",
        "base",
        "windows_performance",
        "hardware_provider",
    }
    assert current["hardware_provider"]["sensors"][0]["Name"] == "GPU Hot Spot"
    assert "sensors" not in current
    assert "derived" not in current
    assert sense.inventory_collected == 1
    assert "cpu.percent" in evidence["recent_metric_history"]
    assert "thermal.max_c" in evidence["recent_metric_history"]


def test_diagnosis_prompt_is_explicitly_evidence_only():
    prompt, evidence = build_system_diagnosis_prompt(_FakeSense(), scope="thermal")

    assert evidence["scope"] == "thermal"
    assert "without using any tools, durable memory" in prompt
    assert "human-facing SystemSense presentation snapshot" in prompt
    assert "GPU Hot Spot" in prompt
    assert "999.0" not in prompt


def test_agent_diagnosis_uses_special_raw_evidence_interface():
    agent = LocalPilotAgent.__new__(LocalPilotAgent)
    agent.systemsense = _FakeSense()
    agent.audit = _Audit()
    calls = []

    def fake_ask(prompt, *, interface="direct"):
        calls.append((prompt, interface))
        return "No action needed."

    agent.ask = fake_ask

    answer = LocalPilotAgent.diagnose_system(agent, "gpu")

    assert answer == "No action needed."
    assert calls and calls[0][1] == "systemsense_diagnostic"
    assert "AMD Radeon RX 9070" in calls[0][0]
    assert agent.audit.events[0][0] == "systemsense_diagnosis_requested"
    assert agent.audit.events[0][1]["sensor_count"] == 1


def test_systemsense_diagnostic_is_not_reclassified_as_product_troubleshooting():
    prompt, _ = build_system_diagnosis_prompt(_FakeSense(), scope="signals")
    issues = LocalPilotAgent._response_behavior_issues(
        prompt,
        (
            "The current machine state looks healthy. CPU and GPU activity are light, "
            "the GPU hot spot is 48 C, and the recent history does not show sustained pressure. "
            "The current readings look like ordinary workload variation rather than a fault. "
            "No action needed."
        ),
    )

    assert "practical_troubleshooting_source_unattributed" not in issues
    assert "unsafe_pla_temperature_example" not in issues
