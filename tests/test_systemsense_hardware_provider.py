from __future__ import annotations

import json
from types import SimpleNamespace

from localpilot import systemsense_collectors
from localpilot.systemsense_hardware import (
    BundledHardwareMonitorCollector,
    SystemSenseHardwareCollector,
)


class FakeBundled:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def collect(self):
        return self.payload

    def close(self):
        self.closed = True


class FakeWmi:
    available = True

    def query(self, namespace, class_name, properties, where=""):
        del class_name, properties, where
        if namespace == r"root\OpenHardwareMonitor":
            return [
                {
                    "Identifier": "/cpu/0/temperature/0",
                    "Name": "CPU Package",
                    "SensorType": "Temperature",
                    "Value": 51.0,
                    "Min": 38.0,
                    "Max": 72.0,
                    "Parent": "/cpu/0",
                }
            ]
        return []


class FakeStdin:
    def __init__(self):
        self.writes = []
        self.flushes = 0

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        self.flushes += 1


def test_package_installs_bundled_first_collector():
    assert systemsense_collectors.LibreHardwareMonitorCollector is SystemSenseHardwareCollector


def test_bundled_provider_wins_when_live_sensors_are_available():
    bundled = FakeBundled(
        {
            "source": "LibreHardwareMonitorLib",
            "available": True,
            "provider_version": "0.9.6.0",
            "errors": [],
            "sensors": [
                {
                    "Identifier": "/gpu-amd/0/temperature/0",
                    "Name": "GPU Core",
                    "SensorType": "Temperature",
                    "Value": 62.0,
                    "Min": 36.0,
                    "Max": 70.0,
                    "Parent": "/gpu-amd/0",
                }
            ],
        }
    )
    collector = SystemSenseHardwareCollector(FakeWmi(), bundled_collector=bundled)

    result = collector.collect()

    assert result["source"] == "LibreHardwareMonitorLib"
    assert result["available"] is True
    assert result["sensors"][0]["Value"] == 62.0


def test_legacy_wmi_sensor_source_remains_a_fallback():
    bundled = FakeBundled(
        {
            "source": "LibreHardwareMonitorLib",
            "available": False,
            "sensors": [],
            "errors": ["bundled-provider:not-installed"],
        }
    )
    collector = SystemSenseHardwareCollector(FakeWmi(), bundled_collector=bundled)

    result = collector.collect()

    assert result["source"] == "OpenHardwareMonitor"
    assert result["available"] is True
    assert result["fallback_from"] == "LibreHardwareMonitorLib"
    assert "bundled-provider:not-installed" in result["errors"]
    assert result["sensors"][0]["Name"] == "CPU Package"


def test_bundled_payload_normalization_rejects_non_sensor_payloads():
    result = BundledHardwareMonitorCollector._normalize_payload(
        {"ok": True, "source": "LibreHardwareMonitorLib", "sensors": "not-a-list"}
    )

    assert result["available"] is False
    assert result["sensors"] == []


def test_provider_readiness_uses_separate_startup_handshake(tmp_path):
    collector = BundledHardwareMonitorCollector(
        tmp_path / "provider.exe",
        timeout_seconds=1.0,
        startup_timeout_seconds=12.0,
    )
    stdin = FakeStdin()
    process = SimpleNamespace(pid=4242, stdin=stdin)
    collector._responses.put(
        (
            4242,
            json.dumps(
                {
                    "ok": True,
                    "source": "LibreHardwareMonitorLib",
                    "command": "pong",
                }
            ),
        )
    )

    ready, errors = collector._wait_until_ready(process)

    assert ready is True
    assert errors == []
    assert stdin.writes == ["ping\n"]
    assert stdin.flushes == 1
    assert collector.startup_timeout_seconds == 12.0
    assert collector.timeout_seconds == 1.0


def test_provider_response_wait_ignores_lines_from_replaced_process(tmp_path):
    collector = BundledHardwareMonitorCollector(tmp_path / "provider.exe")
    process = SimpleNamespace(pid=22)
    collector._responses.put((11, json.dumps({"ok": False, "command": "stale"})))
    collector._responses.put((22, json.dumps({"ok": True, "command": "pong"})))

    payload, error = collector._await_payload(process, timeout_seconds=0.2)

    assert error is None
    assert payload == {"ok": True, "command": "pong"}


def test_collector_close_releases_bundled_provider():
    bundled = FakeBundled(
        {
            "source": "LibreHardwareMonitorLib",
            "available": False,
            "sensors": [],
            "errors": [],
        }
    )
    collector = SystemSenseHardwareCollector(FakeWmi(), bundled_collector=bundled)

    collector.close()

    assert bundled.closed is True
