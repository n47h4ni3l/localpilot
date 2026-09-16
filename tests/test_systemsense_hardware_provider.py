from __future__ import annotations

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
