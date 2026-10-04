from __future__ import annotations

import json
import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

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


class SlowProvider:
    pid = 4242

    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        assert self.stdin.closed
        if self.returncode is None:
            raise subprocess.TimeoutExpired("provider", timeout)
        return self.returncode

    def terminate(self):
        raise AssertionError("An in-flight hardware read must not be terminated")

    def kill(self):
        raise AssertionError("An in-flight hardware read must not be killed")


def test_close_timeout_retains_exact_provider_until_exit(tmp_path):
    launches = []
    collector = BundledHardwareMonitorCollector(
        tmp_path / "provider.exe", popen_factory=lambda *args, **kwargs: launches.append(args)
    )
    provider = SlowProvider()
    collector._process = provider

    with pytest.raises(RuntimeError, match="shutdown-timeout"):
        collector.close()

    assert collector._process is provider
    assert provider.stdin.closed
    assert not provider.stdout.closed
    assert collector.collect()["errors"] == ["bundled-provider:shutdown-pending"]
    assert launches == []
    provider.returncode = 0
    collector.close()
    assert collector._process is None
    assert provider.stdout.closed
    assert collector._shutdown_requested is False


def test_snapshot_failure_reports_slow_shutdown_without_launching_duplicate(tmp_path, monkeypatch):
    launches = []
    collector = BundledHardwareMonitorCollector(
        tmp_path / "provider.exe", popen_factory=lambda *args, **kwargs: launches.append(args)
    )
    provider = SlowProvider()
    collector._process = provider
    monkeypatch.setattr(collector, "_await_payload", lambda *args, **kwargs: (None, "timeout"))

    result = collector.collect()

    assert result["available"] is False
    assert result["errors"] == ["bundled-provider:snapshot-timeout", "bundled-provider:shutdown-timeout"]
    assert collector._process is provider
    assert collector.collect()["errors"] == ["bundled-provider:shutdown-pending"]
    assert launches == []


def test_close_sends_eof_and_allows_slow_helper_cleanup(tmp_path):
    marker = tmp_path / "graceful-exit.txt"
    # This isolated protocol process represents a hardware read that takes
    # longer than the old one-second termination grace period.
    process = subprocess.Popen(
        [sys.executable, "-c", "import pathlib,sys,time; sys.stdin.read(); time.sleep(1.2); pathlib.Path(sys.argv[1]).write_text('clean')", str(marker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    collector = BundledHardwareMonitorCollector(tmp_path / "provider.exe")
    collector._process = process
    try:
        collector.close()
        assert process.returncode == 0
        assert marker.read_text() == "clean"
        assert collector._process is None
    finally:
        # EOF is also the only cleanup request if an assertion fails.
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=5)
