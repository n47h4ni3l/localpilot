from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from localpilot.config import SystemSenseConfig
from localpilot.process_identity import inspect_executable_artifact
from localpilot.systemsense import SystemSense, SystemSenseStore
from localpilot.systemsense_collectors import WindowsPerformanceCollector
from localpilot.systemsense_watch import (
    is_systemsense_process_investigation_request,
    is_systemsense_watch_report_request,
)


def test_systemsense_store_migrates_legacy_memory_watch_into_versioned_generic_schema(tmp_path):
    database = tmp_path / "systemsense.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE memory_watches (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL,
            label TEXT NOT NULL
        );
        CREATE TABLE memory_watch_samples (
            id INTEGER PRIMARY KEY,
            watch_id INTEGER NOT NULL,
            captured_at TEXT NOT NULL,
            system_memory_percent REAL NOT NULL,
            system_available_gb REAL
        );
        CREATE TABLE memory_watch_process_samples (
            sample_id INTEGER NOT NULL,
            pid INTEGER NOT NULL,
            name TEXT NOT NULL,
            ram_mb REAL NOT NULL,
            cpu_percent REAL NOT NULL
        );
        INSERT INTO memory_watches
            VALUES (1, '2026-09-23T00:00:00+00:00', '2026-09-23T08:00:00+00:00', 'completed', 'today');
        INSERT INTO memory_watch_samples
            VALUES (1, 1, '2026-09-23T01:00:00+00:00', 82.0, 5.5);
        INSERT INTO memory_watch_process_samples
            VALUES (1, 42, 'worker.exe', 4096.0, 2.0);
        """
    )
    connection.commit()
    connection.close()

    store = SystemSenseStore(database)
    migrated = store.watch_report(watch_id=1)

    assert migrated["available"] is True
    assert migrated["watch"]["profile"] == "memory"
    assert migrated["samples"] == 1
    assert migrated["peak_system_memory_percent"] == 82.0
    assert migrated["top_memory_consumers"][0]["name"] == "worker.exe"
    with sqlite3.connect(database) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2


def test_executable_fingerprint_is_observation_time_content_identity(tmp_path):
    executable = tmp_path / "worker.exe"
    executable.write_bytes(b"first-binary")
    first = inspect_executable_artifact(str(executable))

    assert first["available"] is True
    assert first["sha256"] == hashlib.sha256(b"first-binary").hexdigest().upper()

    store = SystemSenseStore(tmp_path / "artifacts.db")
    first_id = store.upsert_executable_artifact(first)
    same_id = store.upsert_executable_artifact(first)
    assert first_id == same_id

    executable.write_bytes(b"replacement-binary")
    second = inspect_executable_artifact(str(executable))
    second_id = store.upsert_executable_artifact(second)

    assert second["sha256"] != first["sha256"]
    assert second_id != first_id


class WatchDynamicCollector:
    def __init__(self, executable: Path):
        self.executable = executable
        self.calls = 0

    def collect(self):
        self.calls += 1
        process_rows = []
        if self.calls <= 2:
            process_rows = [
                {
                    "pid": 424242,
                    "name": "worker.exe",
                    "cpu_percent": 21.0,
                    "ram_mb": 3100.0,
                    "io_total_mb": 900.0,
                    "io_read_mb_s": 18.0,
                    "io_write_mb_s": 7.0,
                }
            ]
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cpu": {"percent": 32.0, "frequency_mhz": 4100.0},
            "memory": {"percent": 78.0, "available_gb": 7.0},
            "storage": {"io": {"read_mb_s": 40.0, "write_mb_s": 20.0}, "volumes": []},
            "network": {"send_mbps": 12.0, "receive_mbps": 55.0},
            "battery": None,
            "top_processes": list(process_rows),
            "top_memory_processes": list(process_rows),
            "top_io_processes": list(process_rows),
        }

    def enrich_processes(self, pids):
        if 424242 not in set(pids) or self.calls > 2:
            return []
        return [
            {
                "pid": 424242,
                "name": "worker.exe",
                "cpu_percent": 21.0,
                "ram_mb": 3100.0,
                "vms_mb": 5200.0,
                "private_mb": 2950.0,
                "pagefile_mb": 2950.0,
                "page_faults": 12345,
                "threads": 18,
                "handles": 640,
                "io_read_mb": 900.0,
                "io_write_mb": 400.0,
                "io_read_mb_s": 18.0,
                "io_write_mb_s": 7.0,
                "executable": str(self.executable),
                "command_line": f"{self.executable} --serve",
                "parent_pid": 100,
                "parent_name": "launcher.exe",
                "parent_executable": "launcher.exe",
                "started_at_epoch": 1000.0,
                "username": "PC\\owner",
                "ancestry": [
                    {
                        "pid": 100,
                        "name": "launcher.exe",
                        "executable": "launcher.exe",
                        "started_at_epoch": 900.0,
                    }
                ],
            }
        ]

    def collect_process_connections(self, pids=None):
        if self.calls > 2:
            return []
        rows = [
            {
                "pid": 424242,
                "family": "AddressFamily.AF_INET",
                "type": "SocketKind.SOCK_STREAM",
                "local_endpoint": "192.168.1.2:51000",
                "remote_endpoint": "203.0.113.20:443",
                "status": "ESTABLISHED",
            }
        ]
        if pids is None:
            return rows
        wanted = set(pids)
        return [row for row in rows if row["pid"] in wanted]


class WatchPerformanceCollector:
    def collect(self):
        return {
            "source": "test",
            "available": True,
            "gpu": {"engine_utilization_percent": 72.0},
            "processor": {},
            "power_plan": {},
            "thermal_zones": [],
            "errors": [],
        }

    def collect_process_gpu(self, pids=None):
        if pids is not None and 424242 not in set(pids):
            return {}
        return {
            424242: {
                "gpu_percent": 66.0,
                "gpu_dedicated_mb": 2048.0,
                "gpu_shared_mb": 128.0,
                "gpu_committed_mb": 2176.0,
            }
        }


class EmptySensors:
    def collect(self):
        return {"available": False, "source": "test", "sensors": []}


class EmptyInventory:
    def collect(self):
        return {"available": True, "devices": []}


def test_systemsense_watch_correlates_richer_process_gpu_network_lifecycle_and_fingerprint(
    tmp_path, monkeypatch
):
    executable = tmp_path / "worker.exe"
    executable.write_bytes(b"watched-executable")
    config = SystemSenseConfig()
    config.memory_watch_sample_interval_seconds = 15.0
    dynamic = WatchDynamicCollector(executable)
    sense = SystemSense(
        config,
        tmp_path,
        psutil_collector=dynamic,
        performance_collector=WatchPerformanceCollector(),
        sensor_collector=EmptySensors(),
        inventory_collector=EmptyInventory(),
    )
    watch = sense.start_watch(
        profile="system",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        label="system test",
    )

    monotonic = iter([100.0, 116.0, 132.0])
    monkeypatch.setattr("localpilot.systemsense.time.monotonic", lambda: next(monotonic))

    real_process = psutil.Process

    def fake_process(pid):
        if int(pid) == 424242:
            raise psutil.NoSuchProcess(pid)
        return real_process(pid)

    sense.collect_dynamic()
    sense.collect_dynamic()
    monkeypatch.setattr("localpilot.systemsense.psutil.Process", fake_process)
    sense.collect_dynamic()

    report = sense.watch_report(watch_id=watch["watch_id"])
    assert report["samples"] == 3
    row = next(item for item in report["top_processes"] if item["pid"] == 424242)
    assert row["peak_ram_mb"] == 3100.0
    assert row["peak_private_mb"] == 2950.0
    assert row["peak_handles"] == 640
    assert row["peak_page_faults"] == 12345
    assert row["peak_read_mb_s"] == 18.0
    assert row["peak_gpu_percent"] == 66.0
    assert row["peak_gpu_dedicated_mb"] == 2048.0
    assert row["sha256"] == hashlib.sha256(b"watched-executable").hexdigest().upper()

    identity = sense.watch_process_identity(pid=424242, watch_id=watch["watch_id"])
    assert identity["available"] is True
    assert identity["network_connections"][0]["remote_endpoint"] == "203.0.113.20:443"
    assert identity["ancestry"][0]["name"] == "launcher.exe"
    events = [event["event"] for event in identity["lifecycle"]]
    assert "first_observed" in events
    assert "process_exited" in events


class FakeGpuWmi:
    available = True

    def query(self, _namespace, class_name, _properties, _where=""):
        if class_name.endswith("GPUEngine"):
            return [
                {"Name": "pid_77_luid_0x0_phys_0_eng_0_engtype_3D", "UtilizationPercentage": 35},
                {"Name": "pid_77_luid_0x0_phys_0_eng_1_engtype_Copy", "UtilizationPercentage": 15},
                {"Name": "pid_88_luid_0x0_phys_0_eng_0_engtype_3D", "UtilizationPercentage": 5},
            ]
        if class_name.endswith("GPUProcessMemory"):
            return [
                {
                    "Name": "pid_77_luid_0x0_phys_0",
                    "DedicatedUsage": 1024 * 1024 * 1500,
                    "SharedUsage": 1024 * 1024 * 200,
                    "TotalCommitted": 1024 * 1024 * 1700,
                }
            ]
        return []


def test_windows_gpu_counters_are_attributed_per_process():
    collector = WindowsPerformanceCollector(FakeGpuWmi())
    result = collector.collect_process_gpu(None)

    assert result[77]["gpu_percent"] == 50.0
    assert result[77]["gpu_dedicated_mb"] == 1500.0
    assert result[77]["gpu_shared_mb"] == 200.0
    assert result[88]["gpu_percent"] == 5.0


def test_watch_followup_detection_does_not_hijack_current_state_questions():
    assert not is_systemsense_watch_report_request("What is my memory usage right now?")
    assert not is_systemsense_process_investigation_request(
        "Why is python.exe using memory right now?"
    )
    assert is_systemsense_watch_report_request("What did the GPU watch find?")
    assert is_systemsense_process_investigation_request(
        "Investigate what that pythonw.exe process was during the memory spike"
    )
