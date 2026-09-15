from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import localpilot.systemsense_backend as backend_module
from localpilot.systemsense_backend import BackendTelemetryCollector
from localpilot.tools.systemsense import SystemSenseReader


_GIB = 1024**3
_MIB = 1024**2


class FakeWmi:
    available = True

    def query(self, _namespace, class_name, _properties, _where=""):
        if class_name == "Win32_PerfFormattedData_PerfOS_Memory":
            return [
                {
                    "AvailableBytes": 18 * _GIB,
                    "CacheBytes": 4 * _GIB,
                    "CommittedBytes": 20 * _GIB,
                    "CommitLimit": 40 * _GIB,
                    "PageFaultsPersec": 12,
                    "PageReadsPersec": 1,
                    "PageWritesPersec": 2,
                    "PagesInputPersec": 3,
                    "PagesOutputPersec": 4,
                    "PoolNonpagedBytes": 512 * _MIB,
                    "PoolPagedBytes": 768 * _MIB,
                }
            ]
        return []


class FakePerformance:
    def collect(self):
        return {
            "source": "windows-wmi",
            "available": True,
            "gpu": {"engine_utilization_percent": 44.0},
            "processor": {"PercentPerformanceLimit": 100},
            "thermal_zones": [],
            "power_plan": {"ElementName": "Balanced"},
            "errors": [],
        }


class FakeSensors:
    def collect(self):
        return {
            "source": "LibreHardwareMonitor",
            "available": True,
            "errors": [],
            "sensors": [
                {
                    "Identifier": "/cpu/0/temperature/0",
                    "Name": "CPU Package",
                    "SensorType": "Temperature",
                    "Value": 61.0,
                    "Min": 38.0,
                    "Max": 67.0,
                    "Parent": "/cpu/0",
                }
            ],
        }


class FakeProcess:
    def __init__(self, info):
        self.info = info


def make_collector() -> BackendTelemetryCollector:
    wmi = FakeWmi()
    return BackendTelemetryCollector(
        wmi=wmi,
        performance=FakePerformance(),
        sensors=FakeSensors(),
    )


def test_backend_memory_cross_checks_psutil_with_native_windows_commit(monkeypatch):
    monkeypatch.setattr(
        backend_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            percent=42.7,
            total=32 * _GIB,
            available=18.3 * _GIB,
            used=13.7 * _GIB,
            free=18.3 * _GIB,
        ),
    )
    monkeypatch.setattr(
        backend_module.psutil,
        "swap_memory",
        lambda: SimpleNamespace(
            percent=10.0,
            total=8 * _GIB,
            used=0.8 * _GIB,
            free=7.2 * _GIB,
            sin=0,
            sout=0,
        ),
    )

    payload = make_collector().memory()

    assert payload["physical"]["percent"] == 42.7
    assert payload["physical"]["available_gib"] == 18.3
    native = payload["windows_native"]
    assert native["available"] is True
    assert native["values"]["AvailableBytes_gib"] == 18.0
    assert native["values"]["CommittedBytes_gib"] == 20.0
    assert native["values"]["CommitLimit_gib"] == 40.0
    assert native["values"]["commit_percent"] == 50.0


def test_backend_process_view_separates_cpu_ram_and_aggregates_process_families(monkeypatch):
    def proc(pid, name, cpu, rss_mb, private_mb):
        return FakeProcess(
            {
                "pid": pid,
                "ppid": 1,
                "name": name,
                "cpu_percent": cpu,
                "num_threads": 4,
                "memory_info": SimpleNamespace(
                    rss=rss_mb * _MIB,
                    vms=(rss_mb + 100) * _MIB,
                    private=private_mb * _MIB,
                    pagefile=private_mb * _MIB,
                ),
                "io_counters": SimpleNamespace(
                    read_bytes=10 * _MIB,
                    write_bytes=5 * _MIB,
                ),
            }
        )

    rows = [
        proc(10, "ChatGPT.exe", 2.0, 700, 500),
        proc(11, "ChatGPT.exe", 1.0, 600, 400),
        proc(12, "worker.exe", 30.0, 200, 180),
    ]
    monkeypatch.setattr(backend_module.psutil, "process_iter", lambda _attrs: rows)

    payload = make_collector().processes(limit=10)

    assert payload["top_by_cpu"][0]["name"] == "worker.exe"
    assert payload["top_by_ram"][0]["name"] == "ChatGPT.exe"
    chatgpt = payload["groups_by_ram"][0]
    assert chatgpt["name"] == "ChatGPT.exe"
    assert chatgpt["process_count"] == 2
    assert chatgpt["total_rss_mb"] == 1300.0
    assert chatgpt["total_private_mb"] == 900.0
    assert chatgpt["pids"] == [10, 11]


def test_backend_sensor_detail_is_bounded_and_source_identified():
    payload = make_collector().sensor_detail(limit=1)

    assert payload["available"] is True
    assert payload["source"] == "LibreHardwareMonitor"
    assert payload["count"] == 1
    assert payload["items"][0]["Name"] == "CPU Package"


def test_systemsense_reader_keeps_compact_path_separate_from_backend_detail():
    class FakeSense:
        def summary(self):
            return {"system_health": "good", "memory_percent": 42.7}

        def raw(self, *, category, limit):
            return {"category": category, "limit": limit}

    class FakeBackend:
        def collect(self, *, section, limit):
            return {"section": section, "limit": limit, "detail": "deep"}

    reader = SystemSenseReader(FakeSense(), backend=FakeBackend())

    summary = json.loads(reader.get_system_sense_summary())
    backend = json.loads(
        reader.inspect_raw_system_sense(
            category="backend",
            backend_section="memory",
            limit=25,
        )
    )
    passive = json.loads(reader.inspect_raw_system_sense(category="dynamic", limit=10))

    assert summary == {"system_health": "good", "memory_percent": 42.7}
    assert backend == {"section": "memory", "limit": 25, "detail": "deep"}
    assert passive == {"category": "dynamic", "limit": 10}


def test_backend_section_validation_is_fail_closed():
    with pytest.raises(ValueError, match="section must be one of"):
        make_collector().collect(section="everything-unbounded")
