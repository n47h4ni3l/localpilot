from __future__ import annotations

import json
import io
import sys
from types import SimpleNamespace

import pytest

from localpilot.config import Config, SystemSenseConfig, load_config
from localpilot.audit import AuditLog
from localpilot.agent import LocalPilotAgent
from localpilot.safety import RiskLevel
from localpilot.runtime_worker import RuntimeWorker
from localpilot.systemsense import SystemSense
from localpilot.systemsense_collectors import (
    DriverStoreCollector,
    LibreHardwareMonitorCollector,
    utc_timestamp,
)
from localpilot.tools import registry


class FakeDynamicCollector:
    def __init__(self, cpu: float = 20.0, memory: float = 40.0) -> None:
        self.cpu = cpu
        self.memory = memory

    def collect(self):
        return {
            "captured_at": utc_timestamp(),
            "cpu": {"percent": self.cpu, "frequency_mhz": 4200.0},
            "memory": {"percent": self.memory, "available_gb": 18.0},
            "storage": {
                "io": {"read_mb_s": 1.0, "write_mb_s": 2.0},
                "volumes": [{"device": "C:", "free_percent": 31.0}],
            },
            "network": {"send_mbps": 0.1, "receive_mbps": 0.2},
            "top_processes": [
                {"pid": 99, "name": "competing.exe", "cpu_percent": 17.0, "ram_mb": 800.0}
            ],
        }


class FakePerformanceCollector:
    def collect(self):
        return {
            "gpu": {"engine_utilization_percent": 73.0},
            "processor": {
                "PercentPerformanceLimit": 100,
                "PercentProcessorPerformance": 98,
            },
            "thermal_zones": [],
        }


class FakeSensorCollector:
    def collect(self):
        return {
            "source": "LibreHardwareMonitor",
            "available": True,
            "errors": [],
            "sensors": [
                {
                    "Identifier": "/gpu/0/temperature/0",
                    "Name": "GPU Core",
                    "SensorType": "Temperature",
                    "Value": 71.0,
                    "Min": 40.0,
                    "Max": 72.0,
                    "Parent": "/gpu/0",
                },
                {
                    "Identifier": "/gpu/0/load/0",
                    "Name": "GPU Core",
                    "SensorType": "Load",
                    "Value": 75.0,
                    "Min": 0.0,
                    "Max": 95.0,
                    "Parent": "/gpu/0",
                },
                {
                    "Identifier": "/gpu/0/data/0",
                    "Name": "GPU Memory Used",
                    "SensorType": "SmallData",
                    "Value": 12288.0,
                    "Min": 0.0,
                    "Max": 13000.0,
                    "Parent": "/gpu/0",
                },
            ],
        }


class FakeInventoryCollector:
    def collect(self):
        return {
            "captured_at": "2026-08-28T00:00:00+00:00",
            "available": True,
            "devices": [
                {
                    "DeviceID": r"PCI\VEN_1002&DEV_7550&SUBSYS_0B361002",
                    "HardwareID": [r"PCI\VEN_1002&DEV_7550"],
                    "Name": "GPU",
                    "Service": "amdkmdag",
                    "ConfigManagerErrorCode": 0,
                    "Present": True,
                },
                {
                    "DeviceID": r"USB\VID_1234&PID_5678",
                    "Name": "Disabled USB",
                    "ConfigManagerErrorCode": 22,
                    "Present": True,
                },
                {
                    "DeviceID": r"PCI\VEN_1111&DEV_2222",
                    "Name": "Disconnected",
                    "ConfigManagerErrorCode": 45,
                    "Present": False,
                },
            ],
            "system_drivers": [
                {"Name": "amdkmdag", "State": "Running", "Started": True}
            ],
            "signed_drivers": [
                {
                    "DeviceID": r"PCI\VEN_1002&DEV_7550&SUBSYS_0B361002",
                    "DeviceName": "GPU",
                    "InfName": "oem1.inf",
                    "DriverVersion": "32.0.1",
                },
                {
                    "DeviceID": r"USB\VID_1234&PID_5678",
                    "DeviceName": "Disabled USB",
                    "InfName": "oem2.inf",
                    "DriverVersion": "1.0",
                },
            ],
            "driver_store": {
                "available": True,
                "packages": [
                    {
                        "published_name": "oem1.inf",
                        "original_name": "display.inf",
                        "provider": "AMD",
                        "driver_version": "32.0.1",
                    },
                    {
                        "published_name": "oem9.inf",
                        "original_name": "old-device.inf",
                        "provider": "Vendor",
                        "driver_version": "1.0.0",
                    },
                    {
                        "published_name": "oem10.inf",
                        "original_name": "old-device.inf",
                        "provider": "Vendor",
                        "driver_version": "2.0.0",
                    },
                ],
            },
            "computer_system": [{"Manufacturer": "Example", "Model": "PC"}],
            "motherboard": [{"Manufacturer": "Example", "Product": "Board"}],
            "bios": [{"SMBIOSBIOSVersion": "1.2.3"}],
            "cpu": [{"ProcessorId": "ABC"}],
            "gpu": [{"PNPDeviceID": r"PCI\VEN_1002&DEV_7550"}],
        }


def make_sense(tmp_path, *, cpu: float = 20.0) -> SystemSense:
    return SystemSense(
        SystemSenseConfig(),
        tmp_path,
        psutil_collector=FakeDynamicCollector(cpu=cpu),
        performance_collector=FakePerformanceCollector(),
        sensor_collector=FakeSensorCollector(),
        inventory_collector=FakeInventoryCollector(),
    )


def test_dynamic_collection_derives_compact_health_and_raw_sensor_state(tmp_path):
    sense = make_sense(tmp_path)
    sense.collect_dynamic()
    sense.collect_inventory()

    summary = sense.summary()

    assert summary["system_health"] == "good"
    assert summary["gpu_percent"] == 75.0
    assert summary["max_temperature_c"] == 71.0
    assert summary["thermal_margin_c"] == 24.0
    assert summary["vram_used_mb"] == 12288.0
    assert summary["sensor_source"] == "LibreHardwareMonitor"
    assert summary["background_resource_contention"][0]["name"] == "competing.exe"
    assert "SYSTEMSENSE PASSIVE STATE" in sense.compact_context()
    assert sense.raw(category="sensors")["count"] == 3

    # Raw current state is replaced rather than appended every five seconds;
    # normalized metrics retain the long-term history.
    sense.collect_dynamic()
    with sense.store._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM snapshots WHERE kind='dynamic'"
        ).fetchone()[0]
    assert count == 1


def test_passive_summary_never_collects_when_the_runtime_has_not_sampled(tmp_path):
    dynamic = FakeDynamicCollector()
    calls = []
    original_collect = dynamic.collect

    def collect():
        calls.append(True)
        return original_collect()

    dynamic.collect = collect
    sense = SystemSense(
        SystemSenseConfig(),
        tmp_path,
        psutil_collector=dynamic,
        performance_collector=FakePerformanceCollector(),
        sensor_collector=FakeSensorCollector(),
        inventory_collector=FakeInventoryCollector(),
    )

    passive = sense.summary(collect_if_missing=False)

    assert passive["system_health"] == "unknown"
    assert passive["captured_at"] is None
    assert calls == []
    assert sense.store.latest_snapshot("dynamic") is None

    active = sense.summary()
    assert active["system_health"] == "good"
    assert calls == [True]


def test_passive_context_includes_durable_runtime_and_checkout_evidence(tmp_path):
    audit = AuditLog(tmp_path / "data" / "audit.jsonl")
    audit.write(
        "runtime_lifecycle",
        transition="ready",
        old_pid=None,
        new_pid=901,
        process_started_at="2026-08-29T02:03:04+00:00",
        reason="broker_started",
        return_code=None,
        signal=None,
        request_id=None,
        session_id=None,
        message_id=None,
        affected_requests=[],
        source="broker_startup",
    )
    audit.write(
        "background_worker_cycle_end",
        pid=777,
        sequence=12,
        status="deferred",
        duration_seconds=0.02,
    )
    audit.write(
        "evolve_run_end",
        invocation_id="evolve-1",
        status="candidate_created",
        branch="selfdev/example",
        checks_passed=True,
        summary="Created one isolated candidate.",
    )
    sense = SystemSense(
        SystemSenseConfig(),
        tmp_path / "data",
        project_root=tmp_path,
        psutil_collector=FakeDynamicCollector(),
        performance_collector=FakePerformanceCollector(),
        sensor_collector=FakeSensorCollector(),
        inventory_collector=FakeInventoryCollector(),
    )

    summary = sense.summary(collect_if_missing=False)
    context = sense.compact_context()

    assert summary["runtime"]["current_process"]["pid"] == 901
    assert summary["runtime"]["recent_lifecycle"][0]["source"] == "broker_startup"
    activity = summary["runtime"]["autonomous_activity"]
    assert activity["latest_background_cycle"]["status"] == "deferred"
    assert activity["latest_evolution_run"]["branch"] == "selfdev/example"
    assert activity["learning_boundaries"]["model_weights_changed_by_localpilot"] is False
    assert "\"runtime\"" in context
    assert "\"process_started_at\":\"2026-08-29T02:03:04+00:00\"" in context
    assert "\"ordinary_chat_automatically_persisted_as_learning\":false" in context


def test_inventory_exposes_ids_errors_hidden_devices_and_conservative_driver_classes(tmp_path):
    sense = make_sense(tmp_path)
    inventory = sense.collect_inventory()

    assert inventory["devices"][0]["identifiers"] == {
        "pci_vendor_id": "1002",
        "pci_device_id": "7550",
        "pci_subsystem_id": "0B361002",
    }
    assert inventory["devices"][1]["classification"] == "disabled"
    assert inventory["devices"][1]["identifiers"] == {
        "usb_vendor_id": "1234",
        "usb_product_id": "5678",
    }
    assert inventory["devices"][2]["classification"] == "hidden_or_disconnected"
    assert inventory["drivers"][0]["classification"] == "active_bound"
    assert inventory["drivers"][1]["classification"] == "problematic"

    packages = {row["published_name"]: row for row in inventory["driver_packages"]}
    assert packages["oem1.inf"]["classification"] == "active_bound"
    assert packages["oem9.inf"]["classification"] == "duplicate_older_candidate_requires_review"
    assert packages["oem10.inf"]["classification"] == "orphan_candidate_requires_review"
    assert all(row["safe_to_delete"] is False for row in packages.values())
    assert "never proof" in inventory["summary"]["classification_warning"]

    sense.collect_inventory()
    with sense.store._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM snapshots WHERE kind='inventory'"
        ).fetchone()[0]
    assert count == 1


def test_rolling_baseline_flags_large_anomaly(tmp_path):
    sense = make_sense(tmp_path, cpu=96.0)
    now = utc_timestamp()
    for value in (20.0, 21.0, 19.0, 20.5, 20.0, 21.0):
        sense.store.save_metrics(now, [("cpu.percent", value, "%", "test")])
    sense.collect_dynamic()

    anomalies = sense.summary()["anomalies"]

    assert anomalies[0]["metric"] == "cpu.percent"
    assert anomalies[0]["direction"] == "high"


def test_inference_metrics_and_long_term_correlations_are_observational(tmp_path):
    sense = make_sense(tmp_path)
    for index in range(1, 7):
        dynamic = FakeDynamicCollector(cpu=float(index * 10)).collect()
        payload = {
            "captured_at": f"2026-08-28T00:00:0{index}+00:00",
            "base": dynamic,
            "performance": FakePerformanceCollector().collect(),
            "sensors": {},
            "raw_sensors": {},
            "derived": {},
        }
        sense.store.save_snapshot("dynamic", payload)
        speed = float(70 - index * 5)
        sense.record_inference(
            {
                "phase": "operator",
                "eval_count": 100,
                "eval_duration": int(100 / speed * 1_000_000_000),
                "prompt_eval_count": 50,
                "prompt_eval_duration": 100_000_000,
                "load_duration": 10_000_000,
                "total_duration": 2_000_000_000,
                "context_used_percent": 20.0,
            },
            model="test-model",
        )

    result = sense.correlations()

    cpu = next(row for row in result["correlations"] if row["environment_metric"] == "cpu.percent")
    assert cpu["samples"] == 6
    assert cpu["pearson_r"] < -0.99
    assert "does not establish causality" in result["warning"]
    assert sense.inference_summary()["samples"] == 6


def test_history_and_drill_down_queries_are_allowlisted_and_bounded(tmp_path):
    sense = make_sense(tmp_path)
    sense.collect_dynamic()
    sense.collect_inventory()

    assert sense.history(metric="cpu.percent", limit=50)["samples"] == 1
    assert sense.inventory(section="devices", limit=1)["count"] == 3
    assert len(sense.inventory(section="devices", limit=1)["items"]) == 1
    assert sense.drivers(classification="active_bound")["count"] == 2
    with pytest.raises(ValueError, match="not an exposed"):
        sense.history(metric="sqlite.master")
    with pytest.raises(ValueError, match="section must be"):
        sense.inventory(section="secrets")


def test_registry_exposes_only_read_only_systemsense_surfaces(tmp_path):
    config = Config()
    sense = make_sense(tmp_path / config.agent.data_dir)
    tools = registry(tmp_path, config=config, systemsense=sense)
    names = {
        "get_system_sense_summary",
        "inspect_hardware_inventory",
        "inspect_driver_inventory",
        "get_system_sense_history",
        "get_workload_correlations",
        "inspect_raw_system_sense",
    }
    assert names <= tools.keys()
    assert all(tools[name].risk is RiskLevel.READ_ONLY for name in names)
    assert json.loads(tools["get_system_sense_summary"].fn())["enabled"] is True


def test_systemsense_config_is_bounded_and_uses_separate_database(tmp_path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        "[systemsense]\nsample_interval_seconds=10\nretention_days=90\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.systemsense.enabled is True
    assert config.systemsense.sample_interval_seconds == 10
    assert config.systemsense.retention_days == 90

    path.write_text(
        '[systemsense]\ndatabase="chat.sqlite3"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="remain separate"):
        load_config(path)

    path.write_text(
        "[systemsense]\nsample_interval_seconds=0.1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="between 1 and 300"):
        load_config(path)


def test_driver_store_uses_bounded_native_argv_without_shell(monkeypatch):
    captured = {}
    monkeypatch.setattr("localpilot.systemsense_collectors.os.name", "nt")

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="Published Name,Original Name,Provider Name,Driver Version\noem1.inf,test.inf,Vendor,1.2.3\n",
            stderr="",
        )

    monkeypatch.setattr("localpilot.systemsense_collectors.subprocess.run", fake_run)
    result = DriverStoreCollector().collect()

    assert captured["argv"] == [
        "pnputil.exe",
        "/enum-drivers",
        "/files",
        "/format",
        "csv",
    ]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["timeout"] == 60
    assert result["packages"][0]["published_name"] == "oem1.inf"


def test_librehardwaremonitor_collector_falls_back_to_open_namespace():
    class FakeWmi:
        def query(self, namespace, class_name, properties):
            if "Libre" in namespace:
                raise RuntimeError("not installed")
            return [{"Name": "CPU", "SensorType": "Temperature", "Value": 50}]

    result = LibreHardwareMonitorCollector(FakeWmi()).collect()

    assert result["available"] is True
    assert result["source"] == "OpenHardwareMonitor"
    assert result["sensors"][0]["Value"] == 50


def test_runtime_worker_owns_passive_systemsense_lifecycle(tmp_path, monkeypatch):
    worker = RuntimeWorker(tmp_path)
    calls = []
    worker.systemsense = SimpleNamespace(
        start=lambda: calls.append("start"),
        stop=lambda: calls.append("stop"),
    )
    worker._write = lambda payload: None
    worker._start_background_reader = lambda: None
    monkeypatch.setattr("localpilot.runtime_worker.sys.stdin", io.StringIO(""))

    worker.run()

    assert calls == ["start", "stop"]


def test_compact_environment_state_is_transient_model_context(tmp_path, monkeypatch):
    agent = LocalPilotAgent(Config(), tmp_path)
    dynamic = {
        "captured_at": utc_timestamp(),
        "base": FakeDynamicCollector().collect(),
        "performance": FakePerformanceCollector().collect(),
        "sensors": {"available": False, "source": "test"},
        "raw_sensors": {},
        "derived": {"gpu_utilization_percent": 30.0},
    }
    agent.systemsense.store.replace_latest_snapshot("dynamic", dynamic)
    observed = []

    def fake_chat(**kwargs):
        observed.append([dict(message) for message in kwargs["messages"]])
        return iter(
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="READY", thinking="", tool_calls=[]
                    )
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))

    assert agent.ask("Say exactly READY.") == "READY"
    assert any(
        "SYSTEMSENSE PASSIVE STATE" in str(message.get("content"))
        for message in observed[0]
    )
    assert "SYSTEMSENSE PASSIVE STATE" not in str(agent.messages)
