from __future__ import annotations

import json

from localpilot.config import SystemSenseConfig
from localpilot.systemsense import SystemSense
from localpilot.systemsense_views import normalize_hardware_sensors
from localpilot.tools.systemsense import SystemSenseReader


class _BaseCollector:
    def collect(self):
        return {
            "captured_at": "2026-09-17T00:00:00+00:00",
            "cpu": {"percent": 14.0, "frequency_mhz": 4050.0},
            "memory": {"percent": 44.0, "available_gb": 17.7},
            "storage": {
                "io": {"read_mb_s": 1.0, "write_mb_s": 2.0},
                "volumes": [
                    {"device": "C:", "free_percent": 31.0},
                    {"device": "D:", "free_percent": 8.0},
                ],
            },
            "network": {"send_mbps": 0.1, "receive_mbps": 0.2},
            "top_processes": [],
        }


class _PerformanceCollector:
    def collect(self):
        return {
            "source": "windows-wmi",
            "available": True,
            "gpu": {"engine_utilization_percent": 5.0},
            "processor": {
                "PercentPerformanceLimit": 100,
                "PercentProcessorPerformance": 100,
            },
            "power_plan": {},
            "thermal_zones": [],
            "errors": [],
        }


def _sensor(identifier, name, sensor_type, value, parent, hardware_type, hardware_name):
    return {
        "Identifier": identifier,
        "Name": name,
        "SensorType": sensor_type,
        "Value": value,
        "Min": value,
        "Max": value,
        "Parent": parent,
        "HardwareType": hardware_type,
        "HardwareName": hardware_name,
    }


class _SensorCollector:
    def collect(self):
        return {
            "source": "LibreHardwareMonitorLib",
            "provider_version": "0.9.6.0",
            "available": True,
            "errors": [],
            "sensors": [
                _sensor("/amdcpu/0/temperature/0", "CPU Package", "Temperature", 60.0, "/amdcpu/0", "Cpu", "AMD Ryzen"),
                _sensor("/gpu-amd/0/temperature/0", "GPU Core", "Temperature", 50.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/temperature/1", "GPU Hot Spot", "Temperature", 70.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/temperature/2", "GPU Memory", "Temperature", 60.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/memory/dimm/1/temperature/0", "DIMM #1", "Temperature", 40.0, "/memory/dimm/1", "Memory", "Corsair"),
                _sensor("/memory/dimm/1/temperature/3", "Thermal Sensor High Limit", "Temperature", 95.0, "/memory/dimm/1", "Memory", "Corsair"),
                _sensor("/ssd/0/temperature/0", "Temperature", "Temperature", 45.0, "/ssd/0", "Storage", "SSD"),
                _sensor("/lpc/0/temperature/0", "Temperature #1", "Temperature", 35.0, "/lpc/0", "SuperIO", "Nuvoton"),
                _sensor("/gpu-amd/0/load/0", "D3D 3D", "Load", 12.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/smalldata/2", "GPU Memory Total", "SmallData", 16364.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/smalldata/3", "D3D Dedicated Memory Used", "SmallData", 2336.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/smalldata/4", "D3D Dedicated Memory Free", "SmallData", 13958.0, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
                _sensor("/gpu-amd/0/smalldata/6", "D3D Shared Memory Used", "SmallData", 220.6, "/gpu-amd/0", "GpuAmd", "AMD Radeon"),
            ],
        }


class _InventoryCollector:
    def collect(self):
        return {"captured_at": "2026-09-17T00:00:00+00:00", "available": True}


def _sense(tmp_path):
    return SystemSense(
        SystemSenseConfig(),
        tmp_path,
        psutil_collector=_BaseCollector(),
        performance_collector=_PerformanceCollector(),
        sensor_collector=_SensorCollector(),
        inventory_collector=_InventoryCollector(),
    )


def test_normalization_is_presentation_only_and_classifies_live_like_vram():
    raw = _SensorCollector().collect()
    normalized = normalize_hardware_sensors(raw)

    assert normalized["temperatures"]["system_average_c"] == 46.0
    assert normalized["temperatures"]["system_peak_c"] == 70.0
    assert normalized["temperatures"]["components"]["gpu"]["representative_c"] == 50.0
    assert normalized["temperatures"]["components"]["memory"]["representative_c"] == 40.0
    assert normalized["vram"] == {
        "used_mb": 2336.0,
        "free_mb": 13958.0,
        "total_mb": 16364.0,
        "shared_used_mb": 220.6,
        "utilization_percent": 14.3,
        "adapter_count": 1,
    }


def test_desktop_summary_is_simple_but_model_surface_returns_raw_truth(tmp_path):
    sense = _sense(tmp_path)
    sense.collect_dynamic()

    presentation = sense.summary(collect_if_missing=False)
    assert presentation["presentation_only"] is True
    assert presentation["system_average_temperature_c"] == 46.0
    assert presentation["system_peak_temperature_c"] == 70.0
    assert presentation["max_temperature_c"] == 46.0
    assert presentation["thermal_state"] == "moderate"
    assert presentation["vram_used_mb"] == 2336.0
    assert presentation["vram_total_mb"] == 16364.0

    reader = SystemSenseReader(sense)
    truth = json.loads(reader.get_system_sense_summary())
    assert truth["hardware_truth"]["sensor_provider"]["sensor_count"] == 13
    assert truth["hardware_truth"]["sensors"][1]["Name"] == "GPU Core"
    assert truth["hardware_truth"]["sensors"][2]["Name"] == "GPU Hot Spot"
    assert "presentation_only" not in truth
    assert "system_average_temperature_c" not in truth

    context = sense.compact_context()
    assert "SYSTEMSENSE PASSIVE STATE" in context
    assert "RAW TRUTH" in context
    assert '"minimum_volume_free_percent":8.0' in context
    assert "system_average_temperature_c" not in context
    assert "presentation_only" not in context
