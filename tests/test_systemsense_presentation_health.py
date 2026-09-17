from __future__ import annotations

from localpilot.systemsense_presentation_health import harden_presentation_summary


class _Store:
    def __init__(self, dynamic):
        self.dynamic = dynamic

    def latest_snapshot(self, kind):
        return self.dynamic if kind == "dynamic" else None


class _Sense:
    def __init__(self, dynamic):
        self.store = _Store(dynamic)


def _sense_with_provider(*, available=True, errors=None):
    return _Sense(
        {
            "raw_sensors": {
                "source": "LibreHardwareMonitorLib",
                "provider_version": "0.9.6.0",
                "available": available,
                "errors": list(errors or []),
                "sensors": [
                    {
                        "SensorType": "Temperature",
                        "Name": "CPU Package",
                        "Value": 48.0,
                    },
                    {
                        "SensorType": "Temperature",
                        "Name": "GPU Core",
                        "Value": 42.0,
                    },
                    {
                        "SensorType": "Load",
                        "Name": "GPU Core",
                        "Value": 10.0,
                    },
                ],
            }
        }
    )


def test_large_idle_baseline_deviations_are_context_not_automatic_warnings():
    sense = _sense_with_provider()
    state = {
        "enabled": True,
        "system_health": "degraded",
        "compute_pressure": "low",
        "memory_pressure": "low",
        "thermal_state": "normal",
        "throttling_detected": False,
        "device_problems": 0,
        "anomalies": [
            {
                "metric": "cpu.percent",
                "current": 12.5,
                "baseline_median": 3.6,
                "robust_z": 8.2,
                "direction": "high",
            },
            {
                "metric": "thermal.max_c",
                "current": 68.0,
                "baseline_median": 50.0,
                "robust_z": 7.5,
                "direction": "high",
            },
            {
                "metric": "vram.used_mb",
                "current": 1829.0,
                "baseline_median": 1615.4,
                "robust_z": 6.0,
                "direction": "high",
            },
        ],
    }

    result = harden_presentation_summary(sense, state)

    assert result["system_health"] == "good"
    assert result["anomalies"] == []
    assert len(result["baseline_signals"]) == 3
    assert all(row["attention_relevant"] is False for row in result["baseline_signals"])
    assert result["health_reasons"] == []


def test_baseline_deviation_only_warns_when_absolute_pressure_is_also_real():
    sense = _sense_with_provider()
    state = {
        "enabled": True,
        "system_health": "good",
        "compute_pressure": "moderate",
        "memory_pressure": "low",
        "thermal_state": "moderate",
        "throttling_detected": False,
        "device_problems": 0,
        "anomalies": [
            {
                "metric": "cpu.percent",
                "current": 78.0,
                "baseline_median": 12.0,
                "robust_z": 7.1,
                "direction": "high",
            },
            {
                "metric": "thermal.max_c",
                "current": 76.0,
                "baseline_median": 48.0,
                "robust_z": 6.2,
                "direction": "high",
            },
        ],
    }

    result = harden_presentation_summary(sense, state)

    assert result["system_health"] == "degraded"
    assert len(result["anomalies"]) == 2
    assert all(row["attention_relevant"] is True for row in result["anomalies"])
    assert "unusual readings that also cross pressure thresholds" in result["health_reasons"]


def test_provider_readiness_is_explicit_and_null_cards_are_not_serialized_as_zero():
    sense = _sense_with_provider()
    state = {
        "enabled": True,
        "system_health": "good",
        "compute_pressure": "low",
        "memory_pressure": "low",
        "thermal_state": "unknown",
        "throttling_detected": False,
        "device_problems": 0,
        "anomalies": [],
        "cpu_percent": None,
        "gpu_percent": None,
        "max_temperature_c": None,
        "vram_used_mb": None,
    }

    result = harden_presentation_summary(sense, state)

    assert result["sensor_provider_health"] == {
        "status": "ready",
        "ready": True,
        "available": True,
        "source": "LibreHardwareMonitorLib",
        "provider_version": "0.9.6.0",
        "sensor_count": 3,
        "live_temperature_sensor_count": 2,
        "errors": [],
    }
    # The current frontend maps undefined to an em dash but Number(null) to 0.
    # Omitting null numeric presentation fields prevents false 0° / 0 MB cards.
    assert "cpu_percent" not in result
    assert "gpu_percent" not in result
    assert "max_temperature_c" not in result
    assert "vram_used_mb" not in result


def test_provider_health_reports_unavailable_with_diagnostics():
    sense = _sense_with_provider(available=False, errors=["snapshot-timeout"])
    state = {
        "enabled": True,
        "system_health": "good",
        "compute_pressure": "low",
        "memory_pressure": "low",
        "thermal_state": "unknown",
        "throttling_detected": False,
        "device_problems": 0,
        "anomalies": [],
    }

    result = harden_presentation_summary(sense, state)

    assert result["sensor_provider_health"]["status"] == "unavailable"
    assert result["sensor_provider_health"]["ready"] is False
    assert result["sensor_provider_health"]["errors"] == ["snapshot-timeout"]
