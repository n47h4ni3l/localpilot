from __future__ import annotations

import math
from typing import Any, Callable


_NULLABLE_NUMERIC_PRESENTATION_FIELDS = (
    "cpu_percent",
    "gpu_percent",
    "memory_percent",
    "minimum_volume_free_percent",
    "max_temperature_c",
    "system_average_temperature_c",
    "system_peak_temperature_c",
    "thermal_margin_c",
    "vram_used_mb",
    "vram_free_mb",
    "vram_total_mb",
    "vram_shared_used_mb",
    "vram_utilization_percent",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _attention_relevant_anomaly(anomaly: dict[str, Any]) -> bool:
    """Return whether a baseline deviation also represents current pressure.

    A rolling baseline answers "is this unusual for this machine?"; it does not
    answer "is this unhealthy?". Idle-heavy baselines can otherwise turn normal
    foreground work into a warning. Only deviations that also cross an absolute
    pressure floor are allowed to influence the human-facing health state.
    """
    metric = str(anomaly.get("metric") or "")
    direction = str(anomaly.get("direction") or "")
    current = _finite(anomaly.get("current"))
    if current is None:
        return False

    if metric in {"cpu.percent", "gpu.utilization_percent", "memory.percent"}:
        return direction == "high" and current >= 65.0
    if metric == "memory.available_gb":
        return direction == "low" and current <= 2.0
    if metric == "thermal.max_c":
        return direction == "high" and current >= 70.0

    # Frequency, network/storage throughput and absolute VRAM-used values can be
    # highly workload-dependent. Without a corresponding capacity/limit signal,
    # being far from baseline is context, not evidence of a problem.
    return False


def _provider_health(systemsense: Any) -> dict[str, Any]:
    dynamic = systemsense.store.latest_snapshot("dynamic") or {}
    raw = dynamic.get("raw_sensors") or {}
    rows = [row for row in (raw.get("sensors") or []) if isinstance(row, dict)]
    live_temperatures = 0
    for row in rows:
        if str(row.get("SensorType") or "").casefold() != "temperature":
            continue
        value = _finite(row.get("Value"))
        if value is not None and 0.0 < value < 200.0:
            live_temperatures += 1

    errors = [str(item) for item in (raw.get("errors") or [])]
    available = bool(raw.get("available"))
    if available and rows:
        status = "ready_with_warnings" if errors else "ready"
    elif errors:
        status = "unavailable"
    else:
        status = "waiting"
    return {
        "status": status,
        "ready": status in {"ready", "ready_with_warnings"},
        "available": available,
        "source": raw.get("source"),
        "provider_version": raw.get("provider_version"),
        "sensor_count": len(rows),
        "live_temperature_sensor_count": live_temperatures,
        "errors": errors,
    }


def _strip_null_numeric_fields(output: dict[str, Any]) -> None:
    for key in _NULLABLE_NUMERIC_PRESENTATION_FIELDS:
        if output.get(key) is None:
            output.pop(key, None)


def harden_presentation_summary(systemsense: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Make the human snapshot conservative without changing model evidence."""
    output = dict(state)
    output["sensor_provider_health"] = _provider_health(systemsense)

    # Preserve the existing no-sample contract. A passive summary must remain
    # unknown until the runtime has actually collected a dynamic sample; the
    # presentation wrapper must never turn "no evidence yet" into "healthy".
    if not output.get("captured_at"):
        output.setdefault("baseline_signals", [])
        output.setdefault("health_reasons", [])
        _strip_null_numeric_fields(output)
        return output

    anomalies = [dict(row) for row in (output.get("anomalies") or []) if isinstance(row, dict)]
    baseline_signals: list[dict[str, Any]] = []
    attention_anomalies: list[dict[str, Any]] = []
    for anomaly in anomalies:
        relevant = _attention_relevant_anomaly(anomaly)
        anomaly["attention_relevant"] = relevant
        if relevant:
            attention_anomalies.append(anomaly)
        else:
            baseline_signals.append(anomaly)

    # The existing UI renders `anomalies` as warnings. Keep informational
    # baseline deviations separately so ordinary foreground work does not look
    # like a fault simply because the rolling baseline was learned while idle.
    output["anomalies"] = attention_anomalies
    output["baseline_signals"] = baseline_signals

    compute_pressure = str(output.get("compute_pressure") or "unknown")
    memory_pressure = str(output.get("memory_pressure") or "unknown")
    thermal_state = str(output.get("thermal_state") or "unknown")
    throttling = bool(output.get("throttling_detected"))
    device_problems = int(_finite(output.get("device_problems")) or 0)

    critical = (
        compute_pressure == "critical"
        or memory_pressure == "critical"
        or thermal_state == "critical"
    )
    reasons: list[str] = []
    if throttling:
        reasons.append("processor performance limiting")
    if device_problems:
        reasons.append(f"{device_problems} device problem" + ("s" if device_problems != 1 else ""))
    if compute_pressure in {"high", "critical"}:
        reasons.append(f"{compute_pressure} compute pressure")
    if memory_pressure in {"high", "critical"}:
        reasons.append(f"{memory_pressure} memory pressure")
    if thermal_state in {"high", "critical"}:
        reasons.append(f"{thermal_state} thermal state")
    if attention_anomalies:
        reasons.append("unusual readings that also cross pressure thresholds")

    degraded = bool(
        critical
        or throttling
        or device_problems
        or compute_pressure == "high"
        or memory_pressure == "high"
        or thermal_state == "high"
        or attention_anomalies
    )
    output["system_health"] = "critical" if critical else "degraded" if degraded else "good"
    output["health_reasons"] = reasons

    # app.js correctly renders undefined numeric fields as "—", but JavaScript
    # Number(null) is 0. Until every client is upgraded, never serialize null for
    # the numeric presentation fields consumed by the glance cards.
    _strip_null_numeric_fields(output)

    return output


def install_systemsense_presentation_health() -> None:
    """Wrap the existing presentation-only summary with conservative health semantics."""
    from localpilot.systemsense import SystemSense

    if getattr(SystemSense, "_presentation_health_installed", False):
        return
    prior_summary: Callable[..., dict[str, Any]] = SystemSense.summary

    def presentation_summary(self: Any, *, collect_if_missing: bool = True) -> dict[str, Any]:
        state = prior_summary(self, collect_if_missing=collect_if_missing)
        if not isinstance(state, dict) or not state.get("enabled"):
            return state
        return harden_presentation_summary(self, state)

    SystemSense.summary = presentation_summary
    SystemSense._presentation_health_installed = True
