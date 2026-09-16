from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from typing import Any


_PRESENTATION_COMPONENTS = ("cpu", "gpu", "memory", "storage", "motherboard")
_THRESHOLD_TEMPERATURE_TOKENS = (
    "critical limit",
    "high limit",
    "low limit",
    "thermal limit",
    "temperature limit",
    "trip point",
    "threshold",
    "shutdown",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


def _sensor_identity(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("Identifier", "Parent", "HardwareType", "HardwareName", "Name")
    ).casefold()


def _component(row: dict[str, Any]) -> str:
    hardware_type = str(row.get("HardwareType") or "").casefold()
    identity = _sensor_identity(row)
    if hardware_type in {"gpuamd", "gpunvidia", "gpuintel", "gpu"} or "/gpu" in identity:
        return "gpu"
    if hardware_type == "cpu" or "/cpu" in identity or "/amdcpu" in identity:
        return "cpu"
    if hardware_type == "memory" or "/memory" in identity or "/ram" in identity:
        return "memory"
    if hardware_type == "storage" or any(token in identity for token in ("/ssd", "/hdd")):
        return "storage"
    if hardware_type in {"motherboard", "superio"} or "/lpc" in identity:
        return "motherboard"
    return "other"


def _current_temperature(row: dict[str, Any]) -> float | None:
    if str(row.get("SensorType") or "").casefold() != "temperature":
        return None
    name = str(row.get("Name") or "").casefold()
    if any(token in name for token in _THRESHOLD_TEMPERATURE_TOKENS):
        return None
    value = _finite(row.get("Value"))
    # Zero-valued motherboard placeholders are common; real PC component
    # temperatures below 1 C are not useful for this presentation snapshot.
    if value is None or value < 1.0 or value > 150.0:
        return None
    return value


def _representative_temperature(component: str, rows: list[dict[str, Any]]) -> float | None:
    values = [(row, _current_temperature(row)) for row in rows]
    values = [(row, value) for row, value in values if value is not None]
    if not values:
        return None

    if component == "cpu":
        priorities = (
            "tctl/tdie",
            "cpu package",
            "package",
            "cpu (tctl",
            "cpu",
        )
        for token in priorities:
            matches = [
                value
                for row, value in values
                if token in str(row.get("Name") or "").casefold()
            ]
            if matches:
                return max(matches)
        return max(value for _row, value in values)

    if component == "gpu":
        core = [
            value
            for row, value in values
            if "core" in str(row.get("Name") or "").casefold()
            and "hot" not in str(row.get("Name") or "").casefold()
        ]
        if core:
            return max(core)
        non_hotspot = [
            value
            for row, value in values
            if "hot spot" not in str(row.get("Name") or "").casefold()
            and "hotspot" not in str(row.get("Name") or "").casefold()
        ]
        return max(non_hotspot or [value for _row, value in values])

    return statistics.fmean(value for _row, value in values)


def _sensor_value_mb(row: dict[str, Any]) -> float | None:
    value = _finite(row.get("Value"))
    if value is None or value < 0:
        return None
    sensor_type = str(row.get("SensorType") or "").casefold()
    if sensor_type == "smalldata":
        return value
    if sensor_type == "data":
        return value * 1024.0
    return None


def _adapter_key(row: dict[str, Any]) -> str:
    parent = str(row.get("Parent") or "").strip()
    if parent:
        return parent.casefold()
    identifier = str(row.get("Identifier") or "").strip()
    for token in ("/smalldata/", "/data/"):
        if token in identifier.casefold():
            return identifier[: identifier.casefold().index(token)].casefold()
    return identifier.casefold() or "gpu"


def _vram_kind(row: dict[str, Any]) -> str | None:
    if _component(row) != "gpu":
        return None
    name = str(row.get("Name") or "").casefold()
    if "shared" in name and "used" in name:
        return "shared_used_mb"
    if "total" in name and ("memory" in name or "vram" in name):
        return "total_mb"
    if "free" in name and ("memory" in name or "vram" in name):
        return "free_mb"
    if "used" in name and ("memory" in name or "vram" in name):
        return "used_mb"
    return None


def normalize_hardware_sensors(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive presentation metrics without replacing the canonical raw payload."""
    readings = [row for row in (payload.get("sensors") or []) if isinstance(row, dict)]
    by_component: dict[str, list[dict[str, Any]]] = {
        key: [] for key in (*_PRESENTATION_COMPONENTS, "other")
    }
    for row in readings:
        by_component[_component(row)].append(row)

    temperature_rows = [
        (component, row, value)
        for component, rows in by_component.items()
        for row in rows
        if (value := _current_temperature(row)) is not None
    ]
    component_temperatures: dict[str, dict[str, Any]] = {}
    representatives: list[float] = []
    for component in _PRESENTATION_COMPONENTS:
        rows = by_component[component]
        representative = _representative_temperature(component, rows)
        current_values = [
            value for row in rows if (value := _current_temperature(row)) is not None
        ]
        component_temperatures[component] = {
            "representative_c": _round(representative),
            "peak_c": _round(max(current_values), 1) if current_values else None,
            "sensor_count": len(current_values),
        }
        if representative is not None:
            representatives.append(representative)

    system_average = statistics.fmean(representatives) if representatives else None
    system_peak = max(
        (value for _component_name, _row, value in temperature_rows), default=None
    )

    adapters: dict[str, dict[str, list[float]]] = {}
    for row in readings:
        kind = _vram_kind(row)
        if kind is None:
            continue
        value_mb = _sensor_value_mb(row)
        if value_mb is None:
            continue
        adapters.setdefault(_adapter_key(row), {}).setdefault(kind, []).append(value_mb)

    adapter_values: list[dict[str, float | None]] = []
    for values in adapters.values():
        current = {
            key: max(values.get(key) or [], default=None)
            for key in ("used_mb", "free_mb", "total_mb", "shared_used_mb")
        }
        if (
            current["used_mb"] is None
            and current["total_mb"] is not None
            and current["free_mb"] is not None
        ):
            current["used_mb"] = max(0.0, current["total_mb"] - current["free_mb"])
        if (
            current["total_mb"] is None
            and current["used_mb"] is not None
            and current["free_mb"] is not None
        ):
            current["total_mb"] = current["used_mb"] + current["free_mb"]
        adapter_values.append(current)

    def _sum_field(name: str) -> float | None:
        values = [float(row[name]) for row in adapter_values if row.get(name) is not None]
        return sum(values) if values else None

    used_mb = _sum_field("used_mb")
    free_mb = _sum_field("free_mb")
    total_mb = _sum_field("total_mb")
    shared_used_mb = _sum_field("shared_used_mb")
    utilization = (
        100.0 * used_mb / total_mb
        if used_mb is not None and total_mb is not None and total_mb > 0
        else None
    )

    gpu_loads = [
        value
        for row in by_component["gpu"]
        if str(row.get("SensorType") or "").casefold() == "load"
        and (value := _finite(row.get("Value"))) is not None
        and "memory" not in str(row.get("Name") or "").casefold()
    ]

    return {
        "available": bool(payload.get("available")),
        "source": payload.get("source"),
        "provider_version": payload.get("provider_version"),
        "errors": list(payload.get("errors") or []),
        "sensor_count": len(readings),
        "temperatures": {
            "system_average_c": _round(system_average),
            "system_peak_c": _round(system_peak),
            "components": component_temperatures,
        },
        "vram": {
            "used_mb": _round(used_mb),
            "free_mb": _round(free_mb),
            "total_mb": _round(total_mb),
            "shared_used_mb": _round(shared_used_mb),
            "utilization_percent": _round(utilization),
            "adapter_count": len(adapter_values),
        },
        "gpu_load_percent": _round(max(gpu_loads), 2) if gpu_loads else None,
    }


def _normalized_sensor_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility summary used only for derived storage/history and GUI state."""
    normalized = normalize_hardware_sensors(payload)
    readings = [row for row in (payload.get("sensors") or []) if isinstance(row, dict)]
    grouped: dict[str, list[dict[str, Any]]] = {
        "temperatures": [],
        "fans": [],
        "loads": [],
        "power": [],
        "clocks": [],
        "data": [],
        "small_data": [],
    }
    mapping = {
        "temperature": "temperatures",
        "fan": "fans",
        "load": "loads",
        "power": "power",
        "clock": "clocks",
        "data": "data",
        "smalldata": "small_data",
    }
    for row in readings:
        value = _finite(row.get("Value"))
        group = mapping.get(str(row.get("SensorType") or "").casefold())
        if group is None or value is None:
            continue
        grouped[group].append(
            {
                "id": row.get("Identifier"),
                "name": row.get("Name"),
                "value": value,
                "min": _finite(row.get("Min")),
                "max": _finite(row.get("Max")),
                "parent": row.get("Parent"),
                "hardware_name": row.get("HardwareName"),
                "hardware_type": row.get("HardwareType"),
            }
        )
    return {
        "available": normalized["available"],
        "source": normalized["source"],
        "provider_version": normalized["provider_version"],
        "errors": normalized["errors"],
        **grouped,
        "temperature_components": normalized["temperatures"]["components"],
        "system_average_temperature_c": normalized["temperatures"]["system_average_c"],
        "max_temperature_c": normalized["temperatures"]["system_peak_c"],
        "gpu_load_percent": normalized["gpu_load_percent"],
        "vram_used_mb": normalized["vram"]["used_mb"],
        "vram_free_mb": normalized["vram"]["free_mb"],
        "vram_total_mb": normalized["vram"]["total_mb"],
        "vram_shared_used_mb": normalized["vram"]["shared_used_mb"],
        "vram_utilization_percent": normalized["vram"]["utilization_percent"],
    }


def _thermal_state(peak_c: float | None) -> str:
    if peak_c is None:
        return "unknown"
    if peak_c >= 95:
        return "critical"
    if peak_c >= 85:
        return "high"
    if peak_c >= 70:
        return "moderate"
    return "normal"


def build_user_snapshot(
    systemsense: Any,
    *,
    collect_if_missing: bool = True,
    legacy_summary: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the deliberately simplified presentation view for the desktop UI."""
    if legacy_summary is None:
        legacy_summary = getattr(type(systemsense), "_pre_truth_views_summary", None)
    if legacy_summary is None:
        raise RuntimeError("SystemSense presentation view is not installed")

    state = dict(legacy_summary(systemsense, collect_if_missing=collect_if_missing))
    dynamic = systemsense.store.latest_snapshot("dynamic") or {}
    raw_sensors = dynamic.get("raw_sensors") or {}
    normalized = normalize_hardware_sensors(raw_sensors)
    temperatures = normalized["temperatures"]
    vram = normalized["vram"]
    average_c = temperatures["system_average_c"]
    peak_c = temperatures["system_peak_c"]

    state.update(
        {
            "presentation_only": True,
            "presentation_source": "raw_systemsense_truth",
            "system_average_temperature_c": average_c,
            "system_peak_temperature_c": peak_c,
            # Compatibility field consumed by the current desktop card. It is
            # intentionally the friendly system-average value, not a max sensor.
            "max_temperature_c": average_c,
            "thermal_state": _thermal_state(peak_c),
            "thermal_margin_c": round(95.0 - peak_c, 1) if peak_c is not None else None,
            "temperature_components": temperatures["components"],
            "vram_used_mb": vram["used_mb"],
            "vram_free_mb": vram["free_mb"],
            "vram_total_mb": vram["total_mb"],
            "vram_shared_used_mb": vram["shared_used_mb"],
            "vram_utilization_percent": vram["utilization_percent"],
        }
    )
    return state


def build_agent_truth(systemsense: Any, *, collect_if_missing: bool = True) -> dict[str, Any]:
    """Expose canonical collector output to LocalPilot without presentation normalization."""
    if not systemsense.enabled:
        return {
            "enabled": False,
            "hardware_truth": None,
            "runtime": systemsense.runtime_evidence(),
        }
    dynamic = (
        systemsense._ensure_dynamic()
        if collect_if_missing
        else systemsense.store.latest_snapshot("dynamic")
    ) or {}
    raw_sensors = dynamic.get("raw_sensors") or {}
    sensors = list(raw_sensors.get("sensors") or [])
    return {
        "enabled": True,
        "captured_at": dynamic.get("captured_at"),
        "hardware_truth": {
            "base": dynamic.get("base") or {},
            "windows_performance": dynamic.get("performance") or {},
            "sensor_provider": {
                "source": raw_sensors.get("source"),
                "provider_version": raw_sensors.get("provider_version"),
                "available": bool(raw_sensors.get("available")),
                "errors": list(raw_sensors.get("errors") or []),
                "sensor_count": len(sensors),
            },
            "sensors": sensors,
        },
        "inference": systemsense.inference_summary(),
        "runtime": systemsense.runtime_evidence(),
        "evidence_policy": (
            "Hardware decisions must use these raw collector values or inspect_raw_system_sense. "
            "The desktop SystemSense snapshot is presentation-only and is not model evidence."
        ),
    }


def _raw_compact_context(systemsense: Any) -> str:
    if not systemsense.enabled or not systemsense.config.compact_context_enabled:
        return ""
    dynamic = systemsense.store.latest_snapshot("dynamic") or {}
    runtime = systemsense.runtime_evidence()
    if not dynamic and runtime is None:
        return ""
    raw_sensors = dynamic.get("raw_sensors") or {}
    base = dynamic.get("base") or {}
    volumes = (
        ((base.get("storage") or {}).get("volumes") or [])
        if isinstance(base, dict)
        else []
    )
    free_values = [
        value
        for row in volumes
        if isinstance(row, dict)
        and (value := _finite(row.get("free_percent"))) is not None
    ]
    compact = {
        "captured_at": dynamic.get("captured_at"),
        "raw_collectors": {
            "psutil": base,
            "windows_performance": dynamic.get("performance") or {},
            "hardware_provider": {
                "source": raw_sensors.get("source"),
                "provider_version": raw_sensors.get("provider_version"),
                "available": bool(raw_sensors.get("available")),
                "errors": list(raw_sensors.get("errors") or []),
                "sensor_count": len(raw_sensors.get("sensors") or []),
            },
        },
        # Transparent deterministic index from the raw volume rows. It is not
        # part of the desktop presentation normalization and never replaces the
        # underlying per-volume truth above.
        "minimum_volume_free_percent": min(free_values) if free_values else None,
        "inference": systemsense.inference_summary(),
        "runtime": runtime,
    }
    return (
        "SYSTEMSENSE PASSIVE STATE — RAW TRUTH (read-only current collector output; no user-facing normalization):\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\nFor any hardware-specific conclusion or action, inspect the raw SystemSense sensor/inventory surfaces first. "
        "Never use the desktop presentation snapshot as model evidence."
    )


def install_systemsense_truth_views() -> None:
    """Separate raw agent evidence from normalized desktop presentation."""
    from localpilot.systemsense import SystemSense

    if getattr(SystemSense, "_truth_views_installed", False):
        return

    legacy_summary = SystemSense.summary
    SystemSense._pre_truth_views_summary = legacy_summary
    SystemSense._sensor_summary = staticmethod(_normalized_sensor_summary)

    def presentation_summary(
        self: Any, *, collect_if_missing: bool = True
    ) -> dict[str, Any]:
        return build_user_snapshot(
            self,
            collect_if_missing=collect_if_missing,
            legacy_summary=legacy_summary,
        )

    SystemSense.summary = presentation_summary
    SystemSense.compact_context = _raw_compact_context
    SystemSense.agent_truth = build_agent_truth
    SystemSense._truth_views_installed = True
