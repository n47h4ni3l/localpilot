from __future__ import annotations

import json
from typing import Any

from localpilot.systemsense import SystemSense


DIAGNOSIS_SCOPES = ("signals", "thermal", "memory", "gpu", "cpu", "storage")

_SCOPE_ALIASES = {
    "": "signals",
    "signals": "signals",
    "signal": "signals",
    "temperature": "thermal",
    "temp": "thermal",
    "thermal": "thermal",
    "memory": "memory",
    "ram": "memory",
    "gpu": "gpu",
    "vram": "gpu",
    "cpu": "cpu",
    "storage": "storage",
    "disk": "storage",
}

_SCOPE_METRICS: dict[str, tuple[str, ...]] = {
    "signals": (
        "cpu.percent",
        "cpu.frequency_mhz",
        "memory.percent",
        "memory.available_gb",
        "storage.read_mb_s",
        "storage.write_mb_s",
        "network.send_mbps",
        "network.receive_mbps",
        "gpu.utilization_percent",
        "thermal.max_c",
        "vram.used_mb",
        "processor.performance_limit_percent",
        "processor.performance_percent",
    ),
    "thermal": (
        "thermal.max_c",
        "cpu.percent",
        "cpu.frequency_mhz",
        "gpu.utilization_percent",
    ),
    "memory": (
        "memory.percent",
        "memory.available_gb",
        "vram.used_mb",
    ),
    "gpu": (
        "gpu.utilization_percent",
        "vram.used_mb",
        "thermal.max_c",
    ),
    "cpu": (
        "cpu.percent",
        "cpu.frequency_mhz",
        "thermal.max_c",
        "processor.performance_limit_percent",
        "processor.performance_percent",
    ),
    "storage": (
        "storage.read_mb_s",
        "storage.write_mb_s",
        "memory.percent",
    ),
}


def normalize_diagnosis_scope(value: str) -> str:
    scope = " ".join(str(value or "").strip().lower().split())
    normalized = _SCOPE_ALIASES.get(scope)
    if normalized is None:
        raise ValueError(
            "scope must be one of: " + ", ".join(DIAGNOSIS_SCOPES)
        )
    return normalized


def collect_system_diagnosis_evidence(
    systemsense: SystemSense,
    *,
    scope: str = "signals",
    history_hours: float = 1.0,
    history_limit: int = 60,
) -> dict[str, Any]:
    """Collect fresh read-only evidence without using the presentation snapshot."""
    normalized_scope = normalize_diagnosis_scope(scope)
    dynamic = systemsense.collect_dynamic()

    raw_current = {
        "captured_at": dynamic.get("captured_at"),
        "base": dynamic.get("base") or {},
        "windows_performance": dynamic.get("performance") or {},
        "hardware_provider": dynamic.get("raw_sensors") or {},
    }

    history: dict[str, Any] = {}
    history_errors: dict[str, str] = {}
    for metric in _SCOPE_METRICS[normalized_scope]:
        try:
            history[metric] = systemsense.history(
                metric=metric,
                hours=history_hours,
                limit=history_limit,
            )
        except Exception as exc:
            history_errors[metric] = type(exc).__name__

    inventory: dict[str, Any] | None = None
    inventory_error: str | None = None
    if normalized_scope in {"signals", "storage"}:
        try:
            # Refresh the raw inventory because a diagnosis is an explicit owner
            # request rather than passive UI polling.
            systemsense.collect_inventory()
            inventory = systemsense.raw(category="inventory", limit=100)
        except Exception as exc:
            inventory_error = type(exc).__name__

    correlations: dict[str, Any] | None = None
    if normalized_scope == "signals":
        try:
            correlations = systemsense.correlations(limit=10)
        except Exception:
            correlations = None

    return {
        "kind": "systemsense_raw_diagnosis_evidence",
        "scope": normalized_scope,
        "current_truth": raw_current,
        "recent_metric_history": history,
        "history_errors": history_errors,
        "raw_inventory": inventory,
        "inventory_error": inventory_error,
        "workload_correlations": correlations,
        "evidence_notes": [
            "current_truth is the authoritative current hardware/OS evidence",
            "recent_metric_history is bounded historical context and must not override contradictory current raw sensors",
            "no user-facing SystemSense presentation snapshot is included",
            "workload correlations are observational and do not establish causality",
        ],
    }


def build_system_diagnosis_prompt(
    systemsense: SystemSense,
    *,
    scope: str = "signals",
) -> tuple[str, dict[str, Any]]:
    """Build one bounded evidence-only reasoning turn for Astra."""
    evidence = collect_system_diagnosis_evidence(systemsense, scope=scope)
    normalized_scope = str(evidence["scope"])
    rendered = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)

    prompt = (
        "SYSTEMSENSE DIAGNOSTIC REQUEST\n"
        f"The owner explicitly asked you to diagnose the current {normalized_scope} state. "
        "Reason only from the fresh raw evidence package below, without using any tools, durable memory, "
        "or the human-facing SystemSense presentation snapshot. The evidence package was collected "
        "immediately before this inference.\n\n"
        "Treat current raw sensor/OS values as authoritative for current health. Use recent metric history "
        "only to distinguish a genuinely unhealthy condition from a normal workload or a harmless deviation "
        "from an unusually low rolling baseline. Never call a baseline deviation a fault by itself. "
        "Do not recommend changing a setting merely because a value differs from baseline. "
        "Do not perform or propose an automatic action; this is diagnosis only.\n\n"
        "Lead with a short plain-English conclusion. Then explain the strongest evidence, including any "
        "specific component/process/sensor responsible when the raw data establishes one. End with either "
        "'No action needed' or the smallest sensible next action. State unavailable evidence plainly.\n\n"
        "RAW_EVIDENCE_JSON:\n"
        + rendered
    )
    return prompt, evidence
