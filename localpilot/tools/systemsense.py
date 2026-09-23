from __future__ import annotations

import json

from localpilot.systemsense import SystemSense
from localpilot.systemsense_backend import BackendTelemetryCollector
from localpilot.systemsense_views import build_agent_truth
from localpilot.tools.windows import inspect_executable_metadata, inspect_process_identity


class SystemSenseReader:
    """Raw-truth-first read-only access to passive environmental telemetry."""

    def __init__(
        self,
        systemsense: SystemSense,
        *,
        backend: BackendTelemetryCollector | None = None,
    ) -> None:
        self.systemsense = systemsense
        self.backend = backend or BackendTelemetryCollector()

    @staticmethod
    def _render(payload: object) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def get_system_sense_summary(self) -> str:
        """Return canonical raw collector truth for LocalPilot reasoning."""
        if isinstance(self.systemsense, SystemSense):
            return self._render(build_agent_truth(self.systemsense))
        # Preserve duck-typed test/integration adapters that intentionally
        # provide only the legacy summary surface; production SystemSense
        # instances always take the raw-truth path above.
        return self._render(self.systemsense.summary())

    def inspect_hardware_inventory(
        self, section: str = "overview", limit: int = 50
    ) -> str:
        """Inspect one bounded hardware/firmware/device inventory section."""
        return self._render(self.systemsense.inventory(section=section, limit=limit))

    def inspect_driver_inventory(
        self, classification: str = "all", limit: int = 100
    ) -> str:
        """Inspect bound, inactive, problematic or review-candidate driver records."""
        return self._render(
            self.systemsense.drivers(classification=classification, limit=limit)
        )

    def get_system_sense_history(
        self, metric: str, hours: float = 1.0, limit: int = 120
    ) -> str:
        """Read bounded history for one allow-listed environmental metric."""
        return self._render(
            self.systemsense.history(metric=metric, hours=hours, limit=limit)
        )

    def get_memory_watch_report(self, watch_id: int = 0) -> str:
        """Read the latest or a specific owner-requested RAM watch report."""
        selected = int(watch_id)
        return self._render(
            self.systemsense.memory_watch_report(
                watch_id=selected if selected > 0 else None
            )
        )

    def inspect_memory_watch_process(self, pid: int, watch_id: int = 0) -> str:
        """Inspect one process instance observed by a RAM watch.

        Historical executable/command-line/parent evidence comes from the watch
        sample itself. File metadata/signature is inspected from that recorded
        executable path. Current PID evidence is attached only when Windows still
        reports the same process start time, avoiding PID-reuse confusion.
        """
        selected = int(watch_id)
        identity = self.systemsense.memory_watch_process_identity(
            pid=int(pid),
            watch_id=selected if selected > 0 else None,
        )
        if not identity.get("available"):
            return self._render(identity)

        executable = str(identity.get("executable") or "")
        if executable:
            identity["executable_metadata"] = inspect_executable_metadata(executable)

        try:
            current = json.loads(inspect_process_identity(int(pid)))
        except (json.JSONDecodeError, TypeError, ValueError):
            current = {"available": False, "reason": "current_process_query_failed"}

        historical_start = identity.get("started_at_epoch")
        current_start = current.get("started_at_epoch") if isinstance(current, dict) else None
        same_instance = bool(
            isinstance(current, dict)
            and current.get("available")
            and historical_start is not None
            and current_start is not None
            and abs(float(historical_start) - float(current_start)) <= 2.0
        )
        if same_instance:
            identity["current_process"] = current
        else:
            identity["current_process"] = {
                "available": False,
                "reason": (
                    "process_no_longer_running"
                    if not isinstance(current, dict) or not current.get("available")
                    else "pid_now_refers_to_different_process_instance"
                ),
            }
        identity["identity_note"] = (
            "Watch-captured executable/command-line/parent fields identify the historical "
            "process instance. Current PID evidence is included only when process start "
            "time confirms that the PID still refers to the same instance."
        )
        return self._render(identity)

    def get_workload_correlations(self, limit: int = 10) -> str:
        """Read observational correlations between inference speed and resources."""
        return self._render(self.systemsense.correlations(limit=limit))

    def inspect_raw_system_sense(
        self,
        category: str = "dynamic",
        limit: int = 100,
        backend_section: str = "overview",
    ) -> str:
        """Drill into raw passive data or bounded high-detail backend telemetry."""
        if str(category).strip().casefold() == "backend":
            return self._render(
                self.backend.collect(section=backend_section, limit=limit)
            )
        return self._render(self.systemsense.raw(category=category, limit=limit))
