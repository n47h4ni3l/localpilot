from __future__ import annotations

import json

from localpilot.systemsense import SystemSense
from localpilot.systemsense_backend import BackendTelemetryCollector
from localpilot.systemsense_views import build_agent_truth
from localpilot.tools.windows import inspect_process_identity, inspect_process_launch_context


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

    def get_systemsense_watch_report(
        self,
        watch_id: int = 0,
        profile: str = "",
    ) -> str:
        """Read the latest or a specific owner-requested SystemSense watch report."""
        selected = int(watch_id)
        return self._render(
            self.systemsense.watch_report(
                watch_id=selected if selected > 0 else None,
                profile=str(profile).strip().casefold() or None,
            )
        )

    def inspect_systemsense_watch_process(self, pid: int, watch_id: int = 0) -> str:
        """Inspect a process instance observed by a SystemSense watch.

        The historical fingerprint, command line, ancestry, lifecycle, network
        endpoints and resource peaks come from watch-time evidence. Current PID
        state is attached only when start time proves Windows still refers to the
        same process instance. Launch configuration/crash evidence is queried
        separately and cannot overwrite the historical fingerprint.
        """
        selected = int(watch_id)
        identity = self.systemsense.watch_process_identity(
            pid=int(pid),
            watch_id=selected if selected > 0 else None,
        )
        if not identity.get("available"):
            return self._render(identity)

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

        executable = str(identity.get("executable") or "")
        launch_raw = inspect_process_launch_context(
            executable,
            process_name=str(identity.get("name") or ""),
            pid=int(pid),
            observed_at=str(identity.get("captured_at") or ""),
        )
        try:
            identity["launch_context"] = json.loads(launch_raw)
        except (json.JSONDecodeError, TypeError):
            identity["launch_context"] = {
                "available": False,
                "reason": "launch_context_query_failed",
            }

        identity["identity_note"] = (
            "Executable hash/signature/version fields are the artifact fingerprint "
            "captured during the watch. Current PID evidence is included only when "
            "process start time confirms the same instance. Launch context is "
            "best-effort configuration/crash evidence; it is not proof of causation."
        )
        return self._render(identity)

    # Compatibility read-only names from the first RAM-specific implementation.
    def get_memory_watch_report(self, watch_id: int = 0) -> str:
        return self.get_systemsense_watch_report(watch_id=watch_id, profile="memory")

    def inspect_memory_watch_process(self, pid: int, watch_id: int = 0) -> str:
        return self.inspect_systemsense_watch_process(pid=pid, watch_id=watch_id)

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
