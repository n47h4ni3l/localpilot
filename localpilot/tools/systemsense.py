from __future__ import annotations

import json

from localpilot.systemsense import SystemSense


class SystemSenseReader:
    """Bounded summary-first access to passive environmental telemetry."""

    def __init__(self, systemsense: SystemSense) -> None:
        self.systemsense = systemsense

    @staticmethod
    def _render(payload: object) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def get_system_sense_summary(self) -> str:
        """Return compact derived health, pressure, anomalies and inference state."""
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

    def get_workload_correlations(self, limit: int = 10) -> str:
        """Read observational correlations between inference speed and resources."""
        return self._render(self.systemsense.correlations(limit=limit))

    def inspect_raw_system_sense(
        self, category: str = "dynamic", limit: int = 100
    ) -> str:
        """Drill into bounded raw dynamic, sensor or inventory telemetry."""
        return self._render(self.systemsense.raw(category=category, limit=limit))
