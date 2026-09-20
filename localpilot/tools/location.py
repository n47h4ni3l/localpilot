from __future__ import annotations

from typing import Any

from localpilot.machine_location import MachineLocation


class MachineLocationReader:
    """Read-only model surface for owner-enabled coarse machine location."""

    def __init__(self, location: MachineLocation) -> None:
        self.location = location

    def get_machine_location(self) -> dict[str, Any]:
        status = self.location.public_status()
        if not status["enabled"]:
            return {
                "enabled": False,
                "available": False,
                "message": "Machine location access is disabled by the owner.",
            }

        context = self.location.coarse_model_context(refresh_if_stale=True)
        if context is None:
            refreshed = self.location.public_status()
            return {
                "enabled": True,
                "available": False,
                "source": refreshed.get("source"),
                "error": refreshed.get("error") or "Windows location unavailable",
                "message": "Do not guess the machine location.",
            }

        return {
            "enabled": True,
            "available": True,
            "approximate_latitude": context["approximate_latitude"],
            "approximate_longitude": context["approximate_longitude"],
            "accuracy_m": context["accuracy_m"],
            "updated_at": context["updated_at"],
            "stale": context["stale"],
            "source": context["source"],
            "privacy": (
                "Approximate coordinates only. Exact machine coordinates remain local "
                "and are not exposed through this model tool."
            ),
        }
