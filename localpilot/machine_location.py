from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from localpilot.process import hidden_process_creation_flags


_LOCATION_SOURCE = "Windows Location Service"
_DEFAULT_CACHE_AGE = timedelta(hours=2)


@dataclass(frozen=True, slots=True)
class LocationSnapshot:
    latitude: float
    longitude: float
    accuracy_m: float | None
    updated_at: str
    source: str = _LOCATION_SOURCE


class MachineLocation:
    """Owner-controlled machine location with an explicit local privacy boundary.

    Exact coordinates are kept only in this private machine-local state file.
    Model-facing context is rounded to roughly kilometre scale and is only
    supplied for prompts that actually depend on location.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "machine-location.json"
        self._lock = threading.Lock()

    @staticmethod
    def defaults() -> dict[str, Any]:
        return {
            "enabled": False,
            "latitude": None,
            "longitude": None,
            "accuracy_m": None,
            "updated_at": None,
            "source": None,
            "last_error": None,
        }

    def read(self) -> dict[str, Any]:
        values = self.defaults()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return values
        if not isinstance(raw, dict):
            return values
        for key in values:
            if key in raw:
                values[key] = raw[key]

        values["enabled"] = bool(values["enabled"])
        for key in ("latitude", "longitude", "accuracy_m"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                values[key] = None
            elif not math.isfinite(float(value)):
                values[key] = None
            else:
                values[key] = float(value)
        if not isinstance(values["updated_at"], str):
            values["updated_at"] = None
        if not isinstance(values["source"], str):
            values["source"] = None
        if not isinstance(values["last_error"], str):
            values["last_error"] = None

        latitude = values["latitude"]
        longitude = values["longitude"]
        if latitude is not None and not -90.0 <= latitude <= 90.0:
            values["latitude"] = None
        if longitude is not None and not -180.0 <= longitude <= 180.0:
            values["longitude"] = None
        return values

    def _write(self, values: dict[str, Any]) -> dict[str, Any]:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(values, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return values

    def set_enabled(self, enabled: bool, *, refresh: bool = True) -> dict[str, Any]:
        enabled = bool(enabled)
        with self._lock:
            values = self.read()
            values["enabled"] = enabled
            if not enabled:
                # Turning the feature off also removes retained precise location.
                values.update(
                    latitude=None,
                    longitude=None,
                    accuracy_m=None,
                    updated_at=None,
                    source=None,
                    last_error=None,
                )
            self._write(values)
        if enabled and refresh:
            self.refresh()
        return self.public_status()

    def refresh(self) -> dict[str, Any]:
        values = self.read()
        if not values["enabled"]:
            return self.public_status()

        try:
            snapshot = self._capture_windows_location()
        except Exception as exc:
            with self._lock:
                current = self.read()
                current["last_error"] = self._safe_error(exc)
                self._write(current)
            return self.public_status()

        with self._lock:
            current = self.read()
            if not current["enabled"]:
                return self.public_status()
            current.update(
                latitude=snapshot.latitude,
                longitude=snapshot.longitude,
                accuracy_m=snapshot.accuracy_m,
                updated_at=snapshot.updated_at,
                source=snapshot.source,
                last_error=None,
            )
            self._write(current)
        return self.public_status()

    def exact_snapshot(self, *, refresh_if_stale: bool = True) -> LocationSnapshot | None:
        values = self.read()
        if not values["enabled"]:
            return None
        if refresh_if_stale and self._is_stale(values.get("updated_at")):
            self.refresh()
            values = self.read()
        latitude = values.get("latitude")
        longitude = values.get("longitude")
        updated_at = values.get("updated_at")
        if latitude is None or longitude is None or not isinstance(updated_at, str):
            return None
        return LocationSnapshot(
            latitude=float(latitude),
            longitude=float(longitude),
            accuracy_m=(
                float(values["accuracy_m"])
                if values.get("accuracy_m") is not None
                else None
            ),
            updated_at=updated_at,
            source=str(values.get("source") or _LOCATION_SOURCE),
        )

    def coarse_model_context(self, *, refresh_if_stale: bool = True) -> dict[str, Any] | None:
        snapshot = self.exact_snapshot(refresh_if_stale=refresh_if_stale)
        if snapshot is None:
            return None
        return {
            "kind": "machine_location_context",
            "enabled": True,
            # Two decimal places is roughly kilometre-scale in latitude and is
            # sufficient for weather/local-search disambiguation without
            # handing precise coordinates to the language model.
            "approximate_latitude": round(snapshot.latitude, 2),
            "approximate_longitude": round(snapshot.longitude, 2),
            "accuracy_m": (
                round(snapshot.accuracy_m, 1)
                if snapshot.accuracy_m is not None
                else None
            ),
            "source": snapshot.source,
            "updated_at": snapshot.updated_at,
            "privacy": (
                "Approximate model context only. Exact coordinates remain in "
                "machine-local location state and must not be persisted to "
                "LearningMemory or ordinary chat history."
            ),
        }

    def public_status(self) -> dict[str, Any]:
        values = self.read()
        available = bool(
            values["enabled"]
            and values.get("latitude") is not None
            and values.get("longitude") is not None
            and values.get("updated_at")
        )
        accuracy = values.get("accuracy_m")
        return {
            "ok": True,
            "enabled": bool(values["enabled"]),
            "available": available,
            "accuracyM": round(float(accuracy), 1) if accuracy is not None else None,
            "updatedAt": values.get("updated_at"),
            "source": values.get("source"),
            "stale": self._is_stale(values.get("updated_at")) if available else False,
            "error": values.get("last_error"),
        }

    @staticmethod
    def prompt_needs_location(prompt: str) -> bool:
        text = " ".join(str(prompt or "").lower().split())
        if not text:
            return False
        markers = (
            "weather",
            "forecast",
            "near me",
            "nearby",
            "around here",
            "local area",
            "where am i",
            "where are we",
            "my location",
            "this pc's location",
            "this pc location",
            "sunrise",
            "sunset",
            "air quality",
            "pollen",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_stale(updated_at: Any) -> bool:
        if not isinstance(updated_at, str) or not updated_at:
            return True
        try:
            captured = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=UTC)
        return datetime.now(UTC) - captured.astimezone(UTC) > _DEFAULT_CACHE_AGE

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = " ".join(str(exc).strip().split())
        if not message:
            return type(exc).__name__
        return message[:240]

    @staticmethod
    def _capture_windows_location() -> LocationSnapshot:
        if os.name != "nt":
            raise RuntimeError("Machine location is currently supported on Windows only.")

        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("Windows PowerShell is unavailable.")

        script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Device
$watcher = New-Object System.Device.Location.GeoCoordinateWatcher(
    [System.Device.Location.GeoPositionAccuracy]::Default
)
try {
    if (-not $watcher.TryStart($false, [TimeSpan]::FromSeconds(6))) {
        throw 'Windows Location Service did not return a position. Check Windows location permissions.'
    }
    $coord = $watcher.Position.Location
    if ($null -eq $coord -or $coord.IsUnknown) {
        throw 'Windows Location Service returned an unknown position.'
    }
    $accuracy = $null
    if (-not [Double]::IsNaN($coord.HorizontalAccuracy) -and
        -not [Double]::IsInfinity($coord.HorizontalAccuracy)) {
        $accuracy = [Double]$coord.HorizontalAccuracy
    }
    [ordered]@{
        ok = $true
        latitude = [Double]$coord.Latitude
        longitude = [Double]$coord.Longitude
        accuracy_m = $accuracy
    } | ConvertTo-Json -Compress
}
finally {
    $watcher.Stop()
    $watcher.Dispose()
}
"""
        result = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
            creationflags=hidden_process_creation_flags(),
        )
        if result.returncode != 0:
            detail = " ".join((result.stderr or result.stdout or "").strip().split())
            raise RuntimeError(detail or "Windows Location Service query failed.")
        try:
            payload = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError("Windows Location Service returned invalid data.") from exc

        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        accuracy = payload.get("accuracy_m")
        accuracy_m = float(accuracy) if accuracy is not None else None
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise RuntimeError("Windows Location Service returned invalid coordinates.")

        return LocationSnapshot(
            latitude=latitude,
            longitude=longitude,
            accuracy_m=accuracy_m,
            updated_at=datetime.now(UTC).isoformat(),
        )
