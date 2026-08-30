from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import statistics
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from localpilot.config import SystemSenseConfig
from localpilot.runtime_evidence import RuntimeEvidence
from localpilot.systemsense_collectors import (
    LibreHardwareMonitorCollector,
    PsutilTelemetryCollector,
    WindowsInventoryCollector,
    WindowsPerformanceCollector,
    utc_timestamp,
)

_IDENTIFIER_PATTERNS = {
    "pci_vendor_id": re.compile(r"VEN_([0-9A-F]{4})", re.IGNORECASE),
    "pci_device_id": re.compile(r"DEV_([0-9A-F]{4})", re.IGNORECASE),
    "pci_subsystem_id": re.compile(r"SUBSYS_([0-9A-F]{8})", re.IGNORECASE),
    "usb_vendor_id": re.compile(r"VID_([0-9A-F]{4})", re.IGNORECASE),
    "usb_product_id": re.compile(r"PID_([0-9A-F]{4})", re.IGNORECASE),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _path_get(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _pressure(percent: float | None) -> str:
    if percent is None:
        return "unknown"
    if percent >= 95:
        return "critical"
    if percent >= 85:
        return "high"
    if percent >= 65:
        return "moderate"
    return "low"


def _version_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(value or "")))


class SystemSenseStore:
    """Private local telemetry store with fixed, read-only query surfaces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_kind_time
                    ON snapshots(kind, captured_at DESC);
                CREATE TABLE IF NOT EXISTS metrics (
                    captured_at TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_key_time
                    ON metrics(key, captured_at DESC);
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    phase TEXT,
                    tokens_per_second REAL,
                    time_to_first_token_ms REAL,
                    total_latency_ms REAL,
                    prompt_tokens INTEGER,
                    eval_tokens INTEGER,
                    context_percent REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_inference_time
                    ON inference_metrics(captured_at DESC);
                """
            )

    def save_snapshot(
        self, kind: str, payload: dict[str, Any], captured_at: str | None = None
    ) -> None:
        when = captured_at or str(payload.get("captured_at") or utc_timestamp())
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO snapshots(captured_at, kind, payload_json) VALUES (?, ?, ?)",
                (when, kind, _json(payload)),
            )

    def replace_latest_snapshot(self, kind: str, payload: dict[str, Any]) -> None:
        """Keep one raw current-state row while normalized metrics retain history."""
        when = str(payload.get("captured_at") or utc_timestamp())
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM snapshots WHERE kind=? ORDER BY captured_at DESC LIMIT 1",
                (kind,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO snapshots(captured_at, kind, payload_json) VALUES (?, ?, ?)",
                    (when, kind, _json(payload)),
                )
            else:
                connection.execute(
                    "UPDATE snapshots SET captured_at=?, payload_json=? WHERE id=?",
                    (when, _json(payload), int(row["id"])),
                )

    def save_snapshot_if_changed(self, kind: str, payload: dict[str, Any]) -> bool:
        """Retain slow inventory revisions without duplicating unchanged payloads."""
        previous = self.latest_snapshot(kind)
        comparable = dict(payload)
        comparable.pop("captured_at", None)
        if previous is not None:
            prior = dict(previous)
            prior.pop("captured_at", None)
            if prior == comparable:
                return False
        self.save_snapshot(kind, payload)
        return True

    def save_metrics(
        self,
        captured_at: str,
        rows: Iterable[tuple[str, float, str, str]],
    ) -> None:
        values = [(captured_at, key, value, unit, source) for key, value, unit, source in rows]
        if not values:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO metrics(captured_at, key, value, unit, source) VALUES (?, ?, ?, ?, ?)",
                values,
            )

    def latest_snapshot(self, kind: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM snapshots WHERE kind=? ORDER BY captured_at DESC LIMIT 1",
                (kind,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def snapshots(self, kind: str, *, since: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT captured_at, payload_json FROM snapshots "
                "WHERE kind=? AND captured_at>=? ORDER BY captured_at DESC LIMIT ?",
                (kind, since, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def metric_history(self, key: str, *, since: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT captured_at, value, unit, source FROM metrics "
                "WHERE key=? AND captured_at>=? ORDER BY captured_at DESC LIMIT ?",
                (key, since, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_metric_values(self, key: str, *, since: str, limit: int = 20_000) -> list[float]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT value FROM metrics WHERE key=? AND captured_at>=? "
                "ORDER BY captured_at DESC LIMIT ?",
                (key, since, limit),
            ).fetchall()
        return [float(row["value"]) for row in rows]

    def save_inference(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO inference_metrics(
                    captured_at, model, phase, tokens_per_second,
                    time_to_first_token_ms, total_latency_ms, prompt_tokens,
                    eval_tokens, context_percent, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["captured_at"],
                    payload["model"],
                    payload.get("phase"),
                    payload.get("tokens_per_second"),
                    payload.get("time_to_first_token_ms"),
                    payload.get("total_latency_ms"),
                    payload.get("prompt_tokens"),
                    payload.get("eval_tokens"),
                    payload.get("context_percent"),
                    _json(payload),
                ),
            )

    def inference_rows(self, *, since: str, limit: int = 10_000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM inference_metrics WHERE captured_at>=? "
                "ORDER BY captured_at DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def prune(self, *, before: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM snapshots WHERE captured_at<?", (before,))
            connection.execute("DELETE FROM metrics WHERE captured_at<?", (before,))
            connection.execute("DELETE FROM inference_metrics WHERE captured_at<?", (before,))


class SystemSense:
    """Always-available environmental state engine beneath the LLM."""

    _BASELINE_KEYS = (
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
    )

    def __init__(
        self,
        config: SystemSenseConfig,
        data_dir: str | Path,
        *,
        psutil_collector: PsutilTelemetryCollector | None = None,
        performance_collector: WindowsPerformanceCollector | None = None,
        sensor_collector: LibreHardwareMonitorCollector | None = None,
        inventory_collector: WindowsInventoryCollector | None = None,
        project_root: str | Path | None = None,
        main_branch: str = "main",
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir).resolve()
        self.store = SystemSenseStore(self.data_dir / config.database)
        self.psutil = psutil_collector or PsutilTelemetryCollector(
            max_processes=config.max_processes
        )
        self.performance = performance_collector or WindowsPerformanceCollector()
        self.sensors = sensor_collector or LibreHardwareMonitorCollector()
        self.inventory_collector = inventory_collector or WindowsInventoryCollector()
        self._collect_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prune = 0.0
        self._runtime_evidence = (
            RuntimeEvidence(
                project_root,
                self.data_dir / "audit.jsonl",
                main_branch=main_branch,
            )
            if project_root is not None
            else None
        )

    def configure_runtime_evidence(
        self, project_root: str | Path, *, main_branch: str = "main"
    ) -> None:
        self._runtime_evidence = RuntimeEvidence(
            project_root,
            self.data_dir / "audit.jsonl",
            main_branch=main_branch,
        )

    def runtime_evidence(self) -> dict[str, Any] | None:
        return self._runtime_evidence.snapshot() if self._runtime_evidence is not None else None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> None:
        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="localpilot-systemsense",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        next_inventory = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.collect_dynamic()
            except Exception as exc:
                self.store.save_snapshot(
                    "collector_error",
                    {
                        "captured_at": utc_timestamp(),
                        "collector": "dynamic",
                        "error_type": type(exc).__name__,
                    },
                )
            if time.monotonic() >= next_inventory:
                try:
                    self.collect_inventory()
                except Exception as exc:
                    self.store.save_snapshot(
                        "collector_error",
                        {
                            "captured_at": utc_timestamp(),
                            "collector": "inventory",
                            "error_type": type(exc).__name__,
                        },
                    )
                next_inventory = time.monotonic() + self.config.inventory_interval_seconds
            if time.monotonic() - self._last_prune >= 3600:
                before = (_utc_now() - timedelta(days=self.config.retention_days)).isoformat()
                self.store.prune(before=before)
                self._last_prune = time.monotonic()
            remaining = max(0.1, self.config.sample_interval_seconds - (time.monotonic() - started))
            self._stop.wait(remaining)

    @staticmethod
    def _sensor_summary(payload: dict[str, Any]) -> dict[str, Any]:
        readings = payload.get("sensors") or []
        summary: dict[str, Any] = {
            "available": bool(payload.get("available")),
            "source": payload.get("source"),
            "temperatures": [],
            "fans": [],
            "loads": [],
            "power": [],
            "clocks": [],
            "data": [],
        }
        mapping = {
            "temperature": "temperatures",
            "fan": "fans",
            "load": "loads",
            "power": "power",
            "clock": "clocks",
            "data": "data",
            "smalldata": "data",
        }
        for row in readings:
            group = mapping.get(str(row.get("SensorType") or "").casefold())
            value = _finite(row.get("Value"))
            if group is None or value is None:
                continue
            summary[group].append(
                {
                    "id": row.get("Identifier"),
                    "name": row.get("Name"),
                    "value": value,
                    "min": _finite(row.get("Min")),
                    "max": _finite(row.get("Max")),
                    "parent": row.get("Parent"),
                }
            )
        temperatures = [item["value"] for item in summary["temperatures"]]
        summary["max_temperature_c"] = max(temperatures) if temperatures else None
        gpu_loads = [
            item["value"]
            for item in summary["loads"]
            if "gpu" in f"{item.get('id')} {item.get('parent')} {item.get('name')}".casefold()
        ]
        summary["gpu_load_percent"] = max(gpu_loads) if gpu_loads else None
        memory_data = [
            item["value"]
            for item in summary["data"]
            if "gpu" in f"{item.get('id')} {item.get('parent')}".casefold()
            and any(word in str(item.get("name") or "").casefold() for word in ("memory", "vram"))
        ]
        summary["vram_used_mb"] = max(memory_data) if memory_data else None
        return summary

    @staticmethod
    def _metric_rows(payload: dict[str, Any]) -> list[tuple[str, float, str, str]]:
        paths = {
            "cpu.percent": ("base.cpu.percent", "%", "psutil"),
            "cpu.frequency_mhz": ("base.cpu.frequency_mhz", "MHz", "psutil"),
            "memory.percent": ("base.memory.percent", "%", "psutil"),
            "memory.available_gb": ("base.memory.available_gb", "GiB", "psutil"),
            "storage.read_mb_s": ("base.storage.io.read_mb_s", "MiB/s", "psutil"),
            "storage.write_mb_s": ("base.storage.io.write_mb_s", "MiB/s", "psutil"),
            "network.send_mbps": ("base.network.send_mbps", "Mbit/s", "psutil"),
            "network.receive_mbps": ("base.network.receive_mbps", "Mbit/s", "psutil"),
            "gpu.utilization_percent": ("derived.gpu_utilization_percent", "%", "windows/lhm"),
            "thermal.max_c": ("derived.max_temperature_c", "C", "windows/lhm"),
            "vram.used_mb": ("derived.vram_used_mb", "MiB", "lhm"),
            "processor.performance_limit_percent": (
                "performance.processor.PercentPerformanceLimit",
                "%",
                "windows-wmi",
            ),
            "processor.performance_percent": (
                "performance.processor.PercentProcessorPerformance",
                "%",
                "windows-wmi",
            ),
        }
        rows = []
        for key, (path, unit, source) in paths.items():
            value = _finite(_path_get(payload, path))
            if value is not None:
                rows.append((key, value, unit, source))
        return rows

    def collect_dynamic(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "captured_at": utc_timestamp()}
        with self._collect_lock:
            base = self.psutil.collect()
            performance = self.performance.collect()
            raw_sensors = self.sensors.collect()
            sensors = self._sensor_summary(raw_sensors)
            gpu_perf = _finite(
                _path_get(performance, "gpu.engine_utilization_percent")
            )
            gpu_load = _finite(sensors.get("gpu_load_percent"))
            temperatures = [
                value
                for value in (
                    _finite(sensors.get("max_temperature_c")),
                    *(
                        _finite(row.get("current_c"))
                        for row in performance.get("thermal_zones") or []
                    ),
                )
                if value is not None
            ]
            payload = {
                "captured_at": str(base.get("captured_at") or utc_timestamp()),
                "base": base,
                "performance": performance,
                "sensors": sensors,
                "raw_sensors": raw_sensors,
                "derived": {
                    "gpu_utilization_percent": max(
                        [value for value in (gpu_perf, gpu_load) if value is not None],
                        default=None,
                    ),
                    "max_temperature_c": max(temperatures, default=None),
                    "vram_used_mb": _finite(sensors.get("vram_used_mb")),
                },
            }
            self.store.replace_latest_snapshot("dynamic", payload)
            self.store.save_metrics(
                payload["captured_at"], self._metric_rows(payload)
            )
            return payload

    @staticmethod
    def _hardware_ids(device_id: Any, hardware_ids: Any) -> dict[str, str]:
        values = [str(device_id or "")]
        if isinstance(hardware_ids, list):
            values.extend(str(item) for item in hardware_ids)
        elif hardware_ids:
            values.append(str(hardware_ids))
        joined = " ".join(values)
        output = {}
        for name, pattern in _IDENTIFIER_PATTERNS.items():
            match = pattern.search(joined)
            if match:
                output[name] = match.group(1).upper()
        return output

    @classmethod
    def _classify_inventory(cls, raw: dict[str, Any]) -> dict[str, Any]:
        devices = []
        by_id: dict[str, dict[str, Any]] = {}
        for source in raw.get("devices") or []:
            row = dict(source)
            error_code = int(row.get("ConfigManagerErrorCode") or 0)
            present_raw = row.get("Present")
            present = bool(present_raw) if present_raw is not None else error_code != 45
            if error_code == 22:
                classification = "disabled"
            elif error_code not in (0, 45):
                classification = "problem"
            elif not present:
                classification = "hidden_or_disconnected"
            else:
                classification = "present_ok"
            row.update(
                {
                    "present": present,
                    "classification": classification,
                    "identifiers": cls._hardware_ids(
                        row.get("DeviceID"), row.get("HardwareID")
                    ),
                }
            )
            device_id = str(row.get("DeviceID") or "").casefold()
            if device_id:
                by_id[device_id] = row
            devices.append(row)

        services = {
            str(row.get("Name") or "").casefold(): row
            for row in raw.get("system_drivers") or []
            if row.get("Name")
        }
        drivers = []
        bound_infs: dict[str, list[dict[str, Any]]] = {}
        for source in raw.get("signed_drivers") or []:
            row = dict(source)
            device = by_id.get(str(row.get("DeviceID") or "").casefold())
            service = services.get(str((device or {}).get("Service") or "").casefold())
            present = bool((device or {}).get("present"))
            problem = (device or {}).get("classification") in {"problem", "disabled"}
            running = bool(service and (service.get("Started") or str(service.get("State")).casefold() == "running"))
            if problem:
                classification = "problematic"
            elif present and (running or service is None):
                classification = "active_bound"
            elif device is not None:
                classification = "installed_bound_inactive"
            else:
                classification = "association_unknown"
            row.update(
                {
                    "classification": classification,
                    "device_present": present,
                    "device_problem": bool(problem),
                    "service": service,
                    "device": device,
                }
            )
            inf = str(row.get("InfName") or "").casefold()
            if inf:
                bound_infs.setdefault(inf, []).append(row)
            drivers.append(row)

        packages = []
        package_rows = (raw.get("driver_store") or {}).get("packages") or []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for source in package_rows:
            row = dict(source)
            published = str(row.get("published_name") or "").casefold()
            associations = bound_infs.get(published, [])
            active = any(item.get("classification") == "active_bound" for item in associations)
            if active:
                classification = "active_bound"
                confidence = "high"
            elif associations:
                classification = "installed_bound_inactive"
                confidence = "high"
            else:
                classification = "orphan_candidate_requires_review"
                confidence = "low"
            row.update(
                {
                    "classification": classification,
                    "classification_confidence": confidence,
                    "device_associations": len(associations),
                    "safe_to_delete": False,
                }
            )
            key = (
                str(row.get("provider") or "").casefold(),
                str(row.get("original_name") or "").casefold(),
            )
            if all(key):
                groups.setdefault(key, []).append(row)
            packages.append(row)
        for rows in groups.values():
            latest = max((_version_tuple(row.get("driver_version")) for row in rows), default=())
            for row in rows:
                if (
                    row["classification"] == "orphan_candidate_requires_review"
                    and _version_tuple(row.get("driver_version")) < latest
                ):
                    row["classification"] = "duplicate_older_candidate_requires_review"

        raw["devices"] = devices
        raw["drivers"] = drivers
        raw["driver_packages"] = packages
        raw["summary"] = {
            "devices_total": len(devices),
            "devices_present": sum(row["classification"] == "present_ok" for row in devices),
            "devices_problematic": sum(row["classification"] == "problem" for row in devices),
            "devices_disabled": sum(row["classification"] == "disabled" for row in devices),
            "devices_hidden_or_disconnected": sum(row["classification"] == "hidden_or_disconnected" for row in devices),
            "drivers_active_bound": sum(row["classification"] == "active_bound" for row in drivers),
            "drivers_bound_inactive": sum(row["classification"] == "installed_bound_inactive" for row in drivers),
            "driver_packages_orphan_candidates": sum("orphan_candidate" in row["classification"] for row in packages),
            "classification_warning": (
                "Inactive or unassociated packages are review candidates, never proof that a driver is useless or safe to remove. "
                "Boot, recovery, disconnected-device and rollback packages can be dormant legitimately."
            ),
        }
        return raw

    def collect_inventory(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "captured_at": utc_timestamp()}
        with self._collect_lock:
            payload = self._classify_inventory(self.inventory_collector.collect())
            self.store.save_snapshot_if_changed("inventory", payload)
            return payload

    def _ensure_dynamic(self) -> dict[str, Any] | None:
        latest = self.store.latest_snapshot("dynamic")
        if latest is not None:
            return latest
        try:
            return self.collect_dynamic()
        except Exception:
            return None

    def baselines(self) -> dict[str, dict[str, Any]]:
        since = (_utc_now() - timedelta(hours=self.config.baseline_window_hours)).isoformat()
        output = {}
        for key in self._BASELINE_KEYS:
            values = self.store.recent_metric_values(key, since=since)
            if not values:
                continue
            median = statistics.median(values)
            deviations = [abs(value - median) for value in values]
            output[key] = {
                "samples": len(values),
                "median": round(median, 4),
                "mean": round(statistics.fmean(values), 4),
                "mad": round(statistics.median(deviations), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
        return output

    def _anomalies(
        self, current: dict[str, float], baselines: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        anomalies = []
        for key, value in current.items():
            baseline = baselines.get(key)
            if not baseline or baseline["samples"] < 6:
                continue
            scale = float(baseline["mad"]) * 1.4826
            if scale <= 1e-9:
                spread = float(baseline["max"]) - float(baseline["min"])
                scale = max(spread / 4.0, abs(float(baseline["median"])) * 0.01, 0.01)
            score = (value - float(baseline["median"])) / scale
            if abs(score) >= 3.5:
                anomalies.append(
                    {
                        "metric": key,
                        "current": round(value, 3),
                        "baseline_median": baseline["median"],
                        "robust_z": round(score, 2),
                        "direction": "high" if score > 0 else "low",
                    }
                )
        anomalies.sort(key=lambda row: abs(row["robust_z"]), reverse=True)
        return anomalies[:5]

    def summary(self, *, collect_if_missing: bool = True) -> dict[str, Any]:
        runtime = self.runtime_evidence()
        if not self.enabled:
            return {"enabled": False, "system_health": "unknown", "runtime": runtime}
        dynamic = (
            self._ensure_dynamic()
            if collect_if_missing
            else self.store.latest_snapshot("dynamic")
        ) or {}
        inventory = self.store.latest_snapshot("inventory") or {}
        if not dynamic:
            return {
                "enabled": True,
                "captured_at": None,
                "system_health": "unknown",
                "inference": self.inference_summary(),
                "anomalies": [],
                "probable_causes": [],
                "background_resource_contention": [],
                "runtime": runtime,
            }
        metrics = {key: value for key, value, _unit, _source in self._metric_rows(dynamic)}
        baseline = self.baselines()
        anomalies = self._anomalies(metrics, baseline)
        cpu = _finite(_path_get(dynamic, "base.cpu.percent"))
        memory = _finite(_path_get(dynamic, "base.memory.percent"))
        gpu = _finite(_path_get(dynamic, "derived.gpu_utilization_percent"))
        temperature = _finite(_path_get(dynamic, "derived.max_temperature_c"))
        volumes = _path_get(dynamic, "base.storage.volumes") or []
        min_free = min(
            (_finite(row.get("free_percent")) for row in volumes if _finite(row.get("free_percent")) is not None),
            default=None,
        )
        perf_limit = _finite(
            _path_get(dynamic, "performance.processor.PercentPerformanceLimit")
        )
        perf_percent = _finite(
            _path_get(dynamic, "performance.processor.PercentProcessorPerformance")
        )
        throttling = bool(
            (perf_limit is not None and perf_limit < 99)
            or (cpu is not None and cpu >= 80 and perf_percent is not None and perf_percent < 70)
        )
        if temperature is None:
            thermal_state = "unknown"
        elif temperature >= 95:
            thermal_state = "critical"
        elif temperature >= 85:
            thermal_state = "high"
        elif temperature >= 70:
            thermal_state = "moderate"
        else:
            thermal_state = "normal"
        inventory_summary = inventory.get("summary") or {}
        critical = any(
            value == "critical"
            for value in (_pressure(cpu), _pressure(memory), thermal_state)
        )
        degraded = bool(
            critical
            or throttling
            or inventory_summary.get("devices_problematic")
            or any(abs(item["robust_z"]) >= 5 for item in anomalies)
        )
        top_processes = (_path_get(dynamic, "base.top_processes") or [])[:5]
        inference = self.inference_summary()
        battery = _path_get(dynamic, "base.battery")
        active_power_plan = _path_get(dynamic, "performance.power_plan.ElementName")
        probable_causes = []
        deviation = _finite(inference.get("deviation_percent"))
        if deviation is not None and deviation <= -10:
            if _pressure(memory) in {"high", "critical"}:
                probable_causes.append("model slowdown correlates with high system-memory pressure")
            if gpu is not None and gpu >= 90:
                probable_causes.append("model slowdown coincides with saturated GPU activity")
            if throttling:
                probable_causes.append("model slowdown coincides with processor performance limiting")
            if not probable_causes:
                probable_causes.append("model inference is below baseline; no single resource cause is yet established")
        return {
            "enabled": True,
            "captured_at": dynamic.get("captured_at"),
            "system_health": "critical" if critical else "degraded" if degraded else "good",
            "compute_pressure": _pressure(max(value for value in (cpu, gpu) if value is not None) if any(value is not None for value in (cpu, gpu)) else None),
            "cpu_percent": cpu,
            "gpu_percent": gpu,
            "memory_pressure": _pressure(memory),
            "memory_percent": memory,
            "storage_pressure": "high" if min_free is not None and min_free < 10 else "moderate" if min_free is not None and min_free < 20 else "low" if min_free is not None else "unknown",
            "minimum_volume_free_percent": min_free,
            "thermal_state": thermal_state,
            "max_temperature_c": temperature,
            "thermal_margin_c": round(95.0 - temperature, 1) if temperature is not None else None,
            "throttling_detected": throttling,
            "power_state": {
                "active_plan": active_power_plan,
                "source": (
                    "ac" if isinstance(battery, dict) and battery.get("plugged_in")
                    else "battery" if isinstance(battery, dict)
                    else "unknown"
                ),
                "battery_percent": battery.get("percent") if isinstance(battery, dict) else None,
            },
            "vram_used_mb": _finite(_path_get(dynamic, "derived.vram_used_mb")),
            "device_problems": int(inventory_summary.get("devices_problematic") or 0),
            "disabled_devices": int(inventory_summary.get("devices_disabled") or 0),
            "hidden_or_disconnected_devices": int(inventory_summary.get("devices_hidden_or_disconnected") or 0),
            "inference": inference,
            "anomalies": anomalies,
            "probable_causes": probable_causes[:3],
            "background_resource_contention": top_processes,
            "sensor_source": _path_get(dynamic, "sensors.source"),
            "sensor_source_available": bool(_path_get(dynamic, "sensors.available")),
            "runtime": runtime,
        }

    def compact_context(self) -> str:
        if not self.enabled or not self.config.compact_context_enabled:
            return ""
        # The runtime worker samples before serving turns. Avoid turning passive
        # context injection into an on-demand collector call in short-lived tests
        # or alternate entry points where the service is not running.
        if self.store.latest_snapshot("dynamic") is None and self._runtime_evidence is None:
            return ""
        state = self.summary(collect_if_missing=False)
        compact = {
            key: state.get(key)
            for key in (
                "captured_at",
                "system_health",
                "compute_pressure",
                "cpu_percent",
                "gpu_percent",
                "memory_pressure",
                "memory_percent",
                "storage_pressure",
                "minimum_volume_free_percent",
                "thermal_state",
                "thermal_margin_c",
                "throttling_detected",
                "power_state",
                "vram_used_mb",
                "device_problems",
                "inference",
                "anomalies",
                "probable_causes",
                "runtime",
            )
        }
        return (
            "SYSTEMSENSE PASSIVE STATE (read-only, derived, current-turn only):\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            + "\nUse SystemSense read-only tools only when raw inventory, drivers, sensors, history, or correlations are needed. "
            "Treat driver candidates and correlations as investigative signals, not deletion authority or proof of causality."
        )

    def record_inference(self, runtime: dict[str, Any], *, model: str) -> None:
        if not self.enabled:
            return
        eval_count = _finite(runtime.get("eval_count"))
        eval_duration = _finite(runtime.get("eval_duration"))
        prompt_duration = _finite(runtime.get("prompt_eval_duration"))
        load_duration = _finite(runtime.get("load_duration"))
        total_duration = _finite(runtime.get("total_duration"))
        tokens_per_second = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_count is not None and eval_duration and eval_duration > 0
            else None
        )
        dynamic = self.store.latest_snapshot("dynamic") or {}
        environment = {
            key: value
            for key, value, _unit, _source in self._metric_rows(dynamic)
        }
        payload = {
            "captured_at": utc_timestamp(),
            "model": model,
            "phase": runtime.get("phase"),
            "tokens_per_second": round(tokens_per_second, 4) if tokens_per_second is not None else None,
            "time_to_first_token_ms": round(((load_duration or 0) + (prompt_duration or 0)) / 1_000_000, 3) if (load_duration is not None or prompt_duration is not None) else None,
            "total_latency_ms": round(total_duration / 1_000_000, 3) if total_duration is not None else None,
            "prompt_tokens": int(runtime["prompt_eval_count"]) if runtime.get("prompt_eval_count") is not None else None,
            "eval_tokens": int(runtime["eval_count"]) if runtime.get("eval_count") is not None else None,
            "context_percent": _finite(runtime.get("context_used_percent")),
            "environment": environment,
            "runtime_classification": runtime.get("runtime_classification"),
        }
        self.store.save_inference(payload)

    def inference_summary(self) -> dict[str, Any]:
        since = (_utc_now() - timedelta(days=self.config.correlation_window_days)).isoformat()
        rows = self.store.inference_rows(since=since)
        valid = [row for row in rows if _finite(row.get("tokens_per_second")) is not None]
        if not valid:
            return {"samples": 0, "current_tokens_per_second": None, "baseline_tokens_per_second": None, "deviation_percent": None}
        current = float(valid[0]["tokens_per_second"])
        values = [float(row["tokens_per_second"]) for row in valid]
        baseline = statistics.median(values)
        deviation = 100.0 * (current - baseline) / baseline if baseline > 0 else None
        return {
            "samples": len(values),
            "current_tokens_per_second": round(current, 3),
            "baseline_tokens_per_second": round(baseline, 3),
            "deviation_percent": round(deviation, 2) if deviation is not None else None,
            "time_to_first_token_ms": valid[0].get("time_to_first_token_ms"),
            "total_latency_ms": valid[0].get("total_latency_ms"),
        }

    def correlations(self, *, limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(int(limit), 30))
        since = (_utc_now() - timedelta(days=self.config.correlation_window_days)).isoformat()
        rows = self.store.inference_rows(since=since)
        pairs: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            speed = _finite(row.get("tokens_per_second"))
            if speed is None:
                continue
            for key, value in (row.get("environment") or {}).items():
                number = _finite(value)
                if number is not None:
                    pairs.setdefault(str(key), []).append((number, speed))
        results = []
        for key, values in pairs.items():
            if len(values) < 5:
                continue
            xs = [item[0] for item in values]
            ys = [item[1] for item in values]
            if len(set(xs)) < 2 or len(set(ys)) < 2:
                continue
            coefficient = statistics.correlation(xs, ys)
            results.append(
                {
                    "environment_metric": key,
                    "inference_metric": "tokens_per_second",
                    "pearson_r": round(coefficient, 4),
                    "samples": len(values),
                    "relationship": "positive" if coefficient > 0 else "negative",
                }
            )
        results.sort(key=lambda row: abs(row["pearson_r"]), reverse=True)
        return {
            "window_days": self.config.correlation_window_days,
            "correlations": results[:limit],
            "warning": "Correlation is observational and does not establish causality.",
        }

    def inventory(self, *, section: str = "overview", limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        payload = self.store.latest_snapshot("inventory")
        if payload is None:
            try:
                payload = self.collect_inventory()
            except Exception as exc:
                return {"available": False, "error_type": type(exc).__name__}
        allowed = {
            "overview",
            "computer_system",
            "computer_product",
            "motherboard",
            "bios",
            "cpu",
            "gpu",
            "memory_modules",
            "storage_devices",
            "network_adapters",
            "devices",
            "system_drivers",
        }
        if section not in allowed:
            raise ValueError(f"section must be one of: {', '.join(sorted(allowed))}")
        if section == "overview":
            keys = ("captured_at", "available", "platform", "summary", "computer_system", "motherboard", "bios", "cpu", "gpu")
            return {key: payload.get(key) for key in keys}
        rows = payload.get(section) or []
        return {"section": section, "count": len(rows), "items": rows[:limit]}

    def drivers(self, *, classification: str = "all", limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(int(limit), 300))
        payload = self.store.latest_snapshot("inventory")
        if payload is None:
            try:
                payload = self.collect_inventory()
            except Exception as exc:
                return {"available": False, "error_type": type(exc).__name__}
        rows = list(payload.get("drivers") or []) + list(payload.get("driver_packages") or [])
        classes = {
            "all",
            "active_bound",
            "installed_bound_inactive",
            "problematic",
            "association_unknown",
            "orphan_candidate_requires_review",
            "duplicate_older_candidate_requires_review",
        }
        if classification not in classes:
            raise ValueError(f"classification must be one of: {', '.join(sorted(classes))}")
        if classification != "all":
            rows = [row for row in rows if row.get("classification") == classification]
        return {
            "classification": classification,
            "count": len(rows),
            "items": rows[:limit],
            "warning": (payload.get("summary") or {}).get("classification_warning"),
        }

    def history(self, *, metric: str, hours: float = 1.0, limit: int = 120) -> dict[str, Any]:
        if metric not in self._BASELINE_KEYS and metric not in {
            "processor.performance_limit_percent",
            "processor.performance_percent",
        }:
            raise ValueError("metric is not an exposed SystemSense metric")
        hours = max(0.1, min(float(hours), self.config.retention_days * 24.0))
        limit = max(1, min(int(limit), 1000))
        since = (_utc_now() - timedelta(hours=hours)).isoformat()
        rows = self.store.metric_history(metric, since=since, limit=limit)
        return {"metric": metric, "hours": hours, "samples": len(rows), "items": rows}

    def raw(self, *, category: str = "dynamic", limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(int(limit), 500))
        if category == "dynamic":
            payload = self._ensure_dynamic() or {}
            payload = dict(payload)
            sensors = (payload.get("raw_sensors") or {}).get("sensors") or []
            if payload.get("raw_sensors"):
                payload["raw_sensors"] = dict(payload["raw_sensors"])
                payload["raw_sensors"]["sensors"] = sensors[:limit]
            return payload
        if category == "sensors":
            payload = self._ensure_dynamic() or {}
            rows = (payload.get("raw_sensors") or {}).get("sensors") or []
            return {"count": len(rows), "items": rows[:limit]}
        if category == "inventory":
            payload = self.store.latest_snapshot("inventory") or {}
            return {
                "captured_at": payload.get("captured_at"),
                "sections": {
                    key: (value[:limit] if isinstance(value, list) else value)
                    for key, value in payload.items()
                    if key not in {"drivers", "driver_packages"}
                },
            }
        raise ValueError("category must be dynamic, sensors, or inventory")


_INSTANCES: dict[Path, SystemSense] = {}
_INSTANCES_LOCK = threading.Lock()


def get_system_sense(
    config: SystemSenseConfig,
    data_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    main_branch: str = "main",
) -> SystemSense:
    database = (Path(data_dir).resolve() / config.database).resolve()
    with _INSTANCES_LOCK:
        instance = _INSTANCES.get(database)
        if instance is None:
            instance = SystemSense(
                config,
                data_dir,
                project_root=project_root,
                main_branch=main_branch,
            )
            _INSTANCES[database] = instance
        elif project_root is not None:
            instance.configure_runtime_evidence(project_root, main_branch=main_branch)
        return instance
