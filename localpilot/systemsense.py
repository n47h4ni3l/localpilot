from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import statistics
import threading
import time

import psutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from localpilot.config import SystemSenseConfig
from localpilot.runtime_evidence import RuntimeEvidence
from localpilot.process_identity import inspect_executable_artifact
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
                CREATE TABLE IF NOT EXISTS memory_watches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    label TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_watches_status_time
                    ON memory_watches(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS memory_watch_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watch_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    system_memory_percent REAL NOT NULL,
                    system_available_gb REAL,
                    FOREIGN KEY(watch_id) REFERENCES memory_watches(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_watch_samples_watch_time
                    ON memory_watch_samples(watch_id, captured_at DESC);
                CREATE TABLE IF NOT EXISTS memory_watch_process_samples (
                    sample_id INTEGER NOT NULL,
                    pid INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    ram_mb REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    executable TEXT,
                    command_line TEXT,
                    parent_pid INTEGER,
                    parent_name TEXT,
                    parent_executable TEXT,
                    started_at_epoch REAL,
                    username TEXT,
                    FOREIGN KEY(sample_id) REFERENCES memory_watch_samples(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_watch_process_name
                    ON memory_watch_process_samples(name, ram_mb DESC);

                CREATE TABLE IF NOT EXISTS systemsense_watches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    label TEXT NOT NULL,
                    profile TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_systemsense_watches_status_time
                    ON systemsense_watches(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS systemsense_watch_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watch_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    system_available_gb REAL,
                    storage_read_mb_s REAL,
                    storage_write_mb_s REAL,
                    network_send_mbps REAL,
                    network_receive_mbps REAL,
                    gpu_percent REAL,
                    vram_used_mb REAL,
                    FOREIGN KEY(watch_id) REFERENCES systemsense_watches(id)
                );
                CREATE INDEX IF NOT EXISTS idx_systemsense_watch_samples_time
                    ON systemsense_watch_samples(watch_id, captured_at DESC);

                CREATE TABLE IF NOT EXISTS systemsense_executable_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER,
                    modified_ns INTEGER,
                    sha256 TEXT,
                    company_name TEXT,
                    product_name TEXT,
                    file_description TEXT,
                    file_version TEXT,
                    product_version TEXT,
                    original_filename TEXT,
                    signature_status TEXT,
                    signer_subject TEXT,
                    signer_issuer TEXT,
                    UNIQUE(path, size_bytes, modified_ns)
                );

                CREATE TABLE IF NOT EXISTS systemsense_watch_process_samples (
                    sample_id INTEGER NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at_epoch REAL,
                    name TEXT NOT NULL,
                    cpu_percent REAL,
                    rss_mb REAL,
                    vms_mb REAL,
                    private_mb REAL,
                    pagefile_mb REAL,
                    threads INTEGER,
                    handles INTEGER,
                    page_faults INTEGER,
                    io_read_mb REAL,
                    io_write_mb REAL,
                    io_read_mb_s REAL,
                    io_write_mb_s REAL,
                    gpu_percent REAL,
                    gpu_dedicated_mb REAL,
                    gpu_shared_mb REAL,
                    gpu_committed_mb REAL,
                    executable TEXT,
                    command_line TEXT,
                    parent_pid INTEGER,
                    parent_name TEXT,
                    parent_executable TEXT,
                    ancestry_json TEXT,
                    username TEXT,
                    artifact_id INTEGER,
                    FOREIGN KEY(sample_id) REFERENCES systemsense_watch_samples(id),
                    FOREIGN KEY(artifact_id) REFERENCES systemsense_executable_artifacts(id)
                );
                CREATE INDEX IF NOT EXISTS idx_systemsense_watch_process_ram
                    ON systemsense_watch_process_samples(rss_mb DESC);
                CREATE INDEX IF NOT EXISTS idx_systemsense_watch_process_pid
                    ON systemsense_watch_process_samples(pid, started_at_epoch);

                CREATE TABLE IF NOT EXISTS systemsense_watch_network_samples (
                    sample_id INTEGER NOT NULL,
                    pid INTEGER NOT NULL,
                    family TEXT,
                    socket_type TEXT,
                    local_endpoint TEXT,
                    remote_endpoint TEXT,
                    status TEXT,
                    FOREIGN KEY(sample_id) REFERENCES systemsense_watch_samples(id)
                );
                CREATE INDEX IF NOT EXISTS idx_systemsense_watch_network_pid
                    ON systemsense_watch_network_samples(pid, sample_id);

                CREATE TABLE IF NOT EXISTS systemsense_watch_lifecycle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watch_id INTEGER NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at_epoch REAL,
                    event TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    name TEXT,
                    parent_pid INTEGER,
                    details_json TEXT,
                    FOREIGN KEY(watch_id) REFERENCES systemsense_watches(id)
                );
                CREATE INDEX IF NOT EXISTS idx_systemsense_watch_lifecycle_time
                    ON systemsense_watch_lifecycle(watch_id, event_at DESC);
                """
            )
            self._migrate_schema(connection)

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        if not SystemSenseStore._table_exists(connection, table):
            return set()
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        """Versioned, idempotent migrations for SystemSense's private database."""
        version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)

        if version < 1:
            # Older PR160 builds used memory-specific tables. Preserve any
            # samples already collected by copying their stable core fields into
            # the generic SystemSense watch schema once.
            if (
                self._table_exists(connection, "memory_watches")
                and connection.execute(
                    "SELECT COUNT(*) FROM systemsense_watches"
                ).fetchone()[0] == 0
            ):
                connection.execute(
                    "INSERT INTO systemsense_watches("
                    "id, created_at, expires_at, status, label, profile"
                    ") SELECT id, created_at, expires_at, status, label, 'memory' "
                    "FROM memory_watches"
                )
                if self._table_exists(connection, "memory_watch_samples"):
                    connection.execute(
                        "INSERT INTO systemsense_watch_samples("
                        "id, watch_id, captured_at, memory_percent, system_available_gb"
                        ") SELECT id, watch_id, captured_at, system_memory_percent, "
                        "system_available_gb FROM memory_watch_samples"
                    )
                if self._table_exists(connection, "memory_watch_process_samples"):
                    columns = self._columns(connection, "memory_watch_process_samples")
                    def expression(name: str, fallback: str = "NULL") -> str:
                        return name if name in columns else fallback
                    connection.execute(
                        "INSERT INTO systemsense_watch_process_samples("
                        "sample_id, pid, started_at_epoch, name, cpu_percent, rss_mb, "
                        "executable, command_line, parent_pid, parent_name, "
                        "parent_executable, username"
                        ") SELECT sample_id, pid, "
                        + expression("started_at_epoch")
                        + ", name, cpu_percent, ram_mb, "
                        + expression("executable")
                        + ", "
                        + expression("command_line")
                        + ", "
                        + expression("parent_pid")
                        + ", "
                        + expression("parent_name")
                        + ", "
                        + expression("parent_executable")
                        + ", "
                        + expression("username")
                        + " FROM memory_watch_process_samples"
                    )
            connection.execute("PRAGMA user_version=1")
            version = 1

        if version < 2:
            # v2 establishes the explicit migration boundary for the generic
            # watch/process/network/artifact schema. CREATE TABLE IF NOT EXISTS
            # above makes this safe for both new and existing databases.
            connection.execute("PRAGMA user_version=2")

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

    def start_watch(
        self,
        *,
        profile: str,
        expires_at: str,
        label: str,
    ) -> dict[str, Any]:
        profile = str(profile).strip().casefold()
        if profile not in {"memory", "cpu", "storage", "network", "gpu", "system"}:
            raise ValueError("unsupported SystemSense watch profile")
        created_at = utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE systemsense_watches SET status='replaced' "
                "WHERE status='active' AND profile=?",
                (profile,),
            )
            cursor = connection.execute(
                "INSERT INTO systemsense_watches("
                "created_at, expires_at, status, label, profile"
                ") VALUES (?, ?, 'active', ?, ?)",
                (created_at, expires_at, str(label)[:80], profile),
            )
            watch_id = int(cursor.lastrowid)
        return {
            "watch_id": watch_id,
            "created_at": created_at,
            "expires_at": expires_at,
            "status": "active",
            "label": str(label)[:80],
            "profile": profile,
        }

    def active_watches(self, *, now: str | None = None) -> list[dict[str, Any]]:
        current = now or utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE systemsense_watches SET status='completed' "
                "WHERE status='active' AND expires_at<=?",
                (current,),
            )
            rows = connection.execute(
                "SELECT id, created_at, expires_at, status, label, profile "
                "FROM systemsense_watches WHERE status='active' "
                "ORDER BY created_at ASC"
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["watch_id"] = int(item.pop("id"))
            output.append(item)
        return output

    def latest_watch(
        self,
        *,
        watch_id: int | None = None,
        profile: str | None = None,
    ) -> dict[str, Any] | None:
        self.active_watches()
        clauses = []
        params: list[Any] = []
        if watch_id is not None:
            clauses.append("id=?")
            params.append(int(watch_id))
        if profile:
            clauses.append("profile=?")
            params.append(str(profile).casefold())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at, expires_at, status, label, profile "
                "FROM systemsense_watches"
                + where
                + " ORDER BY created_at DESC LIMIT 1",
                params,
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["watch_id"] = int(item.pop("id"))
        return item

    def stop_watch(
        self,
        *,
        watch_id: int | None = None,
        profile: str | None = None,
    ) -> dict[str, Any] | None:
        watch = self.latest_watch(watch_id=watch_id, profile=profile)
        if watch is None or watch.get("status") != "active":
            return None
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE systemsense_watches SET status='stopped' WHERE id=?",
                (int(watch["watch_id"]),),
            )
        watch["status"] = "stopped"
        return watch

    def save_watch_sample(
        self,
        *,
        watch_id: int,
        captured_at: str,
        system: dict[str, Any],
    ) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO systemsense_watch_samples("
                "watch_id, captured_at, cpu_percent, memory_percent, "
                "system_available_gb, storage_read_mb_s, storage_write_mb_s, "
                "network_send_mbps, network_receive_mbps, gpu_percent, vram_used_mb"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(watch_id),
                    captured_at,
                    _finite(system.get("cpu_percent")),
                    _finite(system.get("memory_percent")),
                    _finite(system.get("system_available_gb")),
                    _finite(system.get("storage_read_mb_s")),
                    _finite(system.get("storage_write_mb_s")),
                    _finite(system.get("network_send_mbps")),
                    _finite(system.get("network_receive_mbps")),
                    _finite(system.get("gpu_percent")),
                    _finite(system.get("vram_used_mb")),
                ),
            )
            return int(cursor.lastrowid)

    def upsert_executable_artifact(self, payload: dict[str, Any]) -> int | None:
        if not payload.get("available") or not payload.get("path"):
            return None
        path = str(payload["path"])[:2000]
        size = payload.get("size_bytes")
        modified = payload.get("modified_ns")
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO systemsense_executable_artifacts("
                "captured_at, path, size_bytes, modified_ns, sha256, company_name, "
                "product_name, file_description, file_version, product_version, "
                "original_filename, signature_status, signer_subject, signer_issuer"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_timestamp(),
                    path,
                    size,
                    modified,
                    payload.get("sha256"),
                    payload.get("company_name"),
                    payload.get("product_name"),
                    payload.get("file_description"),
                    payload.get("file_version"),
                    payload.get("product_version"),
                    payload.get("original_filename"),
                    payload.get("signature_status"),
                    payload.get("signer_subject"),
                    payload.get("signer_issuer"),
                ),
            )
            row = connection.execute(
                "SELECT id FROM systemsense_executable_artifacts "
                "WHERE path=? AND size_bytes IS ? AND modified_ns IS ? "
                "ORDER BY id DESC LIMIT 1",
                (path, size, modified),
            ).fetchone()
        return int(row["id"]) if row else None

    def save_watch_process_samples(
        self,
        *,
        sample_id: int,
        processes: list[dict[str, Any]],
    ) -> None:
        rows = []
        for process in processes:
            try:
                rows.append(
                    (
                        int(sample_id),
                        int(process.get("pid") or 0),
                        _finite(process.get("started_at_epoch")),
                        str(process.get("name") or "unknown")[:200],
                        _finite(process.get("cpu_percent")),
                        _finite(process.get("ram_mb")),
                        _finite(process.get("vms_mb")),
                        _finite(process.get("private_mb")),
                        _finite(process.get("pagefile_mb")),
                        int(process.get("threads") or 0),
                        (
                            int(process["handles"])
                            if process.get("handles") is not None
                            else None
                        ),
                        int(process.get("page_faults") or 0),
                        _finite(process.get("io_read_mb")),
                        _finite(process.get("io_write_mb")),
                        _finite(process.get("io_read_mb_s")),
                        _finite(process.get("io_write_mb_s")),
                        _finite(process.get("gpu_percent")),
                        _finite(process.get("gpu_dedicated_mb")),
                        _finite(process.get("gpu_shared_mb")),
                        _finite(process.get("gpu_committed_mb")),
                        str(process.get("executable") or "")[:1000],
                        str(process.get("command_line") or "")[:4000],
                        int(process.get("parent_pid") or 0),
                        str(process.get("parent_name") or "")[:200],
                        str(process.get("parent_executable") or "")[:1000],
                        _json(process.get("ancestry") or []),
                        str(process.get("username") or "")[:300],
                        (
                            int(process["artifact_id"])
                            if process.get("artifact_id") is not None
                            else None
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
        if not rows:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO systemsense_watch_process_samples("
                "sample_id, pid, started_at_epoch, name, cpu_percent, rss_mb, "
                "vms_mb, private_mb, pagefile_mb, threads, handles, page_faults, "
                "io_read_mb, io_write_mb, io_read_mb_s, io_write_mb_s, "
                "gpu_percent, gpu_dedicated_mb, gpu_shared_mb, gpu_committed_mb, "
                "executable, command_line, parent_pid, parent_name, "
                "parent_executable, ancestry_json, username, artifact_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def save_watch_network_samples(
        self,
        *,
        sample_id: int,
        connections: list[dict[str, Any]],
    ) -> None:
        rows = [
            (
                int(sample_id),
                int(item.get("pid") or 0),
                str(item.get("family") or "")[:100],
                str(item.get("type") or "")[:100],
                str(item.get("local_endpoint") or "")[:500],
                str(item.get("remote_endpoint") or "")[:500],
                str(item.get("status") or "")[:100],
            )
            for item in connections
            if int(item.get("pid") or 0) > 0
        ]
        if rows:
            with self._lock, self._connect() as connection:
                connection.executemany(
                    "INSERT INTO systemsense_watch_network_samples("
                    "sample_id, pid, family, socket_type, local_endpoint, "
                    "remote_endpoint, status"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

    def save_watch_lifecycle(
        self,
        *,
        watch_id: int,
        pid: int,
        started_at_epoch: float | None,
        event: str,
        event_at: str,
        name: str,
        parent_pid: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO systemsense_watch_lifecycle("
                "watch_id, pid, started_at_epoch, event, event_at, name, "
                "parent_pid, details_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(watch_id),
                    int(pid),
                    started_at_epoch,
                    str(event)[:40],
                    event_at,
                    str(name)[:200],
                    parent_pid,
                    _json(details or {}),
                ),
            )

    def watch_process_identity(
        self,
        *,
        pid: int,
        watch_id: int | None = None,
    ) -> dict[str, Any]:
        watch = self.latest_watch(watch_id=watch_id)
        if watch is None:
            return {"available": False, "reason": "no_systemsense_watch"}
        wid = int(watch["watch_id"])
        with self._connect() as connection:
            row = connection.execute(
                "SELECT s.captured_at, p.*, a.sha256, a.size_bytes, a.modified_ns, "
                "a.company_name, a.product_name, a.file_description, a.file_version, "
                "a.product_version, a.original_filename, a.signature_status, "
                "a.signer_subject, a.signer_issuer "
                "FROM systemsense_watch_process_samples p "
                "JOIN systemsense_watch_samples s ON s.id=p.sample_id "
                "LEFT JOIN systemsense_executable_artifacts a ON a.id=p.artifact_id "
                "WHERE s.watch_id=? AND p.pid=? "
                "ORDER BY COALESCE(p.rss_mb,0) DESC, s.captured_at DESC LIMIT 1",
                (wid, int(pid)),
            ).fetchone()
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM systemsense_watch_process_samples p "
                "JOIN systemsense_watch_samples s ON s.id=p.sample_id "
                "WHERE s.watch_id=? AND p.pid=?",
                (wid, int(pid)),
            ).fetchone()
            lifecycle = connection.execute(
                "SELECT event, event_at, name, parent_pid, details_json "
                "FROM systemsense_watch_lifecycle WHERE watch_id=? AND pid=? "
                "ORDER BY event_at ASC LIMIT 50",
                (wid, int(pid)),
            ).fetchall()
            network = connection.execute(
                "SELECT n.local_endpoint, n.remote_endpoint, n.status, n.family, n.socket_type "
                "FROM systemsense_watch_network_samples n "
                "JOIN systemsense_watch_samples s ON s.id=n.sample_id "
                "WHERE s.watch_id=? AND n.pid=? "
                "GROUP BY n.local_endpoint, n.remote_endpoint, n.status, n.family, n.socket_type "
                "ORDER BY MAX(s.captured_at) DESC LIMIT 100",
                (wid, int(pid)),
            ).fetchall()
        if row is None:
            return {
                "available": False,
                "reason": "pid_not_observed_in_watch",
                "watch_id": wid,
                "pid": int(pid),
            }
        output = dict(row)
        output["available"] = True
        output["watch_id"] = wid
        output["watch_profile"] = watch["profile"]
        output["observations"] = int(count["count"] or 0)
        try:
            output["ancestry"] = json.loads(output.pop("ancestry_json") or "[]")
        except json.JSONDecodeError:
            output["ancestry"] = []
        output["lifecycle"] = [
            {
                **{key: item[key] for key in ("event", "event_at", "name", "parent_pid")},
                "details": json.loads(item["details_json"] or "{}"),
            }
            for item in lifecycle
        ]
        output["network_connections"] = [dict(item) for item in network]
        return output

    def watch_report(
        self,
        *,
        watch_id: int | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        watch = self.latest_watch(watch_id=watch_id, profile=profile)
        if watch is None:
            return {"available": False, "reason": "no_systemsense_watch"}
        wid = int(watch["watch_id"])
        with self._connect() as connection:
            stats = connection.execute(
                "SELECT COUNT(*) AS samples, MIN(captured_at) AS first_sample_at, "
                "MAX(captured_at) AS last_sample_at, AVG(cpu_percent) AS avg_cpu, "
                "MAX(cpu_percent) AS peak_cpu, AVG(memory_percent) AS avg_memory, "
                "MAX(memory_percent) AS peak_memory, MIN(system_available_gb) AS min_available, "
                "MAX(storage_read_mb_s) AS peak_read, MAX(storage_write_mb_s) AS peak_write, "
                "MAX(network_send_mbps) AS peak_send, MAX(network_receive_mbps) AS peak_receive, "
                "MAX(gpu_percent) AS peak_gpu, MAX(vram_used_mb) AS peak_vram "
                "FROM systemsense_watch_samples WHERE watch_id=?",
                (wid,),
            ).fetchone()
            processes = connection.execute(
                "SELECT p.pid, p.started_at_epoch, p.name, p.executable, p.command_line, "
                "p.parent_pid, p.parent_name, p.parent_executable, p.username, "
                "COUNT(*) AS observations, MAX(p.rss_mb) AS peak_ram_mb, "
                "AVG(p.rss_mb) AS average_observed_ram_mb, MAX(p.private_mb) AS peak_private_mb, "
                "MAX(p.pagefile_mb) AS peak_pagefile_mb, MAX(p.cpu_percent) AS peak_cpu_percent, "
                "MAX(p.io_read_mb_s) AS peak_read_mb_s, MAX(p.io_write_mb_s) AS peak_write_mb_s, "
                "MAX(p.handles) AS peak_handles, MAX(p.threads) AS peak_threads, "
                "MAX(p.page_faults) AS peak_page_faults, MAX(p.gpu_percent) AS peak_gpu_percent, "
                "MAX(p.gpu_dedicated_mb) AS peak_gpu_dedicated_mb, "
                "MAX(p.gpu_shared_mb) AS peak_gpu_shared_mb, "
                "MAX(p.gpu_committed_mb) AS peak_gpu_committed_mb, "
                "MAX(a.sha256) AS sha256, MAX(a.signature_status) AS signature_status, "
                "MAX(a.signer_subject) AS signer_subject, MAX(a.company_name) AS company_name, "
                "MAX(a.product_name) AS product_name "
                "FROM systemsense_watch_process_samples p "
                "JOIN systemsense_watch_samples s ON s.id=p.sample_id "
                "LEFT JOIN systemsense_executable_artifacts a ON a.id=p.artifact_id "
                "WHERE s.watch_id=? "
                "GROUP BY p.pid, p.started_at_epoch, p.name, p.executable, p.command_line, "
                "p.parent_pid, p.parent_name, p.parent_executable, p.username "
                "ORDER BY MAX(COALESCE(p.rss_mb,0)) DESC LIMIT 40",
                (wid,),
            ).fetchall()
            lifecycle = connection.execute(
                "SELECT pid, started_at_epoch, event, event_at, name, parent_pid "
                "FROM systemsense_watch_lifecycle WHERE watch_id=? "
                "ORDER BY event_at ASC LIMIT 500",
                (wid,),
            ).fetchall()
            network = connection.execute(
                "SELECT n.pid, COUNT(*) AS observations, "
                "COUNT(DISTINCT COALESCE(n.remote_endpoint,'')) AS remote_endpoint_count "
                "FROM systemsense_watch_network_samples n "
                "JOIN systemsense_watch_samples s ON s.id=n.sample_id "
                "WHERE s.watch_id=? GROUP BY n.pid "
                "ORDER BY observations DESC LIMIT 40",
                (wid,),
            ).fetchall()

        process_rows = [dict(row) for row in processes]
        result = {
            "available": True,
            "watch": watch,
            "samples": int(stats["samples"] or 0),
            "first_sample_at": stats["first_sample_at"],
            "last_sample_at": stats["last_sample_at"],
            "system": {
                "average_cpu_percent": _finite(stats["avg_cpu"]),
                "peak_cpu_percent": _finite(stats["peak_cpu"]),
                "average_memory_percent": _finite(stats["avg_memory"]),
                "peak_memory_percent": _finite(stats["peak_memory"]),
                "minimum_available_gb": _finite(stats["min_available"]),
                "peak_storage_read_mb_s": _finite(stats["peak_read"]),
                "peak_storage_write_mb_s": _finite(stats["peak_write"]),
                "peak_network_send_mbps": _finite(stats["peak_send"]),
                "peak_network_receive_mbps": _finite(stats["peak_receive"]),
                "peak_gpu_percent": _finite(stats["peak_gpu"]),
                "peak_vram_used_mb": _finite(stats["peak_vram"]),
            },
            "top_processes": process_rows,
            "process_lifecycle": [dict(row) for row in lifecycle],
            "process_network_summary": [dict(row) for row in network],
            "limitations": [
                "Network attribution records PID-to-endpoint/socket ownership, not per-process byte counts.",
                "A process absent from the bounded sampled process set may still have been running.",
                "GPU process counters depend on Windows exposing GPU Performance Counters for that workload.",
            ],
        }
        if watch["profile"] == "memory":
            result.update(
                average_system_memory_percent=_finite(stats["avg_memory"]),
                peak_system_memory_percent=_finite(stats["peak_memory"]),
                minimum_available_gb=_finite(stats["min_available"]),
                top_memory_consumers=process_rows,
            )
        return result

    def prune_watches(self, *, before: str) -> None:
        with self._lock, self._connect() as connection:
            old_ids = [
                int(row["id"])
                for row in connection.execute(
                    "SELECT id FROM systemsense_watches "
                    "WHERE created_at<? AND status!='active'",
                    (before,),
                ).fetchall()
            ]
            if not old_ids:
                return
            placeholders = ",".join("?" for _ in old_ids)
            sample_ids = [
                int(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM systemsense_watch_samples "
                    f"WHERE watch_id IN ({placeholders})",
                    old_ids,
                ).fetchall()
            ]
            if sample_ids:
                sample_placeholders = ",".join("?" for _ in sample_ids)
                for table in (
                    "systemsense_watch_process_samples",
                    "systemsense_watch_network_samples",
                ):
                    connection.execute(
                        f"DELETE FROM {table} WHERE sample_id IN ({sample_placeholders})",
                        sample_ids,
                    )
            connection.execute(
                f"DELETE FROM systemsense_watch_lifecycle WHERE watch_id IN ({placeholders})",
                old_ids,
            )
            connection.execute(
                f"DELETE FROM systemsense_watch_samples WHERE watch_id IN ({placeholders})",
                old_ids,
            )
            connection.execute(
                f"DELETE FROM systemsense_watches WHERE id IN ({placeholders})",
                old_ids,
            )
            connection.execute(
                "DELETE FROM systemsense_executable_artifacts WHERE id NOT IN ("
                "SELECT DISTINCT artifact_id FROM systemsense_watch_process_samples "
                "WHERE artifact_id IS NOT NULL)"
            )

    def start_memory_watch(self, *, expires_at: str, label: str) -> dict[str, Any]:
        created_at = utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_watches SET status='replaced' WHERE status='active'"
            )
            cursor = connection.execute(
                "INSERT INTO memory_watches(created_at, expires_at, status, label) "
                "VALUES (?, ?, 'active', ?)",
                (created_at, expires_at, str(label)[:80]),
            )
            watch_id = int(cursor.lastrowid)
        return {
            "watch_id": watch_id,
            "created_at": created_at,
            "expires_at": expires_at,
            "status": "active",
            "label": str(label)[:80],
        }

    def active_memory_watch(self, *, now: str | None = None) -> dict[str, Any] | None:
        current = now or utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_watches SET status='completed' "
                "WHERE status='active' AND expires_at<=?",
                (current,),
            )
            row = connection.execute(
                "SELECT id, created_at, expires_at, status, label "
                "FROM memory_watches WHERE status='active' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["watch_id"] = int(result.pop("id"))
        return result

    def stop_memory_watch(self) -> dict[str, Any] | None:
        watch = self.active_memory_watch()
        if watch is None:
            return None
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_watches SET status='stopped' WHERE id=?",
                (int(watch["watch_id"]),),
            )
        watch["status"] = "stopped"
        return watch

    def save_memory_watch_sample(
        self,
        *,
        watch_id: int,
        captured_at: str,
        system_memory_percent: float,
        system_available_gb: float | None,
        processes: list[dict[str, Any]],
    ) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_watch_samples("
                "watch_id, captured_at, system_memory_percent, system_available_gb"
                ") VALUES (?, ?, ?, ?)",
                (
                    int(watch_id),
                    captured_at,
                    float(system_memory_percent),
                    system_available_gb,
                ),
            )
            sample_id = int(cursor.lastrowid)
            rows = []
            for process in processes:
                try:
                    rows.append(
                        (
                            sample_id,
                            int(process.get("pid") or 0),
                            str(process.get("name") or "unknown")[:200],
                            float(process.get("ram_mb") or 0.0),
                            float(process.get("cpu_percent") or 0.0),
                            str(process.get("executable") or "")[:1000],
                            str(process.get("command_line") or "")[:4000],
                            int(process.get("parent_pid") or 0),
                            str(process.get("parent_name") or "")[:200],
                            str(process.get("parent_executable") or "")[:1000],
                            (
                                float(process.get("started_at_epoch"))
                                if process.get("started_at_epoch") is not None
                                else None
                            ),
                            str(process.get("username") or "")[:300],
                        )
                    )
                except (TypeError, ValueError):
                    continue
            if rows:
                connection.executemany(
                    "INSERT INTO memory_watch_process_samples("
                    "sample_id, pid, name, ram_mb, cpu_percent, executable, "
                    "command_line, parent_pid, parent_name, parent_executable, "
                    "started_at_epoch, username"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
        return sample_id

    def memory_watch_report(self, *, watch_id: int | None = None) -> dict[str, Any]:
        now = utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_watches SET status='completed' "
                "WHERE status='active' AND expires_at<=?",
                (now,),
            )
            if watch_id is None:
                watch = connection.execute(
                    "SELECT id, created_at, expires_at, status, label "
                    "FROM memory_watches ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            else:
                watch = connection.execute(
                    "SELECT id, created_at, expires_at, status, label "
                    "FROM memory_watches WHERE id=?",
                    (int(watch_id),),
                ).fetchone()
            if watch is None:
                return {"available": False, "reason": "no_memory_watch"}

            wid = int(watch["id"])
            stats = connection.execute(
                "SELECT COUNT(*) AS samples, "
                "AVG(system_memory_percent) AS avg_memory_percent, "
                "MAX(system_memory_percent) AS peak_memory_percent, "
                "MIN(system_available_gb) AS minimum_available_gb, "
                "MIN(captured_at) AS first_sample_at, "
                "MAX(captured_at) AS last_sample_at "
                "FROM memory_watch_samples WHERE watch_id=?",
                (wid,),
            ).fetchone()
            peak = connection.execute(
                "SELECT captured_at, system_memory_percent, system_available_gb "
                "FROM memory_watch_samples WHERE watch_id=? "
                "ORDER BY system_memory_percent DESC, captured_at DESC LIMIT 1",
                (wid,),
            ).fetchone()
            processes = connection.execute(
                "SELECT p.pid, p.name, p.executable, p.command_line, p.parent_pid, "
                "p.parent_name, p.parent_executable, p.started_at_epoch, p.username, "
                "COUNT(*) AS observations, MAX(p.ram_mb) AS peak_ram_mb, "
                "AVG(p.ram_mb) AS average_observed_ram_mb, "
                "MAX(p.cpu_percent) AS peak_cpu_percent "
                "FROM memory_watch_process_samples p "
                "JOIN memory_watch_samples s ON s.id=p.sample_id "
                "WHERE s.watch_id=? "
                "GROUP BY p.pid, p.name, p.executable, p.command_line, "
                "p.parent_pid, p.parent_name, p.parent_executable, "
                "p.started_at_epoch, p.username "
                "ORDER BY peak_ram_mb DESC LIMIT 20",
                (wid,),
            ).fetchall()

        return {
            "available": True,
            "watch": {
                "watch_id": wid,
                "created_at": watch["created_at"],
                "expires_at": watch["expires_at"],
                "status": watch["status"],
                "label": watch["label"],
            },
            "samples": int(stats["samples"] or 0),
            "first_sample_at": stats["first_sample_at"],
            "last_sample_at": stats["last_sample_at"],
            "average_system_memory_percent": (
                round(float(stats["avg_memory_percent"]), 2)
                if stats["avg_memory_percent"] is not None
                else None
            ),
            "peak_system_memory_percent": (
                round(float(stats["peak_memory_percent"]), 2)
                if stats["peak_memory_percent"] is not None
                else None
            ),
            "minimum_available_gb": (
                round(float(stats["minimum_available_gb"]), 2)
                if stats["minimum_available_gb"] is not None
                else None
            ),
            "peak_system_sample": dict(peak) if peak is not None else None,
            "top_memory_consumers": [
                {
                    "pid": int(row["pid"]),
                    "name": row["name"],
                    "executable": row["executable"] or None,
                    "command_line": row["command_line"] or None,
                    "parent_pid": int(row["parent_pid"] or 0) or None,
                    "parent_name": row["parent_name"] or None,
                    "parent_executable": row["parent_executable"] or None,
                    "started_at_epoch": (
                        float(row["started_at_epoch"])
                        if row["started_at_epoch"] is not None
                        else None
                    ),
                    "username": row["username"] or None,
                    "observations": int(row["observations"]),
                    "peak_ram_mb": round(float(row["peak_ram_mb"]), 2),
                    "average_observed_ram_mb": round(
                        float(row["average_observed_ram_mb"]), 2
                    ),
                    "peak_cpu_percent": round(float(row["peak_cpu_percent"]), 2),
                }
                for row in processes
            ],
            "note": (
                "Process rows are the independently ranked top RAM consumers at each "
                "watch sample. Absence from a sample means the process was outside that "
                "bounded top-memory set, not necessarily that it was not running."
            ),
        }

    def memory_watch_process_identity(
        self,
        *,
        pid: int,
        watch_id: int | None = None,
    ) -> dict[str, Any]:
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")
        now = utc_timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_watches SET status='completed' "
                "WHERE status='active' AND expires_at<=?",
                (now,),
            )
            if watch_id is None:
                watch = connection.execute(
                    "SELECT id, created_at, expires_at, status, label "
                    "FROM memory_watches ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            else:
                watch = connection.execute(
                    "SELECT id, created_at, expires_at, status, label "
                    "FROM memory_watches WHERE id=?",
                    (int(watch_id),),
                ).fetchone()
            if watch is None:
                return {"available": False, "reason": "no_memory_watch"}
            wid = int(watch["id"])
            peak = connection.execute(
                "SELECT s.captured_at, p.pid, p.name, p.ram_mb, p.cpu_percent, "
                "p.executable, p.command_line, p.parent_pid, p.parent_name, "
                "p.parent_executable, p.started_at_epoch, p.username "
                "FROM memory_watch_process_samples p "
                "JOIN memory_watch_samples s ON s.id=p.sample_id "
                "WHERE s.watch_id=? AND p.pid=? "
                "ORDER BY p.ram_mb DESC, s.captured_at DESC LIMIT 1",
                (wid, pid),
            ).fetchone()
            observations = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_watch_process_samples p "
                "JOIN memory_watch_samples s ON s.id=p.sample_id "
                "WHERE s.watch_id=? AND p.pid=?",
                (wid, pid),
            ).fetchone()
        if peak is None:
            return {
                "available": False,
                "reason": "pid_not_observed_in_watch",
                "watch_id": wid,
                "pid": pid,
            }
        row = dict(peak)
        row["observations"] = int(observations["count"] or 0)
        row["watch_id"] = wid
        row["available"] = True
        row["peak_ram_mb"] = round(float(row.pop("ram_mb")), 2)
        row["cpu_percent_at_peak"] = round(float(row.pop("cpu_percent")), 2)
        row["executable"] = row.get("executable") or None
        row["command_line"] = row.get("command_line") or None
        row["parent_pid"] = int(row.get("parent_pid") or 0) or None
        row["parent_name"] = row.get("parent_name") or None
        row["parent_executable"] = row.get("parent_executable") or None
        row["username"] = row.get("username") or None
        return row

    def prune_memory_watch(self, *, before: str) -> None:
        with self._lock, self._connect() as connection:
            old_ids = [
                int(row["id"])
                for row in connection.execute(
                    "SELECT id FROM memory_watches "
                    "WHERE created_at<? AND status!='active'",
                    (before,),
                ).fetchall()
            ]
            if not old_ids:
                return
            placeholders = ",".join("?" for _ in old_ids)
            sample_ids = [
                int(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM memory_watch_samples WHERE watch_id IN ({placeholders})",
                    old_ids,
                ).fetchall()
            ]
            if sample_ids:
                sample_placeholders = ",".join("?" for _ in sample_ids)
                connection.execute(
                    f"DELETE FROM memory_watch_process_samples "
                    f"WHERE sample_id IN ({sample_placeholders})",
                    sample_ids,
                )
            connection.execute(
                f"DELETE FROM memory_watch_samples WHERE watch_id IN ({placeholders})",
                old_ids,
            )
            connection.execute(
                f"DELETE FROM memory_watches WHERE id IN ({placeholders})",
                old_ids,
            )

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
        self._last_memory_watch_sample = 0.0
        self._last_systemsense_watch_sample = 0.0
        self._watch_seen_processes: dict[
            int, dict[tuple[int, float], dict[str, Any]]
        ] = {}
        self._artifact_cache: dict[tuple[str, int, int], int | None] = {}
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
                memory_before = (
                    _utc_now() - timedelta(days=self.config.memory_watch_retention_days)
                ).isoformat()
                self.store.prune_memory_watch(before=memory_before)
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
            self._record_systemsense_watch_samples(payload)
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

    def start_watch(
        self,
        *,
        profile: str,
        expires_at: datetime,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Start one bounded owner-requested SystemSense watch profile."""
        if not self.enabled:
            raise RuntimeError("SystemSense is disabled.")
        profile = str(profile).strip().casefold()
        if profile not in {"memory", "cpu", "storage", "network", "gpu", "system"}:
            raise ValueError("profile must be memory, cpu, storage, network, gpu, or system")
        if expires_at.tzinfo is None:
            expires_at = expires_at.astimezone()
        now = datetime.now(UTC)
        expiry = expires_at.astimezone(UTC)
        if expiry <= now + timedelta(minutes=1):
            raise ValueError("SystemSense watch must run for at least one minute.")
        if expiry > now + timedelta(hours=24):
            expiry = now + timedelta(hours=24)
            label = "24 hours"
        watch = self.store.start_watch(
            profile=profile,
            expires_at=expiry.isoformat(),
            label=label or f"{profile} watch",
        )
        self._last_systemsense_watch_sample = 0.0
        self._watch_seen_processes[int(watch["watch_id"])] = {}
        watch["sample_interval_seconds"] = self.config.memory_watch_sample_interval_seconds
        watch["retention_days"] = self.config.memory_watch_retention_days
        return watch

    def stop_watch(
        self,
        *,
        watch_id: int | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        watch = self.store.stop_watch(watch_id=watch_id, profile=profile)
        if watch is None:
            return {"status": "inactive", "available": False}
        self._watch_seen_processes.pop(int(watch["watch_id"]), None)
        return watch

    def watch_report(
        self,
        *,
        watch_id: int | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        report = self.store.watch_report(watch_id=watch_id, profile=profile)
        report["sample_interval_seconds"] = self.config.memory_watch_sample_interval_seconds
        report["retention_days"] = self.config.memory_watch_retention_days
        return report

    def watch_process_identity(
        self,
        *,
        pid: int,
        watch_id: int | None = None,
    ) -> dict[str, Any]:
        return self.store.watch_process_identity(pid=pid, watch_id=watch_id)

    # Compatibility names retained for callers/tests from the first RAM-watch
    # implementation. RAM is now one SystemSense watch profile, not a separate
    # subsystem or product name.
    def start_memory_watch(
        self,
        *,
        expires_at: datetime,
        label: str = "memory watch",
    ) -> dict[str, Any]:
        return self.start_watch(profile="memory", expires_at=expires_at, label=label)

    def stop_memory_watch(self) -> dict[str, Any]:
        return self.stop_watch(profile="memory")

    def memory_watch_process_identity(
        self,
        *,
        pid: int,
        watch_id: int | None = None,
    ) -> dict[str, Any]:
        return self.watch_process_identity(pid=pid, watch_id=watch_id)

    def memory_watch_report(self, *, watch_id: int | None = None) -> dict[str, Any]:
        return self.watch_report(watch_id=watch_id, profile=None if watch_id else "memory")

    def _artifact_id_for_process(self, process: dict[str, Any]) -> int | None:
        path = str(process.get("executable") or "").strip()
        if not path:
            return None
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        key = (str(Path(path).resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        if key in self._artifact_cache:
            return self._artifact_cache[key]
        artifact = inspect_executable_artifact(path)
        artifact_id = self.store.upsert_executable_artifact(artifact)
        self._artifact_cache[key] = artifact_id
        if len(self._artifact_cache) > 512:
            self._artifact_cache = dict(list(self._artifact_cache.items())[-256:])
        return artifact_id

    @staticmethod
    def _rank_connection_pids(connections: list[dict[str, Any]], limit: int) -> list[int]:
        counts: dict[int, int] = {}
        for row in connections:
            pid = int(row.get("pid") or 0)
            if pid > 0:
                counts[pid] = counts.get(pid, 0) + 1
        return [
            pid
            for pid, _count in sorted(
                counts.items(), key=lambda item: (item[1], item[0]), reverse=True
            )[:limit]
        ]

    def _watch_candidate_pids(
        self,
        payload: dict[str, Any],
        profile: str,
        *,
        all_connections: list[dict[str, Any]],
        all_gpu: dict[int, dict[str, Any]],
    ) -> list[int]:
        base = payload.get("base") or {}
        memory_rows = list(base.get("top_memory_processes") or [])
        cpu_rows = list(base.get("top_processes") or [])
        io_rows = list(base.get("top_io_processes") or [])
        limit = max(1, int(self.config.max_processes))

        def pids(rows: list[dict[str, Any]]) -> list[int]:
            return [int(row.get("pid") or 0) for row in rows if int(row.get("pid") or 0) > 0]

        if profile == "memory":
            candidates = pids(memory_rows)
        elif profile == "cpu":
            candidates = pids(cpu_rows)
        elif profile == "storage":
            candidates = pids(io_rows)
        elif profile == "network":
            candidates = self._rank_connection_pids(all_connections, limit)
        elif profile == "gpu":
            candidates = [
                pid
                for pid, _values in sorted(
                    all_gpu.items(),
                    key=lambda item: (
                        float(item[1].get("gpu_percent") or 0.0),
                        float(item[1].get("gpu_committed_mb") or 0.0),
                    ),
                    reverse=True,
                )[:limit]
            ]
        else:
            candidates = (
                pids(memory_rows)
                + pids(cpu_rows)
                + pids(io_rows)
                + self._rank_connection_pids(all_connections, limit)
                + list(all_gpu)
            )

        return list(dict.fromkeys(pid for pid in candidates if pid > 0))[: limit * 2]

    def _record_process_lifecycle(
        self,
        *,
        watch: dict[str, Any],
        processes: list[dict[str, Any]],
        captured_at: str,
    ) -> None:
        watch_id = int(watch["watch_id"])
        seen = self._watch_seen_processes.setdefault(watch_id, {})
        current_keys: set[tuple[int, float]] = set()
        for process in processes:
            pid = int(process.get("pid") or 0)
            started = _finite(process.get("started_at_epoch"))
            if pid <= 0 or started is None:
                continue
            key = (pid, float(started))
            current_keys.add(key)
            if key not in seen:
                seen[key] = {
                    "pid": pid,
                    "started_at_epoch": started,
                    "name": process.get("name"),
                }
                self.store.save_watch_lifecycle(
                    watch_id=watch_id,
                    pid=pid,
                    started_at_epoch=started,
                    event="first_observed",
                    event_at=captured_at,
                    name=str(process.get("name") or "unknown"),
                    parent_pid=int(process.get("parent_pid") or 0) or None,
                    details={"ancestry": process.get("ancestry") or []},
                )

        for key, prior in list(seen.items()):
            if key in current_keys:
                continue
            pid, started = key
            ended = False
            reason = ""
            try:
                live = psutil.Process(pid)
                current_started = float(live.create_time())
                if abs(current_started - started) > 2.0:
                    ended = True
                    reason = "pid_reused"
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                ended = True
                reason = "process_exited"
            except (psutil.AccessDenied, OSError):
                # Lack of permission is not evidence that the process ended.
                continue
            if ended:
                self.store.save_watch_lifecycle(
                    watch_id=watch_id,
                    pid=pid,
                    started_at_epoch=started,
                    event=reason,
                    event_at=captured_at,
                    name=str(prior.get("name") or "unknown"),
                    details={},
                )
                seen.pop(key, None)

    def _record_systemsense_watch_samples(self, payload: dict[str, Any]) -> None:
        captured_at = str(payload.get("captured_at") or utc_timestamp())
        watches = self.store.active_watches(now=captured_at)
        active_ids = {int(item["watch_id"]) for item in watches}
        for watch_id in list(self._watch_seen_processes):
            if watch_id not in active_ids:
                self._watch_seen_processes.pop(watch_id, None)
        if not watches:
            return

        now_mono = time.monotonic()
        if (
            self._last_systemsense_watch_sample
            and now_mono - self._last_systemsense_watch_sample
            < self.config.memory_watch_sample_interval_seconds
        ):
            return

        try:
            all_connections = self.psutil.collect_process_connections(None)
        except (AttributeError, OSError):
            all_connections = []
        try:
            all_gpu = self.performance.collect_process_gpu(None)
        except (AttributeError, OSError):
            all_gpu = {}

        system = {
            "cpu_percent": _path_get(payload, "base.cpu.percent"),
            "memory_percent": _path_get(payload, "base.memory.percent"),
            "system_available_gb": _path_get(payload, "base.memory.available_gb"),
            "storage_read_mb_s": _path_get(payload, "base.storage.io.read_mb_s"),
            "storage_write_mb_s": _path_get(payload, "base.storage.io.write_mb_s"),
            "network_send_mbps": _path_get(payload, "base.network.send_mbps"),
            "network_receive_mbps": _path_get(payload, "base.network.receive_mbps"),
            "gpu_percent": _path_get(payload, "derived.gpu_utilization_percent"),
            "vram_used_mb": _path_get(payload, "derived.vram_used_mb"),
        }

        for watch in watches:
            profile = str(watch.get("profile") or "system")
            candidate_pids = self._watch_candidate_pids(
                payload,
                profile,
                all_connections=all_connections,
                all_gpu=all_gpu,
            )
            try:
                processes = self.psutil.enrich_processes(candidate_pids)
            except AttributeError:
                base = payload.get("base") or {}
                rows = (
                    list(base.get("top_memory_processes") or [])
                    + list(base.get("top_processes") or [])
                    + list(base.get("top_io_processes") or [])
                )
                by_pid = {int(row.get("pid") or 0): dict(row) for row in rows}
                processes = [by_pid[pid] for pid in candidate_pids if pid in by_pid]

            for process in processes:
                gpu = all_gpu.get(int(process.get("pid") or 0), {})
                process.update(gpu)
                process["artifact_id"] = self._artifact_id_for_process(process)

            selected_pids = {int(row.get("pid") or 0) for row in processes}
            connections = [
                row for row in all_connections if int(row.get("pid") or 0) in selected_pids
            ]
            sample_id = self.store.save_watch_sample(
                watch_id=int(watch["watch_id"]),
                captured_at=captured_at,
                system=system,
            )
            self.store.save_watch_process_samples(sample_id=sample_id, processes=processes)
            self.store.save_watch_network_samples(
                sample_id=sample_id,
                connections=connections,
            )
            self._record_process_lifecycle(
                watch=watch,
                processes=processes,
                captured_at=captured_at,
            )

        self._last_systemsense_watch_sample = now_mono

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
