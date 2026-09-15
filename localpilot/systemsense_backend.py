from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import psutil

from localpilot.systemsense_collectors import (
    LibreHardwareMonitorCollector,
    WindowsPerformanceCollector,
    WmiClient,
    utc_timestamp,
)


_MIB = 1024**2
_GIB = 1024**3


def _round_bytes(value: Any, divisor: int = _GIB) -> float | None:
    try:
        return round(float(value) / divisor, 3)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _namedtuple_dict(value: Any) -> dict[str, Any]:
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        return dict(as_dict())
    return {}


class BackendTelemetryCollector:
    """High-detail read-only telemetry for LocalPilot's backend reasoning.

    This is intentionally separate from the compact user-facing SystemSense
    summary. The UI remains glanceable; LocalPilot can request these bounded
    details only when diagnosis actually needs them.
    """

    _SECTIONS = {
        "overview",
        "memory",
        "processes",
        "compute",
        "storage",
        "network",
        "sensors",
    }

    def __init__(
        self,
        *,
        wmi: WmiClient | None = None,
        performance: WindowsPerformanceCollector | None = None,
        sensors: LibreHardwareMonitorCollector | None = None,
    ) -> None:
        self.wmi = wmi or WmiClient()
        self.performance = performance or WindowsPerformanceCollector(self.wmi)
        self.sensors = sensors or LibreHardwareMonitorCollector(self.wmi)

    def _source_provenance(self) -> dict[str, Any]:
        wmi_available = bool(getattr(self.wmi, "available", os.name == "nt"))
        sensor_state = self.sensors.collect()
        return {
            "psutil": {
                "available": True,
                "provenance": "operating-system counters exposed through psutil",
            },
            "windows_wmi": {
                "available": wmi_available,
                "provenance": "native Windows WMI/performance-counter providers",
            },
            "hardware_monitor": {
                "available": bool(sensor_state.get("available")),
                "source": sensor_state.get("source"),
                "provenance": "optional LibreHardwareMonitor/OpenHardwareMonitor WMI sensor bridge",
                "errors": list(sensor_state.get("errors") or []),
            },
        }

    def _native_memory(self) -> dict[str, Any]:
        if not bool(getattr(self.wmi, "available", os.name == "nt")):
            return {"available": False, "source": "windows-wmi"}
        properties = (
            "AvailableBytes",
            "CacheBytes",
            "CommittedBytes",
            "CommitLimit",
            "PageFaultsPersec",
            "PageReadsPersec",
            "PageWritesPersec",
            "PagesInputPersec",
            "PagesOutputPersec",
            "PoolNonpagedBytes",
            "PoolPagedBytes",
        )
        try:
            rows = self.wmi.query(
                r"root\cimv2",
                "Win32_PerfFormattedData_PerfOS_Memory",
                properties,
            )
        except Exception as exc:
            return {
                "available": False,
                "source": "windows-wmi",
                "error": type(exc).__name__,
            }
        if not rows:
            return {"available": False, "source": "windows-wmi"}
        row = rows[0]
        byte_fields = {
            "AvailableBytes",
            "CacheBytes",
            "CommittedBytes",
            "CommitLimit",
            "PoolNonpagedBytes",
            "PoolPagedBytes",
        }
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                continue
            if key in byte_fields:
                normalized[f"{key}_gib"] = _round_bytes(value)
            else:
                normalized[key] = _number(value)
        committed = _number(row.get("CommittedBytes"))
        limit = _number(row.get("CommitLimit"))
        if committed is not None and limit and limit > 0:
            normalized["commit_percent"] = round(100.0 * committed / limit, 2)
        return {
            "available": True,
            "source": "Win32_PerfFormattedData_PerfOS_Memory",
            "values": normalized,
        }

    def memory(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        vm_raw = _namedtuple_dict(vm)
        memory: dict[str, Any] = {
            "percent": round(float(vm.percent), 2),
            "total_gib": _round_bytes(vm.total),
            "available_gib": _round_bytes(vm.available),
            "used_gib": _round_bytes(getattr(vm, "used", None)),
            "free_gib": _round_bytes(getattr(vm, "free", None)),
        }
        for field in ("active", "inactive", "buffers", "cached", "shared", "slab"):
            if field in vm_raw:
                memory[f"{field}_gib"] = _round_bytes(vm_raw[field])
        return {
            "captured_at": utc_timestamp(),
            "source": "psutil+windows-wmi" if os.name == "nt" else "psutil",
            "physical": memory,
            "swap": {
                "percent": round(float(swap.percent), 2),
                "total_gib": _round_bytes(swap.total),
                "used_gib": _round_bytes(swap.used),
                "free_gib": _round_bytes(swap.free),
                "swap_in_mib": _round_bytes(getattr(swap, "sin", 0), _MIB),
                "swap_out_mib": _round_bytes(getattr(swap, "sout", 0), _MIB),
            },
            "windows_native": self._native_memory(),
        }

    @staticmethod
    def _process_row(info: dict[str, Any]) -> dict[str, Any]:
        memory = info.get("memory_info")
        io = info.get("io_counters")
        row = {
            "pid": int(info.get("pid") or 0),
            "ppid": int(info.get("ppid") or 0),
            "name": str(info.get("name") or "unknown")[:200],
            "cpu_percent": round(float(info.get("cpu_percent") or 0.0), 2),
            "threads": int(info.get("num_threads") or 0),
            "rss_mb": round(float(getattr(memory, "rss", 0) or 0) / _MIB, 2),
            "vms_mb": round(float(getattr(memory, "vms", 0) or 0) / _MIB, 2),
        }
        private = getattr(memory, "private", None)
        pagefile = getattr(memory, "pagefile", None)
        if private is not None:
            row["private_mb"] = round(float(private) / _MIB, 2)
        if pagefile is not None:
            row["pagefile_mb"] = round(float(pagefile) / _MIB, 2)
        if io is not None:
            row["io_read_mb"] = round(float(getattr(io, "read_bytes", 0) or 0) / _MIB, 2)
            row["io_write_mb"] = round(float(getattr(io, "write_bytes", 0) or 0) / _MIB, 2)
        return row

    def processes(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        rows: list[dict[str, Any]] = []
        for process in psutil.process_iter(
            ["pid", "ppid", "name", "cpu_percent", "memory_info", "io_counters", "num_threads"]
        ):
            try:
                info = process.info
                if int(info.get("pid") or 0) == 0:
                    continue
                rows.append(self._process_row(info))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
                continue

        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "name": "",
                "process_count": 0,
                "total_cpu_percent": 0.0,
                "total_rss_mb": 0.0,
                "total_private_mb": 0.0,
                "pids": [],
            }
        )
        for row in rows:
            key = str(row["name"]).casefold()
            group = groups[key]
            group["name"] = row["name"]
            group["process_count"] += 1
            group["total_cpu_percent"] += float(row.get("cpu_percent") or 0.0)
            group["total_rss_mb"] += float(row.get("rss_mb") or 0.0)
            group["total_private_mb"] += float(row.get("private_mb") or 0.0)
            if len(group["pids"]) < 32:
                group["pids"].append(row["pid"])
        grouped = list(groups.values())
        for group in grouped:
            for field in ("total_cpu_percent", "total_rss_mb", "total_private_mb"):
                group[field] = round(float(group[field]), 2)

        return {
            "captured_at": utc_timestamp(),
            "sampled_processes": len(rows),
            "top_by_cpu": sorted(
                rows,
                key=lambda row: (row.get("cpu_percent", 0), row.get("rss_mb", 0)),
                reverse=True,
            )[:limit],
            "top_by_ram": sorted(
                rows,
                key=lambda row: (row.get("rss_mb", 0), row.get("cpu_percent", 0)),
                reverse=True,
            )[:limit],
            "groups_by_cpu": sorted(
                grouped,
                key=lambda row: (row["total_cpu_percent"], row["total_rss_mb"]),
                reverse=True,
            )[:limit],
            "groups_by_ram": sorted(
                grouped,
                key=lambda row: (row["total_rss_mb"], row["total_cpu_percent"]),
                reverse=True,
            )[:limit],
        }

    def compute(self) -> dict[str, Any]:
        frequency = psutil.cpu_freq()
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        performance = self.performance.collect()
        return {
            "captured_at": utc_timestamp(),
            "cpu": {
                "percent": round(float(psutil.cpu_percent(interval=None)), 2),
                "per_logical_cpu_percent": [round(float(value), 2) for value in per_cpu],
                "logical_cpus": psutil.cpu_count(logical=True),
                "physical_cpus": psutil.cpu_count(logical=False),
                "frequency_mhz": round(float(frequency.current), 1) if frequency else None,
                "max_frequency_mhz": round(float(frequency.max), 1) if frequency else None,
            },
            "windows_performance": performance,
        }

    def storage(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        volumes = []
        for part in psutil.disk_partitions(all=False):
            if os.name == "nt" and ("cdrom" in part.opts.casefold() or not part.fstype):
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (OSError, PermissionError):
                continue
            volumes.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "filesystem": part.fstype,
                    "total_gib": _round_bytes(usage.total),
                    "used_gib": _round_bytes(usage.used),
                    "free_gib": _round_bytes(usage.free),
                    "percent": round(float(usage.percent), 2),
                }
            )
        io_rows = []
        try:
            counters = psutil.disk_io_counters(perdisk=True) or {}
        except (OSError, AttributeError):
            counters = {}
        for name, counter in counters.items():
            raw = _namedtuple_dict(counter)
            row = {"device": str(name)}
            for field in ("read_count", "write_count", "read_time", "write_time", "busy_time"):
                if field in raw:
                    row[field] = raw[field]
            for field in ("read_bytes", "write_bytes"):
                if field in raw:
                    row[f"{field}_gib"] = _round_bytes(raw[field])
            io_rows.append(row)
        return {
            "captured_at": utc_timestamp(),
            "volumes": volumes[:limit],
            "device_io": io_rows[:limit],
        }

    def network(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        io = psutil.net_io_counters(pernic=True) or {}
        stats = psutil.net_if_stats() or {}
        addresses = psutil.net_if_addrs() or {}
        rows = []
        for name in sorted(set(io) | set(stats) | set(addresses)):
            io_row = _namedtuple_dict(io.get(name)) if name in io else {}
            stat = stats.get(name)
            addr_rows = []
            for address in addresses.get(name, []):
                addr_rows.append(
                    {
                        "family": str(getattr(address, "family", "")),
                        "address": getattr(address, "address", None),
                        "netmask": getattr(address, "netmask", None),
                        "broadcast": getattr(address, "broadcast", None),
                    }
                )
            rows.append(
                {
                    "name": name,
                    "is_up": bool(getattr(stat, "isup", False)) if stat is not None else None,
                    "speed_mbps": getattr(stat, "speed", None) if stat is not None else None,
                    "mtu": getattr(stat, "mtu", None) if stat is not None else None,
                    "bytes_sent_gib": _round_bytes(io_row.get("bytes_sent")),
                    "bytes_recv_gib": _round_bytes(io_row.get("bytes_recv")),
                    "packets_sent": io_row.get("packets_sent"),
                    "packets_recv": io_row.get("packets_recv"),
                    "errors_in": io_row.get("errin"),
                    "errors_out": io_row.get("errout"),
                    "drops_in": io_row.get("dropin"),
                    "drops_out": io_row.get("dropout"),
                    "addresses": addr_rows[:8],
                }
            )
        return {"captured_at": utc_timestamp(), "interfaces": rows[:limit]}

    def sensor_detail(self, *, limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(int(limit), 500))
        payload = self.sensors.collect()
        rows = list(payload.get("sensors") or [])
        return {
            "captured_at": utc_timestamp(),
            "source": payload.get("source"),
            "available": bool(payload.get("available")),
            "errors": list(payload.get("errors") or []),
            "count": len(rows),
            "items": rows[:limit],
        }

    def overview(self, *, limit: int = 12) -> dict[str, Any]:
        limit = max(1, min(int(limit), 50))
        memory = self.memory()
        processes = self.processes(limit=limit)
        compute = self.compute()
        sensor = self.sensor_detail(limit=1)
        return {
            "captured_at": utc_timestamp(),
            "sources": self._source_provenance(),
            "memory": memory,
            "cpu": compute["cpu"],
            "windows_performance": compute.get("windows_performance"),
            "top_process_groups_by_ram": processes["groups_by_ram"][:limit],
            "top_process_groups_by_cpu": processes["groups_by_cpu"][:limit],
            "hardware_sensor_source": {
                "available": sensor["available"],
                "source": sensor["source"],
                "errors": sensor["errors"],
            },
        }

    def collect(self, *, section: str = "overview", limit: int = 50) -> dict[str, Any]:
        section = str(section).strip().casefold()
        if section not in self._SECTIONS:
            raise ValueError(f"section must be one of: {', '.join(sorted(self._SECTIONS))}")
        if section == "overview":
            return self.overview(limit=limit)
        if section == "memory":
            return self.memory()
        if section == "processes":
            return self.processes(limit=limit)
        if section == "compute":
            return self.compute()
        if section == "storage":
            return self.storage(limit=limit)
        if section == "network":
            return self.network(limit=limit)
        return self.sensor_detail(limit=limit)
