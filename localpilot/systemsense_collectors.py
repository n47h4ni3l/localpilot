from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
import time
from datetime import UTC, datetime
from io import StringIO
from typing import Any, Iterable

import psutil

from localpilot.process import hidden_process_creation_flags


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, Iterable):
        try:
            return [_json_value(item) for item in value]
        except TypeError:
            pass
    return str(value)


class WmiClient:
    """Small read-only COM/WMI adapter with no PowerShell dependency."""

    @property
    def available(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import pythoncom  # noqa: F401
            import win32com.client  # noqa: F401
        except ImportError:
            return False
        return True

    def query(
        self,
        namespace: str,
        class_name: str,
        properties: tuple[str, ...],
        where: str = "",
    ) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            import pythoncom
            import win32com.client
        except ImportError:
            return []

        pythoncom.CoInitialize()
        locator = None
        service = None
        result_set = None
        item = None
        try:
            locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
            service = locator.ConnectServer(".", namespace)
            service.Security_.ImpersonationLevel = 3
            query = f"SELECT {','.join(properties)} FROM {class_name}"
            if where:
                query += f" WHERE {where}"
            rows: list[dict[str, Any]] = []
            result_set = service.ExecQuery(query, "WQL", 0x30)
            for item in result_set:
                row: dict[str, Any] = {}
                for prop in properties:
                    try:
                        row[prop] = _json_value(getattr(item, prop))
                    except Exception:
                        row[prop] = None
                rows.append(row)
            return rows
        finally:
            item = None
            result_set = None
            service = None
            locator = None
            pythoncom.CoUninitialize()


class PsutilTelemetryCollector:
    """Cheap cross-platform counters; Windows-native collectors enrich these."""

    def __init__(self, *, max_processes: int = 12) -> None:
        self.max_processes = max(1, min(int(max_processes), 50))
        self._last_at: float | None = None
        self._last_disk: Any = None
        self._last_network: Any = None

    @staticmethod
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

    def collect(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(0.001, now - self._last_at) if self._last_at is not None else None
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        frequency = psutil.cpu_freq()
        disk = psutil.disk_io_counters()
        network = psutil.net_io_counters()

        io: dict[str, float | None] = {
            "read_mb_s": None,
            "write_mb_s": None,
            "read_iops": None,
            "write_iops": None,
        }
        net: dict[str, float | None] = {"send_mbps": None, "receive_mbps": None}
        if elapsed is not None and disk is not None and self._last_disk is not None:
            io = {
                "read_mb_s": round(max(0, disk.read_bytes - self._last_disk.read_bytes) / elapsed / 1024**2, 3),
                "write_mb_s": round(max(0, disk.write_bytes - self._last_disk.write_bytes) / elapsed / 1024**2, 3),
                "read_iops": round(max(0, disk.read_count - self._last_disk.read_count) / elapsed, 2),
                "write_iops": round(max(0, disk.write_count - self._last_disk.write_count) / elapsed, 2),
            }
        if elapsed is not None and network is not None and self._last_network is not None:
            net = {
                "send_mbps": round(max(0, network.bytes_sent - self._last_network.bytes_sent) * 8 / elapsed / 1_000_000, 3),
                "receive_mbps": round(max(0, network.bytes_recv - self._last_network.bytes_recv) * 8 / elapsed / 1_000_000, 3),
            }
        self._last_at = now
        self._last_disk = disk
        self._last_network = network

        volumes: list[dict[str, Any]] = []
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
                    "total_gb": round(usage.total / 1024**3, 2),
                    "free_gb": round(usage.free / 1024**3, 2),
                    "free_percent": round(100.0 - usage.percent, 2),
                }
            )

        processes: list[dict[str, Any]] = []
        for process in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "io_counters"]
        ):
            try:
                info = process.info
                if int(info.get("pid") or 0) == 0:
                    continue
                memory = info.get("memory_info")
                proc_io = info.get("io_counters")
                processes.append(
                    {
                        "pid": int(info.get("pid") or 0),
                        "name": str(info.get("name") or "unknown")[:200],
                        "cpu_percent": round(float(info.get("cpu_percent") or 0.0), 2),
                        "ram_mb": round(float(memory.rss if memory else 0) / 1024**2, 2),
                        "io_total_mb": round(
                            float((proc_io.read_bytes + proc_io.write_bytes) if proc_io else 0)
                            / 1024**2,
                            2,
                        ),
                    }
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
                continue
        processes.sort(key=lambda row: (row["cpu_percent"], row["ram_mb"]), reverse=True)

        battery = None
        try:
            value = psutil.sensors_battery()
            if value is not None:
                battery = {
                    "percent": round(float(value.percent), 1),
                    "plugged_in": bool(value.power_plugged),
                    "seconds_left": int(value.secsleft),
                }
        except (AttributeError, OSError):
            pass

        cpu_percent = round(float(psutil.cpu_percent(interval=None)), 2)
        return {
            "captured_at": utc_timestamp(),
            "source": "psutil",
            "cpu": {
                "percent": cpu_percent,
                "pressure": self._pressure(cpu_percent),
                "logical_cpus": psutil.cpu_count(logical=True),
                "physical_cpus": psutil.cpu_count(logical=False),
                "frequency_mhz": round(float(frequency.current), 1) if frequency else None,
                "max_frequency_mhz": round(float(frequency.max), 1) if frequency else None,
            },
            "memory": {
                "percent": round(float(vm.percent), 2),
                "pressure": self._pressure(float(vm.percent)),
                "total_gb": round(vm.total / 1024**3, 2),
                "available_gb": round(vm.available / 1024**3, 2),
                "swap_percent": round(float(swap.percent), 2),
            },
            "storage": {"io": io, "volumes": volumes},
            "network": net,
            "battery": battery,
            "top_processes": processes[: self.max_processes],
        }


class WindowsPerformanceCollector:
    """GPU, processor-limit and thermal observations from native WMI providers."""

    def __init__(self, wmi: WmiClient | None = None) -> None:
        self.wmi = wmi or WmiClient()

    def collect(self) -> dict[str, Any]:
        available = bool(getattr(self.wmi, "available", os.name == "nt"))
        if not available:
            return {"source": "windows-wmi", "available": False}
        output: dict[str, Any] = {
            "source": "windows-wmi",
            "available": True,
            "gpu": {},
            "processor": {},
            "power_plan": {},
            "thermal_zones": [],
            "errors": [],
        }
        try:
            engines = self.wmi.query(
                r"root\cimv2",
                "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
                ("Name", "UtilizationPercentage"),
            )
            values = [float(row["UtilizationPercentage"]) for row in engines if row.get("UtilizationPercentage") is not None]
            output["gpu"]["engine_utilization_percent"] = round(max(values), 2) if values else None
        except Exception as exc:
            output["errors"].append(f"gpu-engine:{type(exc).__name__}")
        try:
            memory = self.wmi.query(
                r"root\cimv2",
                "Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory",
                ("Name", "DedicatedUsage", "SharedUsage", "TotalCommitted"),
            )
            output["gpu"]["adapter_memory"] = memory
        except Exception as exc:
            output["errors"].append(f"gpu-memory:{type(exc).__name__}")
        try:
            processor = self.wmi.query(
                r"root\cimv2",
                "Win32_PerfFormattedData_Counters_ProcessorInformation",
                (
                    "Name",
                    "PercentProcessorPerformance",
                    "PercentPerformanceLimit",
                    "ProcessorFrequency",
                ),
                "Name='_Total'",
            )
            output["processor"] = processor[0] if processor else {}
        except Exception as exc:
            output["errors"].append(f"processor:{type(exc).__name__}")
        try:
            zones = self.wmi.query(
                r"root\wmi",
                "MSAcpi_ThermalZoneTemperature",
                ("InstanceName", "CurrentTemperature", "CriticalTripPoint"),
            )
            for row in zones:
                current = row.get("CurrentTemperature")
                critical = row.get("CriticalTripPoint")
                row["current_c"] = round(float(current) / 10.0 - 273.15, 2) if current is not None else None
                row["critical_c"] = round(float(critical) / 10.0 - 273.15, 2) if critical is not None else None
            output["thermal_zones"] = zones
        except Exception as exc:
            output["errors"].append(f"thermal:{type(exc).__name__}")
        try:
            plans = self.wmi.query(
                r"root\cimv2\power",
                "Win32_PowerPlan",
                ("ElementName", "InstanceID", "IsActive"),
                "IsActive=True",
            )
            output["power_plan"] = plans[0] if plans else {}
        except Exception as exc:
            output["errors"].append(f"power-plan:{type(exc).__name__}")
        return output


class LibreHardwareMonitorCollector:
    """Reads Libre/OpenHardwareMonitor's read-only WMI sensor namespace when present."""

    def __init__(self, wmi: WmiClient | None = None) -> None:
        self.wmi = wmi or WmiClient()

    def collect(self) -> dict[str, Any]:
        properties = ("Identifier", "Name", "SensorType", "Value", "Min", "Max", "Parent")
        errors: list[str] = []
        for namespace, source in (
            (r"root\LibreHardwareMonitor", "LibreHardwareMonitor"),
            (r"root\OpenHardwareMonitor", "OpenHardwareMonitor"),
        ):
            try:
                rows = self.wmi.query(namespace, "Sensor", properties)
            except Exception as exc:
                errors.append(f"{source}:{type(exc).__name__}")
                continue
            if rows:
                return {"source": source, "available": True, "sensors": rows, "errors": errors}
        return {"source": "LibreHardwareMonitor", "available": False, "sensors": [], "errors": errors}


class DriverStoreCollector:
    """Read third-party driver-store packages through Windows' native PnPUtil."""

    def collect(self) -> dict[str, Any]:
        if os.name != "nt":
            return {"source": "pnputil", "available": False, "packages": []}
        try:
            completed = subprocess.run(
                ["pnputil.exe", "/enum-drivers", "/files", "/format", "csv"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "source": "pnputil",
                "available": False,
                "packages": [],
                "error": type(exc).__name__,
            }
        if completed.returncode != 0:
            return {
                "source": "pnputil",
                "available": False,
                "packages": [],
                "error": (completed.stderr or "pnputil failed").strip()[:500],
            }
        rows = list(csv.DictReader(StringIO(completed.stdout)))
        packages_by_name: dict[str, dict[str, Any]] = {}
        for row in rows:
            normalized = {
                "".join(ch for ch in str(key).casefold() if ch.isalnum()): value
                for key, value in row.items()
                if key is not None
            }
            published_name = (
                normalized.get("publishedname")
                or normalized.get("infname")
                or normalized.get("drivername")
            )
            if not published_name:
                continue
            raw_version = str(
                normalized.get("driverversion") or normalized.get("version") or ""
            ).strip()
            date = normalized.get("driverdate") or normalized.get("date")
            version = raw_version
            parts = raw_version.split()
            if len(parts) >= 2 and re.fullmatch(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}", parts[0]):
                date = date or parts[0]
                version = parts[-1]
            key = str(published_name).casefold()
            package = packages_by_name.setdefault(
                key,
                {
                    "published_name": published_name,
                    "original_name": None,
                    "provider": None,
                    "class_name": None,
                    "class_guid": None,
                    "driver_version": None,
                    "driver_date": None,
                    "signer": None,
                    "files": [],
                },
            )
            candidates = {
                "original_name": normalized.get("originalname"),
                "provider": normalized.get("providername") or normalized.get("provider"),
                "class_name": normalized.get("classname") or normalized.get("class"),
                "class_guid": normalized.get("classguid"),
                "driver_version": version or None,
                "driver_date": date,
                "signer": normalized.get("signername") or normalized.get("signer"),
            }
            for field, value in candidates.items():
                if value not in (None, "") and package.get(field) in (None, ""):
                    package[field] = value
            file_name = normalized.get("file") or normalized.get("driverfiles") or normalized.get("files")
            if file_name and file_name not in package["files"]:
                package["files"].append(file_name)
        packages = list(packages_by_name.values())
        return {
            "source": "pnputil",
            "available": bool(packages),
            "packages": packages,
            "unparsed_output": not bool(packages),
        }


class WindowsInventoryCollector:
    """Slow-changing hardware, firmware, PnP and driver associations."""

    _QUERIES: dict[str, tuple[str, tuple[str, ...]]] = {
        "computer_system": (
            "Win32_ComputerSystem",
            ("Manufacturer", "Model", "SystemType", "TotalPhysicalMemory", "PCSystemType"),
        ),
        "computer_product": (
            "Win32_ComputerSystemProduct",
            ("Vendor", "Name", "Version", "IdentifyingNumber", "UUID"),
        ),
        "motherboard": (
            "Win32_BaseBoard",
            ("Manufacturer", "Product", "Version", "SerialNumber", "Tag"),
        ),
        "bios": (
            "Win32_BIOS",
            ("Manufacturer", "SMBIOSBIOSVersion", "ReleaseDate", "SerialNumber", "SMBIOSMajorVersion", "SMBIOSMinorVersion"),
        ),
        "cpu": (
            "Win32_Processor",
            ("DeviceID", "Name", "Manufacturer", "ProcessorId", "NumberOfCores", "NumberOfLogicalProcessors", "MaxClockSpeed", "SocketDesignation"),
        ),
        "gpu": (
            "Win32_VideoController",
            ("PNPDeviceID", "Name", "AdapterCompatibility", "AdapterRAM", "DriverVersion", "DriverDate", "VideoProcessor", "Status"),
        ),
        "memory_modules": (
            "Win32_PhysicalMemory",
            ("DeviceLocator", "BankLabel", "Manufacturer", "PartNumber", "SerialNumber", "Capacity", "Speed", "ConfiguredClockSpeed", "SMBIOSMemoryType"),
        ),
        "storage_devices": (
            "Win32_DiskDrive",
            ("DeviceID", "Model", "Manufacturer", "SerialNumber", "FirmwareRevision", "InterfaceType", "PNPDeviceID", "Size", "Status"),
        ),
        "network_adapters": (
            "Win32_NetworkAdapter",
            ("GUID", "Name", "Manufacturer", "MACAddress", "PNPDeviceID", "NetConnectionID", "NetEnabled", "PhysicalAdapter", "ServiceName", "AdapterType"),
        ),
        "devices": (
            "Win32_PnPEntity",
            ("DeviceID", "PNPClass", "Name", "Manufacturer", "Service", "Status", "ConfigManagerErrorCode", "Present", "HardwareID", "CompatibleID", "ClassGuid"),
        ),
        "signed_drivers": (
            "Win32_PnPSignedDriver",
            ("DeviceID", "DeviceName", "Manufacturer", "DriverProviderName", "DriverVersion", "DriverDate", "InfName", "IsSigned", "Signer", "DeviceClass", "DriverName"),
        ),
        "system_drivers": (
            "Win32_SystemDriver",
            ("Name", "DisplayName", "State", "Status", "Started", "StartMode", "PathName", "ServiceType", "ErrorControl"),
        ),
    }

    def __init__(
        self,
        wmi: WmiClient | None = None,
        driver_store: DriverStoreCollector | None = None,
    ) -> None:
        self.wmi = wmi or WmiClient()
        self.driver_store = driver_store or DriverStoreCollector()

    def collect(self) -> dict[str, Any]:
        wmi_available = bool(getattr(self.wmi, "available", os.name == "nt"))
        inventory: dict[str, Any] = {
            "captured_at": utc_timestamp(),
            "platform": platform.platform(),
            "source": "windows-wmi",
            "available": wmi_available,
            "errors": [],
        }
        if not wmi_available:
            inventory["errors"].append("windows-wmi-unavailable")
            inventory["driver_store"] = self.driver_store.collect()
            return inventory
        for section, (class_name, properties) in self._QUERIES.items():
            try:
                inventory[section] = self.wmi.query(
                    r"root\cimv2", class_name, properties
                )
            except Exception as exc:
                inventory[section] = []
                inventory["errors"].append(f"{section}:{type(exc).__name__}")
        inventory["driver_store"] = self.driver_store.collect()
        return inventory
