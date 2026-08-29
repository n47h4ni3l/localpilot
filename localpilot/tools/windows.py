from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time

import psutil

from localpilot.process import hidden_process_creation_flags


def _powershell(script: str, timeout: int = 20) -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if not executable:
        return "PowerShell is not available."
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=hidden_process_creation_flags(),
    )
    if completed.returncode != 0:
        return f"PowerShell error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def get_system_summary() -> str:
    """Return a read-only Windows, CPU, RAM and uptime summary."""
    vm = psutil.virtual_memory()
    data = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }
    return json.dumps(data, indent=2)


def get_storage_summary() -> str:
    """Return usage for mounted local disks without modifying anything."""
    rows = []
    for part in psutil.disk_partitions(all=False):
        if os.name == "nt" and ("cdrom" in part.opts.lower() or not part.fstype):
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        rows.append({
            "device": part.device,
            "mountpoint": part.mountpoint,
            "filesystem": part.fstype,
            "total_gb": round(usage.total / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
            "free_percent": round(100 - usage.percent, 1),
        })
    return json.dumps(rows, indent=2)


def get_top_processes(limit: int = 12) -> str:
    """Return top workloads using Windows Task Manager-style CPU percentages."""
    limit = max(1, min(int(limit), 30))
    logical_cpus = psutil.cpu_count(logical=True) or 1
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        if p.info["pid"] == 0 or str(p.info["name"] or "").casefold() == "system idle process":
            continue
        try:
            p.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    time.sleep(0.2)
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        if p.info["pid"] == 0 or str(p.info["name"] or "").casefold() == "system idle process":
            continue
        try:
            mem = p.info["memory_info"].rss if p.info["memory_info"] else 0
            procs.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                # Process.cpu_percent uses top-style per-core percentages and can
                # exceed 100. Divide by logical CPUs to match Windows Task Manager.
                "cpu_percent": round(p.cpu_percent(None) / logical_cpus, 1),
                "ram_mb": round(mem / 1024**2, 1),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    procs.sort(key=lambda row: (row["cpu_percent"], row["ram_mb"]), reverse=True)
    return json.dumps(procs[:limit], indent=2)


def get_startup_items() -> str:
    """Return Windows startup entries."""
    return _powershell(r'''Get-CimInstance Win32_StartupCommand -ErrorAction SilentlyContinue |
Select-Object Name, Command, Location, User | ConvertTo-Json -Depth 3''')


def get_active_power_plan() -> str:
    """Return the current Windows power plan."""
    return _powershell("powercfg /GETACTIVESCHEME")


def get_defender_summary() -> str:
    """Return basic Microsoft Defender protection state."""
    return _powershell(r'''Get-MpComputerStatus -ErrorAction SilentlyContinue |
Select-Object AntivirusEnabled, RealTimeProtectionEnabled, BehaviorMonitorEnabled, IoavProtectionEnabled, NISEnabled |
ConvertTo-Json''')


def get_device_problem_summary() -> str:
    """Return currently connected PnP devices with a non-OK status."""
    return _powershell(r'''Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
Where-Object Status -ne 'OK' | Select-Object Class, FriendlyName, Status, Problem |
ConvertTo-Json -Depth 3''')
