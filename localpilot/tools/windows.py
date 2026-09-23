from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time

import psutil

from localpilot.process import hidden_process_creation_flags
from localpilot.process_identity import sanitize_command_line


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


def _powershell_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def inspect_executable_metadata(path: str) -> dict:
    """Inspect one executable path without executing it."""
    executable = str(path or "").strip()
    if not executable:
        return {"available": False, "reason": "missing_executable_path"}
    if os.name != "nt":
        return {
            "available": False,
            "reason": "executable_metadata_currently_requires_windows",
            "path": executable,
        }
    quoted = _powershell_literal(executable)
    script = f"""
$ErrorActionPreference = 'Stop'
$path = {quoted}
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {{
    [ordered]@{{ available = $false; reason = 'file_not_found'; path = $path }} |
        ConvertTo-Json -Compress
    exit 0
}}
$item = Get-Item -LiteralPath $path -ErrorAction Stop
$version = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)
$signature = Get-AuthenticodeSignature -LiteralPath $path -ErrorAction SilentlyContinue
$hash = Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction SilentlyContinue
[ordered]@{{
    available = $true
    path = $item.FullName
    size_bytes = [Int64]$item.Length
    modified_at = $item.LastWriteTimeUtc.ToString('o')
    company_name = $version.CompanyName
    product_name = $version.ProductName
    file_description = $version.FileDescription
    file_version = $version.FileVersion
    product_version = $version.ProductVersion
    original_filename = $version.OriginalFilename
    signature_status = if ($signature) {{ [string]$signature.Status }} else {{ $null }}
    signer_subject = if ($signature -and $signature.SignerCertificate) {{
        [string]$signature.SignerCertificate.Subject
    }} else {{ $null }}
    signer_issuer = if ($signature -and $signature.SignerCertificate) {{
        [string]$signature.SignerCertificate.Issuer
    }} else {{ $null }}
    sha256 = if ($hash) {{ [string]$hash.Hash }} else {{ $null }}
}} | ConvertTo-Json -Compress -Depth 4
"""
    raw = _powershell(script, timeout=20)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {
            "available": False,
            "reason": "metadata_query_failed",
            "path": executable,
            "detail": str(raw)[:500],
        }
    return payload if isinstance(payload, dict) else {
        "available": False,
        "reason": "metadata_query_invalid",
        "path": executable,
    }


def inspect_process_identity(pid: int) -> str:
    """Inspect one running process's local identity and executable provenance."""
    pid = int(pid)
    if pid <= 0:
        raise ValueError("pid must be a positive integer")
    result = {
        "available": False,
        "pid": pid,
        "running": False,
    }
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            executable = process.exe()
            result.update(
                available=True,
                running=process.is_running(),
                name=process.name(),
                executable=executable or None,
                command_line=sanitize_command_line(process.cmdline()) or None,
                parent_pid=int(process.ppid() or 0) or None,
                started_at_epoch=float(process.create_time()),
                username=process.username() or None,
            )
        try:
            parent = process.parent()
            if parent is not None:
                with parent.oneshot():
                    result["parent"] = {
                        "pid": int(parent.pid),
                        "name": parent.name(),
                        "executable": parent.exe() or None,
                    }
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
            result["parent"] = None
    except psutil.NoSuchProcess:
        result["reason"] = "process_not_running"
        return json.dumps(result, indent=2)
    except (psutil.AccessDenied, psutil.ZombieProcess, OSError) as exc:
        result["reason"] = type(exc).__name__
        return json.dumps(result, indent=2)

    executable = str(result.get("executable") or "")
    if executable:
        result["executable_metadata"] = inspect_executable_metadata(executable)

    if os.name == "nt":
        service_script = (
            "Get-CimInstance Win32_Service -ErrorAction SilentlyContinue | "
            f"Where-Object {{ $_.ProcessId -eq {pid} }} | "
            "Select-Object Name,DisplayName,State,StartMode,PathName | "
            "ConvertTo-Json -Compress -Depth 3"
        )
        raw_services = _powershell(service_script, timeout=15)
        try:
            services = json.loads(raw_services) if raw_services else []
        except json.JSONDecodeError:
            services = []
        if isinstance(services, dict):
            services = [services]
        result["services"] = services if isinstance(services, list) else []

    return json.dumps(result, indent=2)


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
