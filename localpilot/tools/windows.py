from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time

import psutil

from localpilot.process import hidden_process_creation_flags
from localpilot.process_identity import (
    inspect_executable_artifact,
    sanitize_command_line,
    sanitize_command_text,
)


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
    """Compatibility surface for read-only historical file provenance."""
    return inspect_executable_artifact(path)


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


def inspect_process_launch_context(
    executable: str,
    *,
    process_name: str = "",
    pid: int = 0,
    observed_at: str = "",
) -> str:
    """Best-effort read-only evidence about what can launch an executable.

    Matches Windows services, StartupCommand entries and Scheduled Tasks against
    the recorded executable/name, and returns nearby application crash events.
    Event-log process creation is not assumed to be enabled; absence of an event
    is explicitly non-evidence.
    """
    path = str(executable or "").strip()
    name = str(process_name or "").strip()
    if os.name != "nt":
        return json.dumps(
            {
                "available": False,
                "reason": "launch_context_currently_requires_windows",
                "executable": path or None,
                "process_name": name or None,
            },
            indent=2,
        )

    path_literal = _powershell_literal(path)
    name_literal = _powershell_literal(name)
    observed_literal = _powershell_literal(str(observed_at or ""))
    script = f"""
$ErrorActionPreference = 'SilentlyContinue'
$path = {path_literal}
$name = {name_literal}
$observed = {observed_literal}
$leaf = if ($path) {{ [IO.Path]::GetFileName($path) }} else {{ $name }}
function Match-Text([string]$value) {{
    if (-not $value) {{ return $false }}
    if ($path -and $value.IndexOf($path, [StringComparison]::OrdinalIgnoreCase) -ge 0) {{ return $true }}
    if ($leaf -and $value.IndexOf($leaf, [StringComparison]::OrdinalIgnoreCase) -ge 0) {{ return $true }}
    return $false
}}
$services = @(Get-CimInstance Win32_Service | Where-Object {{ Match-Text $_.PathName }} |
    Select-Object Name,DisplayName,State,StartMode,StartName,PathName)
$startup = @(Get-CimInstance Win32_StartupCommand | Where-Object {{ Match-Text $_.Command }} |
    Select-Object Name,Command,Location,User)
$tasks = @()
Get-ScheduledTask | ForEach-Object {{
    $task = $_
    foreach ($action in @($task.Actions)) {{
        $joined = [string]$action.Execute + ' ' + [string]$action.Arguments
        if (Match-Text $joined) {{
            $tasks += [pscustomobject]@{{
                TaskName = $task.TaskName
                TaskPath = $task.TaskPath
                State = [string]$task.State
                Execute = [string]$action.Execute
                Arguments = [string]$action.Arguments
            }}
            break
        }}
    }}
}}
$since = (Get-Date).AddDays(-7)
if ($observed) {{
    try {{ $since = ([DateTimeOffset]::Parse($observed)).LocalDateTime.AddHours(-6) }} catch {{}}
}}
$events = @(
    Get-WinEvent -FilterHashtable @{{LogName='Application'; StartTime=$since}} -MaxEvents 300 |
    Where-Object {{
        ($_.Id -in 1000,1001,1026) -and
        (($leaf -and $_.Message -like ('*' + $leaf + '*')) -or
         ($name -and $_.Message -like ('*' + $name + '*')))
    }} |
    Select-Object -First 20 TimeCreated,Id,ProviderName,LevelDisplayName,Message
)
[ordered]@{{
    available = $true
    executable = $path
    process_name = $name
    pid = {int(pid)}
    services = $services
    startup_items = $startup
    scheduled_tasks = $tasks
    recent_application_events = $events
    event_note = 'Windows process-creation auditing is not assumed enabled; this is launch-configuration and crash evidence, not a complete creation log.'
}} | ConvertTo-Json -Compress -Depth 7
"""
    raw = _powershell(script, timeout=30)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        payload = {
            "available": False,
            "reason": "launch_context_query_failed",
            "detail": str(raw)[:1000],
        }
    if isinstance(payload, dict):
        for service in payload.get("services") or []:
            if isinstance(service, dict) and service.get("PathName"):
                service["PathName"] = sanitize_command_text(service["PathName"])
        for item in payload.get("startup_items") or []:
            if isinstance(item, dict) and item.get("Command"):
                item["Command"] = sanitize_command_text(item["Command"])
        for task in payload.get("scheduled_tasks") or []:
            if isinstance(task, dict):
                if task.get("Execute"):
                    task["Execute"] = sanitize_command_text(task["Execute"])
                if task.get("Arguments"):
                    task["Arguments"] = sanitize_command_text(task["Arguments"])
    return json.dumps(payload, indent=2)



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
