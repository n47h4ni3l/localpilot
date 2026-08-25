from __future__ import annotations

import json
from types import SimpleNamespace

import psutil

from localpilot.tools import windows


class FakeProcess:
    def __init__(self, pid, name, *, cpu_samples=(0.0, 0.0), rss=0, error=None):
        self.info = {
            "pid": pid,
            "name": name,
            "memory_info": SimpleNamespace(rss=rss),
        }
        self._cpu_samples = iter(cpu_samples)
        self._error = error

    def cpu_percent(self, interval):
        assert interval is None
        if self._error:
            raise self._error
        return next(self._cpu_samples)


def test_powershell_uses_argument_vector_and_returns_output(monkeypatch):
    monkeypatch.setattr(windows.shutil, "which", lambda name: "C:/PowerShell/pwsh.exe" if name == "pwsh" else None)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=" result \n", stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    assert windows._powershell("Get-Date", timeout=7) == "result"
    argv, kwargs = calls[0]
    assert argv == ["C:/PowerShell/pwsh.exe", "-NoProfile", "-NonInteractive", "-Command", "Get-Date"]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "timeout": 7,
        "check": False,
    }


def test_powershell_reports_unavailable_and_command_errors(monkeypatch):
    monkeypatch.setattr(windows.shutil, "which", lambda name: None)
    assert windows._powershell("Get-Date") == "PowerShell is not available."

    monkeypatch.setattr(windows.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(
        windows.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=" denied \n"),
    )
    assert windows._powershell("Get-Date") == "PowerShell error: denied"


def test_system_summary_is_mocked_and_json_serializable(monkeypatch):
    monkeypatch.setattr(windows.platform, "platform", lambda: "Windows-Test")
    monkeypatch.setattr(windows.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(windows.platform, "processor", lambda: "Example CPU")
    monkeypatch.setattr(windows.psutil, "cpu_count", lambda logical: 16 if logical else 8)
    monkeypatch.setattr(
        windows.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=12 * 1024**3),
    )
    monkeypatch.setattr(windows.psutil, "boot_time", lambda: 1000.0)
    monkeypatch.setattr(windows.time, "time", lambda: 1000.0 + 2.5 * 3600)

    data = json.loads(windows.get_system_summary())

    assert data == {
        "os": "Windows-Test",
        "machine": "AMD64",
        "processor": "Example CPU",
        "logical_cpus": 16,
        "physical_cpus": 8,
        "ram_total_gb": 32.0,
        "ram_available_gb": 12.0,
        "uptime_hours": 2.5,
    }


def test_storage_summary_filters_windows_optical_and_inaccessible_volumes(monkeypatch):
    parts = [
        SimpleNamespace(device="C:", mountpoint="C:/", fstype="NTFS", opts="rw"),
        SimpleNamespace(device="D:", mountpoint="D:/", fstype="UDF", opts="cdrom"),
        SimpleNamespace(device="E:", mountpoint="E:/", fstype="", opts="rw"),
        SimpleNamespace(device="F:", mountpoint="F:/", fstype="NTFS", opts="rw"),
    ]
    monkeypatch.setattr(windows.os, "name", "nt")
    monkeypatch.setattr(windows.psutil, "disk_partitions", lambda all: parts)

    def fake_usage(mountpoint):
        if mountpoint == "F:/":
            raise PermissionError
        return SimpleNamespace(total=100 * 1024**3, free=25 * 1024**3, percent=75.0)

    monkeypatch.setattr(windows.psutil, "disk_usage", fake_usage)

    assert json.loads(windows.get_storage_summary()) == [{
        "device": "C:",
        "mountpoint": "C:/",
        "filesystem": "NTFS",
        "total_gb": 100.0,
        "free_gb": 25.0,
        "free_percent": 25.0,
    }]


def test_top_processes_uses_task_manager_cpu_and_excludes_idle(monkeypatch):
    idle = FakeProcess(0, "System Idle Process", cpu_samples=(0.0, 800.0), rss=1)
    busy = FakeProcess(10, "busy.exe", cpu_samples=(0.0, 320.0), rss=512 * 1024**2)
    helper = FakeProcess(11, "helper.exe", cpu_samples=(0.0, 80.0), rss=128 * 1024**2)
    denied = FakeProcess(12, "denied.exe", error=psutil.AccessDenied(12))
    processes = [idle, busy, helper, denied]
    calls = []
    monkeypatch.setattr(windows.psutil, "cpu_count", lambda logical: 8)
    monkeypatch.setattr(windows.psutil, "process_iter", lambda attrs: processes)
    monkeypatch.setattr(windows.time, "sleep", calls.append)

    rows = json.loads(windows.get_top_processes(limit=30))

    assert calls == [0.2]
    assert rows == [
        {"pid": 10, "name": "busy.exe", "cpu_percent": 40.0, "ram_mb": 512.0},
        {"pid": 11, "name": "helper.exe", "cpu_percent": 10.0, "ram_mb": 128.0},
    ]
    assert all(row["name"] != "System Idle Process" for row in rows)


def test_top_process_limit_is_clamped_and_missing_cpu_count_is_safe(monkeypatch):
    process = FakeProcess(7, "worker.exe", cpu_samples=(0.0, 25.0), rss=1024**2)
    monkeypatch.setattr(windows.psutil, "cpu_count", lambda logical: None)
    monkeypatch.setattr(windows.psutil, "process_iter", lambda attrs: [process])
    monkeypatch.setattr(windows.time, "sleep", lambda seconds: None)

    assert json.loads(windows.get_top_processes(limit=0)) == [
        {"pid": 7, "name": "worker.exe", "cpu_percent": 25.0, "ram_mb": 1.0}
    ]


def test_windows_powershell_tools_delegate_to_read_only_queries(monkeypatch):
    scripts = []
    monkeypatch.setattr(windows, "_powershell", lambda script: scripts.append(script) or "ok")

    assert windows.get_startup_items() == "ok"
    assert windows.get_active_power_plan() == "ok"
    assert windows.get_defender_summary() == "ok"
    assert windows.get_device_problem_summary() == "ok"

    assert "Win32_StartupCommand" in scripts[0]
    assert scripts[1] == "powercfg /GETACTIVESCHEME"
    assert "Get-MpComputerStatus" in scripts[2]
    assert "Get-PnpDevice" in scripts[3]
