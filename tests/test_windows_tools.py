from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import psutil

from localpilot.process import hidden_process_creation_flags
from localpilot.process_identity import sanitize_command_line
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
        "creationflags": hidden_process_creation_flags(),
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



def test_inspect_executable_metadata_delegates_to_observation_time_fingerprint(monkeypatch):
    calls = []

    monkeypatch.setattr(
        windows,
        "inspect_executable_artifact",
        lambda path: (
            calls.append(path)
            or {
                "available": True,
                "path": path,
                "company_name": "Example Corp",
                "product_name": "Example Worker",
                "file_version": "1.2.3",
                "signature_status": "Valid",
                "signer_subject": "CN=Example Corp",
                "sha256": "ABC123",
            }
        ),
    )

    result = windows.inspect_executable_metadata(r"C:\Apps\worker.exe")

    assert calls == [r"C:\Apps\worker.exe"]
    assert result["company_name"] == "Example Corp"
    assert result["signature_status"] == "Valid"
    assert result["sha256"] == "ABC123"


def test_inspect_process_identity_returns_runtime_parent_services_and_file_provenance(
    monkeypatch,
):
    class IdentityProcess:
        pid = 42

        def oneshot(self):
            return nullcontext()

        def exe(self):
            return r"C:\Apps\worker.exe"

        def is_running(self):
            return True

        def name(self):
            return "worker.exe"

        def cmdline(self):
            return [r"C:\Apps\worker.exe", "--serve"]

        def ppid(self):
            return 7

        def create_time(self):
            return 1234.5

        def username(self):
            return r"PC\owner"

        def parent(self):
            return ParentProcess()

    class ParentProcess:
        pid = 7

        def oneshot(self):
            return nullcontext()

        def name(self):
            return "launcher.exe"

        def exe(self):
            return r"C:\Apps\launcher.exe"

    monkeypatch.setattr(windows.psutil, "Process", lambda pid: IdentityProcess())
    monkeypatch.setattr(
        windows,
        "inspect_executable_metadata",
        lambda path: {
            "available": True,
            "path": path,
            "company_name": "Example Corp",
            "signature_status": "Valid",
        },
    )
    monkeypatch.setattr(windows.os, "name", "nt")
    monkeypatch.setattr(
        windows,
        "_powershell",
        lambda script, timeout=20: json.dumps(
            {
                "Name": "ExampleService",
                "DisplayName": "Example Service",
                "State": "Running",
                "StartMode": "Auto",
                "PathName": r"C:\Apps\worker.exe --service",
            }
        ),
    )

    result = json.loads(windows.inspect_process_identity(42))

    assert result["available"] is True
    assert result["running"] is True
    assert result["name"] == "worker.exe"
    assert result["executable"] == r"C:\Apps\worker.exe"
    assert result["command_line"].endswith("--serve")
    assert result["parent"]["pid"] == 7
    assert result["parent"]["name"] == "launcher.exe"
    assert result["executable_metadata"]["company_name"] == "Example Corp"
    assert result["services"][0]["Name"] == "ExampleService"



def test_process_command_line_redacts_common_secret_values():
    rendered = sanitize_command_line(
        [
            "python.exe",
            "worker.py",
            "--token",
            "super-secret-token",
            "--api-key=abcdef",
            "https://user:password@example.test/path",
            "--mode",
            "safe",
        ]
    )

    assert "super-secret-token" not in rendered
    assert "abcdef" not in rendered
    assert "password@example" not in rendered
    assert "--token <redacted>" in rendered
    assert "--api-key=<redacted>" in rendered
    assert "https://user:<redacted>@example.test/path" in rendered
    assert "--mode safe" in rendered



def test_launch_context_redacts_task_and_startup_secrets(monkeypatch):
    monkeypatch.setattr(windows.os, "name", "nt")
    monkeypatch.setattr(
        windows,
        "_powershell",
        lambda script, timeout=30: json.dumps(
            {
                "available": True,
                "services": [
                    {
                        "Name": "Worker",
                        "PathName": r"C:\Apps\worker.exe --token service-secret",
                    }
                ],
                "startup_items": [
                    {
                        "Name": "Worker",
                        "Command": r"C:\Apps\worker.exe --api-key=startup-secret",
                    }
                ],
                "scheduled_tasks": [
                    {
                        "TaskName": "Worker",
                        "Execute": r"C:\Apps\worker.exe",
                        "Arguments": "--password task-secret --mode safe",
                    }
                ],
                "recent_application_events": [],
            }
        ),
    )

    payload = json.loads(
        windows.inspect_process_launch_context(
            r"C:\Apps\worker.exe",
            process_name="worker.exe",
            pid=42,
        )
    )

    rendered = json.dumps(payload)
    assert "service-secret" not in rendered
    assert "startup-secret" not in rendered
    assert "task-secret" not in rendered
    assert "<redacted>" in rendered
