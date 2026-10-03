from pathlib import Path
import subprocess
import sys

from localpilot.config import Config
from localpilot.doctor import doctor


def test_module_cli_executes_help():
    result = subprocess.run(
        [sys.executable, "-m", "localpilot.cli", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "desktop" in result.stdout
    assert "doctor" in result.stdout


def test_doctor_reports_missing_windows_sensor_helper(monkeypatch, tmp_path):
    import localpilot.doctor as module
    import localpilot.systemsense_hardware as hardware
    if module.os.name != "nt":
        return
    monkeypatch.setattr(hardware, "bundled_helper_path", lambda: tmp_path / "missing.exe")
    monkeypatch.setattr(module, "_ollama_models", lambda: ({"gpt-oss:20b"}, "test"))
    checks = doctor(Config(), Path(__file__).resolve().parents[1])
    check = next(row for row in checks if row[0] == "SystemSense hardware provider")
    assert check[1] is False
    assert "build-systemsense-hardware.ps1" in check[2]


def test_bundled_provider_is_windows_only(monkeypatch, tmp_path):
    import localpilot.systemsense_hardware as hardware
    collector = hardware.BundledHardwareMonitorCollector(tmp_path / "provider.exe")
    with monkeypatch.context() as patch:
        patch.setattr(hardware.os, "name", "posix")
        process, errors = collector._start_process()
    assert process is None
    assert errors == ["bundled-provider:windows-only"]
