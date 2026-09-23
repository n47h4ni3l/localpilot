from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "training" / "scripts" / "training_memory_policy.ps1"
GUARD = ROOT / "training" / "scripts" / "start_guarded_adapter.ps1"


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell policy")


def _decision(*, available: float, commit: float, elapsed: float, mode: str) -> dict:
    command = (
        f". '{POLICY}'; "
        "$h=@{}; "
        f"$d=Get-TrainingMemoryDecision -AvailableGiB {available} "
        f"-CommitHeadroomGiB {commit} -ElapsedSeconds {elapsed} "
        f"-History $h -Mode '{mode}'; "
        "$d | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return json.loads(completed.stdout.strip())


def _sustained_decision(*, mode: str) -> dict:
    command = (
        f". '{POLICY}'; "
        "$h=@{}; "
        "$null=Get-TrainingMemoryDecision -AvailableGiB 0.8 "
        "-CommitHeadroomGiB 2.0 -ElapsedSeconds 0 -History $h "
        f"-Mode '{mode}'; "
        "$d=Get-TrainingMemoryDecision -AvailableGiB 0.8 "
        "-CommitHeadroomGiB 2.0 -ElapsedSeconds 35 -History $h "
        f"-Mode '{mode}'; "
        "$d | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return json.loads(completed.stdout.strip())


def test_monitor_only_records_pressure_without_stopping_training() -> None:
    result = _sustained_decision(mode="MonitorOnly")

    assert result["warning"] is True
    assert result["pressure_reason"] in {"sustained_low_ram", "sustained_low_commit"}
    assert result["stop"] is False
    assert result["reason"] is None
    assert result["mode"] == "MonitorOnly"


def test_protective_mode_preserves_the_old_sustained_stop_option() -> None:
    result = _sustained_decision(mode="Protective")

    assert result["stop"] is True
    assert result["reason"] in {"sustained_low_ram", "sustained_low_commit"}


def test_critical_only_ignores_sustained_pressure_but_stops_at_extreme_headroom() -> None:
    sustained = _sustained_decision(mode="CriticalOnly")
    critical = _decision(
        available=4.0,
        commit=0.5,
        elapsed=10.0,
        mode="CriticalOnly",
    )

    assert sustained["stop"] is False
    assert critical["stop"] is True
    assert critical["reason"] == "critical_commit"
    assert critical["severity"] == "critical"


def test_training_launcher_defaults_to_observation_not_intervention() -> None:
    text = GUARD.read_text(encoding="utf-8")
    assert "[string]$MemoryPolicy = 'MonitorOnly'" in text
    assert "top_memory_processes" in text
    assert "gpu_process_memory" in text
    assert "latest_complete_checkpoint_step" in text
