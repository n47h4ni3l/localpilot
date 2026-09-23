"""Execute the real PowerShell policy without launching WSL or a model."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


PWSH = shutil.which("pwsh")
POLICY = Path(__file__).resolve().parents[1] / "scripts/training_memory_policy.ps1"
pytestmark = pytest.mark.skipif(PWSH is None, reason="PowerShell 7 required")


def decisions(samples, mode="MonitorOnly"):
    script = """
    $ErrorActionPreference = 'Stop'
    . $env:LOCALPILOT_TEST_POLICY
    $history = @{}
    $samples = [Console]::In.ReadToEnd() | ConvertFrom-Json -NoEnumerate
    $results = @(foreach ($sample in $samples) {
        Get-TrainingMemoryDecision -AvailableGiB $sample[0] -CommitHeadroomGiB $sample[1] -ElapsedSeconds $sample[2] -History $history -Mode $env:LOCALPILOT_TEST_MODE
    })
    ConvertTo-Json -InputObject $results -Compress
    """
    import os
    result = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        input=json.dumps(samples), text=True, capture_output=True, timeout=20,
        env={
            **os.environ,
            "LOCALPILOT_TEST_POLICY": str(POLICY),
            "LOCALPILOT_TEST_MODE": mode,
        },
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def test_previous_stop_can_recover_without_interrupting():
    result = decisions([[4, 8, 0], [0.61, 6.69, 3], [0.7, 6, 10], [1.5, 7, 20]])
    assert not any(item["stop"] for item in result)
    assert result[1]["warning"]
    assert result[-1]["low_ram_seconds"] == 0


@pytest.mark.parametrize("samples,reason", [
    ([[0.7, 6, 0], [0.7, 6, 29], [0.7, 6, 30]], "sustained_low_ram"),
    ([[5, 2, 0], [5, 2, 14], [5, 2, 15]], "sustained_low_commit"),
])
def test_sustained_pressure_uses_elapsed_time(samples, reason):
    result = decisions(samples, mode="Protective")
    assert not any(item["stop"] for item in result[:-1])
    assert result[-1]["stop"] and result[-1]["reason"] == reason


@pytest.mark.parametrize("ram,commit,reason", [
    (0.24, 6, "critical_ram"), (5, 0.99, "critical_commit"),
    (5, -0.1, "critical_commit"),
])
def test_emergency_stops_immediately(ram, commit, reason):
    result = decisions([[ram, commit, 0]], mode="Protective")[0]
    assert result["stop"] and result["reason"] == reason


def test_warning_zone_does_not_eventually_stop():
    result = decisions([[1.1, 4, 0], [1.1, 4, 300]])
    assert all(item["warning"] and not item["stop"] for item in result)


def test_recovery_resets_each_timer_independently():
    result = decisions([[0.7, 2, 0], [2, 3, 10], [0.7, 2, 11], [0.7, 3, 25], [0.7, 3, 40]])
    assert not any(item["stop"] for item in result)
    assert result[-1]["low_ram_seconds"] == 29
    assert result[-1]["low_commit_seconds"] == 0


def test_exact_thresholds_are_not_below_threshold():
    assert not any(item["stop"] for item in decisions([[1, 2.5, 0], [1, 2.5, 300]]))


@pytest.mark.parametrize("samples", [
    [[-1, 5, 0]], [[5, 5, -1]], [[5, 5, 10], [5, 5, 9]],
])
def test_invalid_reading_fails_closed(samples):
    with pytest.raises(RuntimeError):
        decisions(samples)
