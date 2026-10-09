from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "training" / "scripts" / "training_memory_policy.ps1"
GUARD = ROOT / "training" / "scripts" / "start_guarded_adapter.ps1"
LAUNCH = ROOT / "training" / "scripts" / "launch_adapter_v1.sh"
MONITOR = ROOT / "training" / "scripts" / "show_adapter_training.ps1"


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
    critical = _decision(
        available=0.1,
        commit=0.5,
        elapsed=40.0,
        mode="MonitorOnly",
    )

    assert result["warning"] is True
    assert result["pressure_reason"] in {"sustained_low_ram", "sustained_low_commit"}
    assert result["stop"] is False
    assert result["reason"] is None
    assert result["mode"] == "MonitorOnly"

    # MonitorOnly is deliberately observational even at critical pressure.
    assert critical["severity"] == "critical"
    assert critical["warning"] is True
    assert critical["stop"] is False
    assert critical["reason"] is None


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
    assert "windows_memory_breakdown" in text
    assert "Get-WindowsMemoryBreakdown" in text
    assert "wsl_memory" in text
    assert "Get-WslMemoryState" in text
    assert "checkpoint_activity" in text
    assert "Get-CheckpointActivity" in text
    assert "latest_complete_checkpoint_step" in text
    assert "$minimumWslMemoryGiB = 24.0" in text
    assert "$minimumWslSwapGiB = 24.0" in text
    assert "offload_embeddings" in text
    assert "Unsloth disables embedding offload on WSL" in text
    assert "$dryRunPath" in text

    launch = LAUNCH.read_text(encoding="utf-8")
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in launch
    assert "unset PYTORCH_ALLOC_CONF" in launch
    assert 'run_name="$(python -c' in launch

    monitor = MONITOR.read_text(encoding="utf-8")
    assert "config.output.directory" in monitor
    assert "estimated_optimizer_steps" in monitor


@pytest.mark.parametrize("resources,accepted", [
    ({"offload_embeddings": False, "cpu_offload": "none"}, True),
    ({"offload_embeddings": True, "cpu_offload": "embeddings_only"}, False),
    ({"offload_embeddings": False, "cpu_offload": "embeddings_only"}, False),
    ({"offload_embeddings": True, "cpu_offload": "none"}, False),
    ({"offload_embeddings": "false", "cpu_offload": "none"}, False),
    ({"offload_embeddings": 0, "cpu_offload": "none"}, False),
    ({"cpu_offload": "none"}, False),
    ({"offload_embeddings": False}, False),
])
def test_guard_validates_wsl_offload_request_without_claiming_runtime_placement(resources, accepted) -> None:
    # Execute only the real launcher's config gate, without touching WSL,
    # scheduled tasks, processes, or training outputs.
    command = (
        "$ErrorActionPreference='Stop'; "
        "$tokens=$null; $errors=$null; "
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{GUARD}', [ref]$tokens, [ref]$errors); "
        "$gate=$ast.Find({param($node) "
        "$node -is [System.Management.Automation.Language.IfStatementAst] "
        "-and $node.Extent.Text -match '^if \\(\\$config\\.resources\\.offload_embeddings'}, $true); "
        "if ($null -eq $gate) { throw 'Missing offload validation gate' }; "
        "$config=@{resources=([Console]::In.ReadToEnd() | ConvertFrom-Json)}; "
        "& ([scriptblock]::Create($gate.Extent.Text))"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        input=json.dumps(resources), capture_output=True, text=True, timeout=20,
    )
    if accepted:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "Unsloth disables embedding offload on WSL" in result.stderr



@pytest.mark.parametrize("shell", ["powershell.exe"] + ([shutil.which("pwsh")] if shutil.which("pwsh") else []))
def test_guard_sleep_prevention_sets_and_releases_the_windows_idle_sleep_request(shell) -> None:
    # Run only the guard's power-management block in a child process. Verify
    # the real Windows API accepted its flags, then clear them before exit.
    command = (
        "$ErrorActionPreference='Stop'; $WarningPreference='Stop'; "
        "$tokens=$null; $errors=$null; "
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{GUARD}', [ref]$tokens, [ref]$errors); "
        "$block=$ast.Find({param($node) "
        "$node -is [System.Management.Automation.Language.TryStatementAst] "
        "-and $node.Extent.Text.Contains('Add-Type -Name PowerManagement')}, $true); "
        "if ($null -eq $block) { throw 'Missing sleep-prevention block' }; "
        "try { "
        "& ([scriptblock]::Create($block.Extent.Text)); "
        "$previous=[LocalPilot.PowerManagement]::SetThreadExecutionState([uint32]2147483648); "
        "if ($previous -ne [uint32]2147483649) { throw ('Unexpected execution state: ' + $previous) }; "
        "} finally { "
        "if ('LocalPilot.PowerManagement' -as [type]) { "
        "$null=[LocalPilot.PowerManagement]::SetThreadExecutionState([uint32]2147483648) } }"
    )
    # Windows CI can spend more than 20 seconds cold-starting PowerShell 7
    # and compiling Add-Type with .NET. The subprocess must still execute the
    # real SetThreadExecutionState checks above; only allow more startup time.
    subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, text=True, timeout=90,
    )


def test_training_powershell_scripts_parse_cleanly() -> None:
    for path in (POLICY, GUARD, MONITOR):
        command = (
            "$tokens=$null; $errors=$null; "
            f"[void][System.Management.Automation.Language.Parser]::ParseFile('{path}', "
            "[ref]$tokens, [ref]$errors); "
            "if ($errors.Count -gt 0) { "
            "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
        )
        subprocess.run(
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
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
