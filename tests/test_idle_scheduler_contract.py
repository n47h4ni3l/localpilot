from pathlib import Path


def test_idle_scheduler_preserves_evolve_safety_contract():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install-idle-evolve-task.ps1"
    ).read_text(encoding="utf-8")

    assert 'TrustedBranch = "main"' in script
    assert "branch --show-current" in script
    assert "status --porcelain --untracked-files=all" in script
    assert ".venv\\Scripts\\pythonw.exe" in script
    assert "-m localpilot.background_worker" in script
    assert "--interval-seconds $PollSeconds" in script
    assert "AtLogOn" in script
    assert "RepetitionInterval" in script
    assert "WatchdogMinutes = 1" in script
    assert "MultipleInstances IgnoreNew" in script
    assert "RestartCount 3" in script
    assert "MainWindowHandle -eq 0" in script
    assert "background_worker_cycle_start" in script
    assert "Disable-ScheduledTask" in script
    assert "TaskName and LegacyTaskName must be different" in script
    assert script.index("background_worker_cycle_start") < script.index("Disable-ScheduledTask")
    assert "--force" not in script.split("Register-ScheduledTask", 1)[0]
