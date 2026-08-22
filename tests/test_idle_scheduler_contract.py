from pathlib import Path


def test_idle_scheduler_preserves_evolve_safety_contract():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install-idle-evolve-task.ps1"
    ).read_text(encoding="utf-8")

    assert 'TrustedBranch = "main"' in script
    assert "branch --show-current" in script
    assert "status --porcelain --untracked-files=all" in script
    assert ".venv\\Scripts\\localpilot.exe" in script
    assert 'actionArguments = "--config' in script
    assert " evolve\"" in script
    assert "MultipleInstances IgnoreNew" in script
    assert "--force" not in script.split("Register-ScheduledTask", 1)[0]
