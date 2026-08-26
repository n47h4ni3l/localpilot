from localpilot.safety import RiskLevel
from localpilot.tools import registry


def test_pc_registry_has_only_observations_and_the_exact_reversible_allowlist():
    tools = registry()
    reversible = {
        name for name, spec in tools.items()
        if spec.risk is RiskLevel.REVERSIBLE
    }

    assert reversible == {
        "open_windows_app",
        "open_windows_settings",
        "set_active_power_plan",
        "restore_power_plan",
    }
    assert all(spec.risk is not RiskLevel.DESTRUCTIVE for spec in tools.values())
    assert all(
        spec.risk is RiskLevel.READ_ONLY
        for name, spec in tools.items()
        if name not in reversible
    )
