from localpilot.safety import RiskLevel
from localpilot.tools import registry


def test_v01_pc_tools_are_observation_only():
    assert registry()
    assert all(spec.risk is RiskLevel.READ_ONLY for spec in registry().values())
