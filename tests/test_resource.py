from localpilot.config import ResourceConfig
from localpilot.resource import ResourceGovernor, ResourceState


def test_resource_config_defaults_are_sane():
    cfg = ResourceConfig()
    assert cfg.background_idle_seconds >= 60
    assert 0 < cfg.max_cpu_percent_for_background <= 100
    assert 0 < cfg.max_memory_percent_for_background <= 100
    governor = ResourceGovernor(cfg)
    assert governor.config is cfg


def test_force_bypasses_idle_wait_but_not_capacity_protection():
    idle_only = ResourceState(
        0,
        10,
        40,
        False,
        "user active",
        idle_allowed=False,
        capacity_allowed=True,
        idle_reason="user active",
    )
    assert idle_only.allows_selfdev(ignore_idle=True) is True

    memory_pressure = ResourceState(
        1000,
        10,
        99,
        False,
        "memory pressure",
        idle_allowed=True,
        capacity_allowed=False,
        capacity_reason="memory pressure",
    )
    assert memory_pressure.allows_selfdev(ignore_idle=True) is False
    assert memory_pressure.blocking_reason(ignore_idle=True) == "memory pressure"
