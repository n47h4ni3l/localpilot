from localpilot.config import ResourceConfig
from localpilot.resource import ResourceGovernor


def test_resource_config_defaults_are_sane():
    cfg = ResourceConfig()
    assert cfg.background_idle_seconds >= 60
    assert 0 < cfg.max_cpu_percent_for_background <= 100
    assert 0 < cfg.max_memory_percent_for_background <= 100
    governor = ResourceGovernor(cfg)
    assert governor.config is cfg
