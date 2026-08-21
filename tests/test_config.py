from pathlib import Path

import pytest

from localpilot.config import Config, load_config


def test_toml_config_is_real(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\nname = "test-model"\n[resource]\nbackground_idle_seconds = 42\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.name == "test-model"
    assert cfg.resource.background_idle_seconds == 42
    assert cfg.source_path == path.resolve()


def test_everyday_and_developer_models_are_separate():
    cfg = Config()
    assert cfg.model.name == "gpt-oss:20b"
    assert cfg.selfdev.developer_model == "qwen2.5:32b"


def test_toml_loads_developer_model(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\nname = "daily"\n[selfdev]\ndeveloper_model = "dev"\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.name == "daily"
    assert cfg.selfdev.developer_model == "dev"


def test_auto_promotion_cannot_be_enabled(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text("[selfdev]\nauto_promote = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="auto_promote"):
        load_config(path)

