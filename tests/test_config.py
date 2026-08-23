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
    assert cfg.selfdev.developer_model_fallbacks == ["qwen2.5:14b"]
    assert cfg.selfdev.ollama_keep_alive == 0
    assert cfg.selfdev.candidate_file_soft_budget == 100
    assert cfg.selfdev.candidate_file_hard_ceiling == 500
    assert cfg.selfdev.candidate_resource_quota_gb == 8.0


def test_toml_loads_developer_model(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\nname = "daily"\n[selfdev]\ndeveloper_model = "dev"\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.name == "daily"
    assert cfg.selfdev.developer_model == "dev"


def test_toml_loads_resource_aware_model_options(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        '[selfdev]\ndeveloper_model_fallbacks = ["small"]\n'
        'model_memory_overhead_gb = 2.5\nollama_keep_alive = "0s"\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.selfdev.developer_model_fallbacks == ["small"]
    assert cfg.selfdev.model_memory_overhead_gb == 2.5
    assert cfg.selfdev.ollama_keep_alive == "0s"


def test_auto_promotion_cannot_be_enabled(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text("[selfdev]\nauto_promote = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="auto_promote"):
        load_config(path)


def test_legacy_eight_file_config_migrates_to_useful_budget(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text("[selfdev]\nmax_files_per_cycle = 8\n", encoding="utf-8")

    cfg = load_config(path)

    assert cfg.selfdev.candidate_file_soft_budget == 100
    assert cfg.selfdev.candidate_file_hard_ceiling == 500

