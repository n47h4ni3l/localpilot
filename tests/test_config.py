from pathlib import Path

from localpilot.config import load_config


def test_toml_config_is_real(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\nname = "test-model"\n[resource]\nbackground_idle_seconds = 42\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.name == "test-model"
    assert cfg.resource.background_idle_seconds == 42
    assert cfg.source_path == path.resolve()
