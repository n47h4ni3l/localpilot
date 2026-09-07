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


def test_one_model_is_used_across_operator_planning_review_and_implementation():
    cfg = Config()
    assert cfg.model.name == "gpt-oss:20b"
    assert cfg.model.think == "high"
    assert cfg.model.context_tokens == 32768
    assert cfg.model.ollama_keep_alive == "30m"
    assert cfg.model.memory_embeddings_enabled is False
    assert cfg.model.memory_embedding_model == "embeddinggemma"
    assert cfg.selfdev.developer_model == "gpt-oss:20b"
    assert cfg.selfdev.developer_model_fallbacks == []
    assert cfg.selfdev.implementation_backend == "claude_code"
    assert cfg.selfdev.implementation_model == "gpt-oss:20b"
    assert cfg.selfdev.implementation_context_tokens == 65536
    assert cfg.selfdev.implementation_review_repair_passes == 1
    assert cfg.selfdev.implementation_max_output_tokens == 2048
    assert cfg.selfdev.ollama_keep_alive == 0
    assert cfg.selfdev.candidate_file_soft_budget == 100
    assert cfg.selfdev.candidate_file_hard_ceiling == 500
    assert cfg.selfdev.candidate_resource_quota_gb == 8.0
    assert cfg.selfdev.cycle_wall_clock_seconds == 900
    assert cfg.selfdev.max_tool_calls_per_cycle == 32
    assert cfg.selfdev.max_web_calls_per_cycle == 8
    assert cfg.selfdev.opportunity_similarity_threshold == 0.82


def test_toml_loads_developer_model(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[selfdev]\ndeveloper_model = "dev"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must use only gpt-oss:20b"):
        load_config(path)


def test_previous_shipped_qwen_defaults_migrate_to_one_model(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        '[selfdev]\ndeveloper_model = "qwen2.5:32b"\n'
        'developer_model_fallbacks = ["qwen2.5:14b"]\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.selfdev.developer_model == "gpt-oss:20b"
    assert cfg.selfdev.developer_model_fallbacks == []


def test_toml_loads_operator_context_window(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\ncontext_tokens = 16384\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.context_tokens == 16384


def test_toml_loads_bounded_memory_embedding_options(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        '[model]\nmemory_embeddings_enabled = true\n'
        'memory_embedding_model = "all-minilm"\n'
        'memory_semantic_weight = 9.5\n'
        'memory_semantic_min_similarity = 0.4\n'
        'memory_embedding_batch_size = 32\n'
        'memory_embedding_migration_limit = 200\n',
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.model.memory_embeddings_enabled is True
    assert cfg.model.memory_embedding_model == "all-minilm"
    assert cfg.model.memory_semantic_weight == 9.5
    assert cfg.model.memory_semantic_min_similarity == 0.4
    assert cfg.model.memory_embedding_batch_size == 32
    assert cfg.model.memory_embedding_migration_limit == 200


def test_invalid_memory_embedding_bounds_fail_closed(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        '[model]\nmemory_embeddings_enabled = true\n'
        'memory_embedding_model = ""\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="memory_embedding_model"):
        load_config(path)


def test_context_window_rejects_accidentally_tiny_or_invalid_values(tmp_path: Path):
    tiny = tmp_path / "tiny.toml"
    tiny.write_text('[model]\ncontext_tokens = 2048\n', encoding="utf-8")
    with pytest.raises(ValueError, match="between 4096 and 131072"):
        load_config(tiny)

    boolean = tmp_path / "boolean.toml"
    boolean.write_text('[model]\ncontext_tokens = true\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integer token count"):
        load_config(boolean)


def test_toml_loads_resource_aware_model_options(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        '[selfdev]\nmodel_memory_overhead_gb = 2.5\nollama_keep_alive = "0s"\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.selfdev.developer_model_fallbacks == []
    assert cfg.selfdev.model_memory_overhead_gb == 2.5
    assert cfg.selfdev.ollama_keep_alive == "0s"


def test_implementation_backend_bounds_and_loopback_are_validated(tmp_path: Path):
    too_small = tmp_path / "small.toml"
    too_small.write_text(
        '[selfdev]\nimplementation_context_tokens = 32768\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least 65536"):
        load_config(too_small)

    remote = tmp_path / "remote.toml"
    remote.write_text(
        '[selfdev]\nimplementation_base_url = "https://api.anthropic.com"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loopback"):
        load_config(remote)

    too_many_tokens = tmp_path / "tokens.toml"
    too_many_tokens.write_text(
        '[selfdev]\nimplementation_max_output_tokens = 128\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="max_output_tokens"):
        load_config(too_many_tokens)


def test_local_tools_backend_is_an_explicit_rollback_setting(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[selfdev]\nimplementation_backend = "local_tools"\n', encoding="utf-8")
    assert load_config(path).selfdev.implementation_backend == "local_tools"


def test_gpt_oss_boolean_thinking_migrates_to_explicit_levels(tmp_path: Path):
    enabled = tmp_path / "enabled.toml"
    enabled.write_text('[model]\nname = "gpt-oss:20b"\nthink = true\n', encoding="utf-8")
    assert load_config(enabled).model.think == "high"

    disabled = tmp_path / "disabled.toml"
    disabled.write_text('[model]\nname = "gpt-oss:20b"\nthink = false\n', encoding="utf-8")
    assert load_config(disabled).model.think == "low"


def test_gpt_oss_rejects_unsupported_reasoning_level(tmp_path: Path):
    path = tmp_path / "localpilot.toml"
    path.write_text('[model]\nname = "gpt-oss:20b"\nthink = "max"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="low, medium, or high"):
        load_config(path)


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
