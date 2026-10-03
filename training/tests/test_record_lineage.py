from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from training.scripts import record_lineage as recorder
from training.scripts import train_adapter as runner


def _args(config: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        package=1,
        candidate_model="nestra:20b-p1",
        candidate_digest="1" * 64,
        candidate_gguf_sha256="2" * 64,
        rollback_model="gpt-oss:20b",
        rollback_digest="3" * 64,
        eval_overall=2.04,
        eval_critical=1.875,
        eval_hard_failures=3,
        execution_overall=4.0,
        execution_hard_failures=1,
        execution_scope_violations=0,
        note="Owner accepted P1 as the working lineage while retaining mixed benchmark evidence.",
        benchmark_promoted=False,
    )


def test_build_manifest_binds_completed_adapter_and_owner_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "training/outputs"
    output = output_root / "package-1"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"p1-adapter")
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    adapter_files = runner._artifact_inventory(
        adapter, ("adapter_model.safetensors", "adapter_config.json")
    )
    marker = {
        "schema_version": 1,
        "global_step": 11112,
        "run_identity_sha256": "4" * 64,
        "training_config_sha256": "5" * 64,
        "adapter_files": adapter_files,
    }
    (output / runner.TRAINING_COMPLETE_FILE).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    config_path = tmp_path / "training/configs/p1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    config = {
        "model": {
            "base_identity": "openai/gpt-oss-20b",
            "base_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "training_model_id": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "revision": "093fba6992ef5a7152481afec0bdfca1ac486998",
        },
        "output": {"directory": "training/outputs/package-1"},
    }

    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "TRAINING_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(recorder, "load_config", lambda _path: config)

    result = recorder.build_manifest(_args(config_path))

    assert result["artifact_type"] == "nestra_lineage_head"
    assert result["package"] == 1
    assert result["accepted_as_lineage_head"] is True
    assert result["benchmark_promoted"] is False
    assert result["training"]["adapter_files"] == adapter_files
    assert result["training"]["completion_marker_sha256"] == runner._sha256_file(
        output / runner.TRAINING_COMPLETE_FILE
    )
    assert result["deployment"]["model"] == "nestra:20b-p1"
    assert result["rollback"]["model"] == "gpt-oss:20b"
    assert result["evaluation"]["eval_v1"]["hard_failures"] == 3
    assert result["evaluation"]["evolution_execution_v1"]["hard_failures"] == 1


def test_build_manifest_refuses_adapter_changed_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "training/outputs"
    output = output_root / "package-1"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"original")
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    original_files = runner._artifact_inventory(
        adapter, ("adapter_model.safetensors", "adapter_config.json")
    )
    (output / runner.TRAINING_COMPLETE_FILE).write_text(
        json.dumps({
            "schema_version": 1,
            "global_step": 11112,
            "run_identity_sha256": "4" * 64,
            "training_config_sha256": "5" * 64,
            "adapter_files": original_files,
        }),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    config_path = tmp_path / "training/configs/p1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "TRAINING_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        recorder,
        "load_config",
        lambda _path: {
            "model": {
                "base_identity": "openai/gpt-oss-20b",
                "base_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                "training_model_id": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                "revision": "093fba6992ef5a7152481afec0bdfca1ac486998",
            },
            "output": {"directory": "training/outputs/package-1"},
        },
    )

    with pytest.raises(RuntimeError, match="Final adapter differs"):
        recorder.build_manifest(_args(config_path))
