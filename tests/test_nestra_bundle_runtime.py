from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from localpilot.config import load_config
from localpilot.nestra_bundle_runtime import configure


def _installation(tmp_path, monkeypatch):
    app, runtime, models = (tmp_path / name for name in ("app", "runtime", "models"))
    app.mkdir()
    (runtime / "claude").mkdir(parents=True)
    (runtime / "claude/claude.exe").write_bytes(b"test-runtime")
    manifest = models / "manifests-v2/ollama.com/library/nestra/p1.2"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"fixture":"exact accepted metadata"}')
    monkeypatch.setattr("localpilot.nestra_bundle_model.ACCEPTED_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    return app, runtime, models


def test_fresh_config_is_private_guide_only_and_same_model(tmp_path, monkeypatch):
    app, runtime, models = _installation(tmp_path, monkeypatch)
    identity = configure(app, runtime, models)
    cfg = load_config(app / "localpilot.toml")
    assert cfg.agent.name == "Nestra"
    assert cfg.agent.scaffold_mode == "guide_first"
    assert cfg.model.name == cfg.selfdev.implementation_model == cfg.selfdev.developer_model == "nestra:p1.2"
    assert cfg.model.temperature == 1.0
    assert cfg.selfdev.implementation_backend == "claude_code"
    assert cfg.selfdev.implementation_base_url == "http://127.0.0.1:11435"
    assert cfg.desktop.port == 8766
    assert cfg.library.root == str(tmp_path / "library")
    assert cfg.library.enabled
    assert not cfg.github.enabled and not cfg.github.auto_push_candidates
    assert cfg.safety.require_confirmation_for_destructive
    assert not cfg.selfdev.auto_promote
    assert identity == json.loads((app / ".nestra-runtime.json").read_text())
    assert not list((app / "localpilot-data").iterdir())


@pytest.mark.parametrize("existing", ["localpilot.toml", ".nestra-runtime.json"])
def test_existing_owner_configuration_is_preserved(tmp_path, monkeypatch, existing):
    app, runtime, models = _installation(tmp_path, monkeypatch)
    (app / existing).write_bytes(b"owner original")
    with pytest.raises(FileExistsError):
        configure(app, runtime, models)
    assert (app / existing).read_bytes() == b"owner original"


def test_wrong_model_fails_before_creating_config(tmp_path, monkeypatch):
    app, runtime, models = _installation(tmp_path, monkeypatch)
    (models / "manifests-v2/ollama.com/library/nestra/p1.2").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="differs"):
        configure(app, runtime, models)
    assert not (app / "localpilot.toml").exists()


def test_refuses_shared_models_and_production_port(tmp_path, monkeypatch):
    app, runtime, models = _installation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reserves"):
        configure(app, runtime, models, 11434)
    with pytest.raises(ValueError, match="private"):
        configure(app, runtime, tmp_path / "global-models")
    assert not (app / "localpilot.toml").exists()


def test_private_harness_paths_survive_without_forwarding_credentials(monkeypatch):
    from localpilot.implementation_backend import ClaudeCodeBackend
    # Exercise the environment builder without constructing an execution request.
    backend = object.__new__(ClaudeCodeBackend)
    backend.base_url = "http://127.0.0.1:11435"
    backend.context_tokens = 65536
    backend.max_output_tokens = 2048
    monkeypatch.setenv("CLAUDE_CODE_GIT_BASH_PATH", "X:/Nestra/runtime/git/bin/bash.exe")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "X:/Nestra/state/claude")
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-owner-key")
    monkeypatch.setenv("GITHUB_TOKEN", "private-owner-github-token")
    environment = backend._environment()
    assert environment["CLAUDE_CODE_GIT_BASH_PATH"].endswith("bash.exe")
    assert environment["CLAUDE_CONFIG_DIR"].endswith("state/claude")
    assert environment["DISABLE_AUTOUPDATER"] == "1"
    assert environment["ANTHROPIC_API_KEY"] == ""
    assert "GITHUB_TOKEN" not in environment
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert environment["ANTHROPIC_BASE_URL"] == backend.base_url
