from __future__ import annotations

import sys
from types import SimpleNamespace

from localpilot.doctor import _ollama_models

import pytest

from localpilot.config import Config
from localpilot.doctor import doctor


def test_ollama_models_uses_the_python_client_without_a_cli(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(
            list=lambda: SimpleNamespace(
                models=[SimpleNamespace(model="gpt-oss:20b"), SimpleNamespace(model="qwen2.5:32b")]
            )
        ),
    )
    monkeypatch.setattr("localpilot.doctor.shutil.which", lambda _name: None)

    models, source = _ollama_models()

    assert models == {"gpt-oss:20b", "qwen2.5:32b"}
    assert source == "Python client"


@pytest.mark.parametrize("backend,enabled,exists", [("claude_code", True, True), ("claude_code", True, False), ("local_tools", True, False), ("claude_code", False, False)])
def test_doctor_checks_only_the_selected_enabled_claude_executable(monkeypatch, tmp_path, backend, enabled, exists):
    cfg = Config()
    cfg.selfdev.enabled = enabled
    cfg.selfdev.implementation_backend = backend
    executable = tmp_path / "custom-claude.exe"
    cfg.selfdev.implementation_executable = str(executable)
    if exists:
        executable.write_bytes(b"executable fixture")
    monkeypatch.setattr("localpilot.doctor.shutil.which", lambda _name: None)
    monkeypatch.setattr("localpilot.doctor._ollama_models", lambda: ({cfg.model.name}, "test"))
    checks = {name: (healthy, detail) for name, healthy, detail in doctor(cfg, tmp_path)}
    name = "Claude Code implementation executable"
    if enabled and backend == "claude_code":
        assert checks[name][0] is exists
        assert str(executable) in checks[name][1]
    else:
        assert name not in checks
