from __future__ import annotations

import sys
from types import SimpleNamespace

from localpilot.doctor import _ollama_models


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
