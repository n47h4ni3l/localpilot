from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from localpilot.config import Config
from localpilot.github_integration import GitHubIntegration
from localpilot.process import hidden_process_creation_flags


def _version_ok() -> bool:
    return sys.version_info >= (3, 11)


def _ollama_models() -> tuple[set[str], str]:
    try:
        import ollama

        response = ollama.list()
        models = {
            str(getattr(model, "model", "") or getattr(model, "name", ""))
            for model in getattr(response, "models", [])
        }
        return {model for model in models if model}, "Python client"
    except Exception as client_exc:
        executable = shutil.which("ollama")
        if not executable:
            return set(), f"unavailable: {client_exc}"
        try:
            out = subprocess.run(
                [executable, "list"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=hidden_process_creation_flags(),
            )
            if out.returncode != 0:
                return set(), f"CLI failed with return code {out.returncode}"
            models = {
                line.split()[0]
                for line in out.stdout.splitlines()[1:]
                if line.strip()
            }
            return models, "CLI"
        except Exception as cli_exc:
            return set(), f"check failed: {cli_exc}"


def doctor(config: Config, project_root: str | Path) -> list[tuple[str, bool, str]]:
    root = Path(project_root).resolve()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python 3.11+", _version_ok(), platform.python_version()))
    checks.append(("Windows", os.name == "nt", platform.platform()))
    checks.append(("Ollama CLI (optional)", True, shutil.which("ollama") or "not on PATH; Python client is supported"))
    checks.append(("Git", shutil.which("git") is not None, shutil.which("git") or "not found"))
    checks.append(("GitHub CLI (optional)", shutil.which("gh") is not None, shutil.which("gh") or "not found"))

    models, model_source = _ollama_models()
    model_ok = config.model.name in models
    model_detail = f"{config.model.name}: {'installed' if model_ok else 'not installed'} ({model_source})"
    checks.append(("Configured model", model_ok, model_detail))

    gh = GitHubIntegration(root, config.github)
    checks.append(("Local Git repo", gh.is_git_repo(), gh.status()))
    return checks
