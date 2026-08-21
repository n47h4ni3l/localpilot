from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from localpilot.config import Config
from localpilot.github_integration import GitHubIntegration


def _version_ok() -> bool:
    return sys.version_info >= (3, 11)


def doctor(config: Config, project_root: str | Path) -> list[tuple[str, bool, str]]:
    root = Path(project_root).resolve()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python 3.11+", _version_ok(), platform.python_version()))
    checks.append(("Windows", os.name == "nt", platform.platform()))
    checks.append(("Ollama CLI", shutil.which("ollama") is not None, shutil.which("ollama") or "not found"))
    checks.append(("Git", shutil.which("git") is not None, shutil.which("git") or "not found"))
    checks.append(("GitHub CLI (optional)", shutil.which("gh") is not None, shutil.which("gh") or "not found"))

    model_ok = False
    model_detail = "Ollama unavailable"
    if shutil.which("ollama"):
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15, check=False)
            model_ok = out.returncode == 0 and config.model.name in out.stdout
            model_detail = f"{config.model.name}: {'installed' if model_ok else 'not installed'}"
        except Exception as exc:
            model_detail = f"check failed: {exc}"
    checks.append(("Configured model", model_ok, model_detail))

    gh = GitHubIntegration(root, config.github)
    checks.append(("Local Git repo", gh.is_git_repo(), gh.status()))
    return checks
