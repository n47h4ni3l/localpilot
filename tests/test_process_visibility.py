from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import localpilot.process as process_helpers
from localpilot.config import GitHubConfig
from localpilot.github_integration import GitHubIntegration


def test_hidden_process_creation_flags_are_platform_safe(monkeypatch) -> None:
    monkeypatch.setattr(process_helpers.os, "name", "posix")
    assert process_helpers.hidden_process_creation_flags() == 0

    monkeypatch.setattr(process_helpers.os, "name", "nt")
    monkeypatch.setattr(process_helpers.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    assert process_helpers.hidden_process_creation_flags() == 0x08000000


def test_github_commands_are_started_without_a_console_window(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    integration = GitHubIntegration(tmp_path, GitHubConfig())

    assert integration._run(["git", "branch", "--show-current"]).ok
    assert calls[0][1]["creationflags"] == process_helpers.hidden_process_creation_flags()
