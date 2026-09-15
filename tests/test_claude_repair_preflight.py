from __future__ import annotations

from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.evolution_reliability import SelfDeveloper
from localpilot.implementation_backend import ClaudeCodeBackend, ImplementationPreflight
from localpilot.selfdev import CandidateTools
from localpilot.selfdev_results import CyclePaused


def test_one_verified_preflight_is_reused_within_a_claude_backend_instance(
    tmp_path: Path,
    monkeypatch,
):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    calls: list[int] = []

    def fake_preflight(self):
        calls.append(1)
        return ImplementationPreflight(
            True,
            self.name,
            self.model,
            self.executable,
            version="test",
            context_tokens=self.context_tokens,
        )

    monkeypatch.setattr(ClaudeCodeBackend, "preflight", fake_preflight)

    backend = developer._claude_code_backend(
        force=True,
        branch="localpilot/candidate-preflight-test",
    )

    assert backend.preflight().healthy is True
    assert backend.preflight().healthy is True
    assert backend.preflight().healthy is True
    assert len(calls) == 1


def test_transient_claude_repair_backend_failure_pauses_and_preserves_candidate(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.implementation_backend = "claude_code"
    developer = SelfDeveloper(config, project)
    tools = CandidateTools(workspace)
    task = {
        "id": "repair-preflight",
        "title": "Repair preflight",
        "acceptance": ["candidate remains recoverable"],
        "evaluation": {
            "metric": "tests",
            "baseline": "failing",
            "success_criterion": "passing",
            "measurement_method": "CI",
        },
    }
    developer._active_checkpoint = {
        "cycle_id": 12,
        "task": task,
        "branch": "localpilot/candidate-preflight-test",
        "workspace": workspace,
        "tools": tools,
    }

    monkeypatch.setattr(
        developer,
        "_claude_repair_scope",
        lambda _workspace, _tools: ("module.py",),
    )
    monkeypatch.setattr(developer, "_evolution_context", lambda _task: "bounded contract")
    monkeypatch.setattr(developer.memory, "reusable_lessons", lambda _limit: [])

    def unavailable(**_kwargs):
        raise RuntimeError(
            "implementation backend backend_unavailable: Ollama allocated unknown "
            "context tokens after loading gpt-oss:20b; at least 65536 are required"
        )

    monkeypatch.setattr(developer, "_run_claude_code_implementation", unavailable)

    with pytest.raises(CyclePaused, match="candidate preserved"):
        developer._tool_stage(
            chat=lambda **_kwargs: None,
            model="gpt-oss:20b",
            messages=[{"role": "system", "content": "repair the recorded CI failure"}],
            functions=[],
            rounds=14,
            force=True,
            branch="localpilot/candidate-preflight-test",
            stage="ci_repair",
        )
