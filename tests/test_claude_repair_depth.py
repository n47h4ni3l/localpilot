from __future__ import annotations

import json
import subprocess
from pathlib import Path

from localpilot.config import Config
from localpilot.implementation_backend import (
    ImplementationPreflight,
    ImplementationResult,
    ImplementationStatus,
)
from localpilot.selfdev import CandidateTools, SelfDeveloper


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_claude_can_be_sent_back_more_than_six_times_before_acceptance(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Repair Depth Test")
    _git(root, "config", "user.email", "repair-depth@invalid.local")
    _git(root, "add", "module.py")
    _git(root, "commit", "--quiet", "-m", "fixture")

    config = Config()
    config.agent.data_dir = "data"
    assert config.selfdev.implementation_review_repair_passes == 12

    class RepairingBackend:
        name = "claude_code"

        def __init__(self) -> None:
            self.calls = 0
            self.repair_passes: list[int] = []

        def preflight(self):
            return ImplementationPreflight(
                True,
                self.name,
                "gpt-oss:20b",
                "fake-test-double",
                version="test",
                context_tokens=65536,
            )

        def run(self, request, *, repair_pass=0):
            self.calls += 1
            self.repair_passes.append(repair_pass)
            (request.workspace / "module.py").write_text(
                f"VALUE = {self.calls + 1}\n", encoding="utf-8"
            )
            return ImplementationResult(
                ImplementationStatus.COMPLETED,
                self.name,
                "gpt-oss:20b",
                "candidate changed",
                changed_paths=("module.py",),
                diff_digest=f"{self.calls:064x}"[-64:],
                tests=(
                    {
                        "command": "python -m pytest",
                        "passed": True,
                        "exit_code": 0,
                        "output_digest": "b" * 64,
                    },
                ),
                session_id="same-session",
                repair_pass=repair_pass,
            )

    backend = RepairingBackend()
    developer = SelfDeveloper(config, root, implementation_backend=backend)

    reviews = iter(
        [
            {
                "content": json.dumps(
                    {
                        "approved": False,
                        "feedback": [f"repair issue {index}"],
                        "summary": f"reject {index}",
                    }
                )
            }
            for index in range(1, 9)
        ]
        + [
            {
                "content": json.dumps(
                    {"approved": True, "feedback": [], "summary": "accept"}
                )
            }
        ]
    )
    monkeypatch.setattr(developer, "_developer_chat", lambda *args, **kwargs: next(reviews))

    task = {
        "id": "deep-repair-fixture",
        "title": "deep repair fixture",
        "acceptance": ["review eventually passes"],
        "hypothesis": "bounded repeated repair converges",
        "evaluation": {
            "metric": "review",
            "baseline": "rejected",
            "measurement_method": "independent review",
        },
    }
    result = developer._run_claude_code_implementation(
        chat=lambda **kwargs: None,
        developer_model="gpt-oss:20b",
        task=task,
        branch="candidate/test",
        workspace=root,
        tools=CandidateTools(root),
        cycle_id=1,
        research="evidence",
        grounding_plan={"referenced_paths": ["module.py"], "new_runtime_paths": []},
        grounding_evidence=["module.py"],
        evolution_context="bounded",
        lessons=[],
        force=True,
    )

    assert backend.calls == 9
    assert backend.repair_passes == list(range(9))
    assert "LocalPilot review approved" in result
