import json
from pathlib import Path

from localpilot.cli import evolve_exit_code
from localpilot.config import Config
from localpilot.selfdev import SelfDeveloper


def test_every_run_records_a_terminal_outcome(tmp_path: Path):
    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.enabled = False

    result = SelfDeveloper(config, tmp_path).run_once()

    rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["evolve_run_start", "evolve_run_end"]
    assert rows[-1]["status"] == "disabled"
    assert rows[0]["invocation_id"] == rows[-1]["invocation_id"]


def test_scheduler_exit_code_exposes_internal_failures():
    assert evolve_exit_code("failed") == 1
    assert evolve_exit_code("sync_blocked") == 1
    assert evolve_exit_code("candidate_needs_work") == 1
    assert evolve_exit_code("deferred") == 0
    assert evolve_exit_code("paused") == 0
    assert evolve_exit_code("candidate_pending_validation") == 0
