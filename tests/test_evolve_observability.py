import json
from pathlib import Path

from rich.console import Console

from localpilot.cli import _show_status, evolve_exit_code
from localpilot.config import Config
from localpilot.learning import LearningMemory
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


def test_status_exposes_current_capability_experiment(tmp_path: Path):
    config = Config()
    config.agent.data_dir = "data"
    task = {
        "id": "capability-status",
        "title": "Improve retrieval selection",
        "status": "todo",
        "evolution_class": "improve_cognition",
        "capability_target": "retrieval quality",
        "question": "Can query-aware ranking improve retrieval?",
        "observed_limitation": "Current retrieval has 0.55 precision at five.",
        "hypothesis": "Query-aware ranking will raise precision at five above 0.70.",
        "evaluation": {
            "metric": "precision at five",
            "baseline": "0.55",
            "success_criterion": "above 0.70",
            "measurement_method": "held-out retrieval benchmark in CI",
        },
    }
    memory = LearningMemory(tmp_path / "data" / config.selfdev.learning_database)
    memory.record_experiment(task)
    memory.update_experiment_outcome(
        task["id"],
        status="evaluation_pending",
        outcome="pending_ci",
    )
    console = Console(record=True, width=120)

    _show_status(console, config, tmp_path)

    rendered = console.export_text()
    assert "% system-wide" in rendered
    assert "improve_cognition" in rendered
    assert "retrieval quality" in rendered
    assert "Query-aware ranking" in rendered
    assert "precision at five" in rendered
    assert "pending_ci" in rendered
