from pathlib import Path

from localpilot.github_integration import classify_workflow_run
from localpilot.learning import LearningMemory


def test_workflow_run_classification():
    assert classify_workflow_run("in_progress", None) == "pending"
    assert classify_workflow_run("completed", "success") == "passed"
    assert classify_workflow_run("completed", "failure") == "failed"
    assert classify_workflow_run("completed", "timed_out") == "failed"


def test_learning_memory_exposes_failed_candidate_for_repair(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")

    cycle = memory.start_cycle(
        task_id="operator-foundation",
        branch="localpilot/candidate-test",
        everyday_model="gpt-oss:20b",
        developer_model="qwen2.5:32b",
    )

    memory.finish_cycle(
        cycle,
        status="candidate_pending_validation",
        summary="candidate pushed",
        reusable_lesson="validate with CI",
        checks_passed=True,
        pushed=True,
    )

    memory.update_candidate_review(
        cycle,
        validation_state="failed",
        merged=False,
        pull_request_url=None,
    )

    failed = memory.failed_candidates()

    assert len(failed) == 1
    assert failed[0].cycle_id == cycle
    assert failed[0].task_id == "operator-foundation"
    assert failed[0].branch == "localpilot/candidate-test"
