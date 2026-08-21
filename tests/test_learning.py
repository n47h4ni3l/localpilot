from pathlib import Path

from localpilot.learning import LearningMemory


def _cycle(memory: LearningMemory) -> int:
    return memory.start_cycle(
        task_id="task-1",
        branch="localpilot/candidate-task-1",
        everyday_model="gpt-oss:20b",
        developer_model="qwen2.5:32b",
    )


def test_learning_memory_persists_outcomes_and_lessons(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    memory = LearningMemory(path)
    cycle = _cycle(memory)
    memory.finish_cycle(
        cycle,
        status="failed",
        summary="Static check failed.",
        reusable_lesson="Compile every edited Python file before pushing.",
        checks_passed=False,
        pushed=False,
    )
    reopened = LearningMemory(path)
    assert reopened.reusable_lessons() == ["Compile every edited Python file before pushing."]


def test_memory_schema_has_no_hidden_reasoning_fields(tmp_path: Path):
    columns = LearningMemory(tmp_path / "learning.sqlite3").schema_columns()
    forbidden = {"reasoning", "chain_of_thought", "thinking", "prompt", "transcript", "messages"}
    assert not columns & forbidden


def test_task_completes_only_after_checks_pass_and_pr_is_merged(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    cycle = _cycle(memory)
    memory.finish_cycle(
        cycle,
        status="candidate_pending_validation",
        summary="Candidate pushed.",
        reusable_lesson="Keep candidate changes focused.",
        checks_passed=True,
        pushed=True,
    )
    memory.update_candidate_review(cycle, validation_state="passed", merged=False, pull_request_url="https://example/pr/1")
    assert memory.completed_task_ids() == set()
    assert memory.pending_task_ids() == {"task-1"}
    memory.update_candidate_review(cycle, validation_state="pending", merged=True, pull_request_url="https://example/pr/1")
    assert memory.completed_task_ids() == set()
    memory.update_candidate_review(cycle, validation_state="passed", merged=True, pull_request_url="https://example/pr/1")
    assert memory.completed_task_ids() == {"task-1"}
    assert memory.pending_task_ids() == set()

