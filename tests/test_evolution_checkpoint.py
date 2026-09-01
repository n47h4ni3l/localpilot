import json
from dataclasses import asdict
from pathlib import Path

import pytest

from localpilot.checkpoint import CheckpointStore, EvolutionCheckpoint
from localpilot.config import Config
from localpilot.foreground import write_foreground_turns
from localpilot.github_integration import CandidateSnapshot
from localpilot.resource import ResourceState
from localpilot.selfdev import (
    CandidateTools,
    CyclePaused,
    DeveloperModelSelection,
    EvolutionResult,
    SelfDeveloper,
)


TASK = {
    "id": "checkpoint-task",
    "title": "Add resumable evolution checkpoints",
    "status": "todo",
    "acceptance": ["Resume the same candidate", "Reject stale state"],
}
BRANCH = "localpilot/candidate-checkpoint-task"
HEAD = "a" * 40


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "selfdev-backlog.json").write_text(
        json.dumps({"tasks": [TASK]}),
        encoding="utf-8",
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    return project, workspace


def _developer(tmp_path: Path) -> tuple[SelfDeveloper, Path]:
    project, workspace = _project(tmp_path)
    config = Config()
    config.agent.data_dir = "data"
    return SelfDeveloper(config, project), workspace


def _snapshot(*, digest: str = "d" * 64, changed: tuple[str, ...] = ()) -> CandidateSnapshot:
    return CandidateSnapshot(BRANCH, HEAD, changed, digest)


def _start_cycle(developer: SelfDeveloper, workspace: Path) -> int:
    return developer.memory.start_cycle(
        task_id=TASK["id"],
        branch=BRANCH,
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=True,
    )


def _save_checkpoint(
    developer: SelfDeveloper,
    workspace: Path,
    monkeypatch,
    *,
    milestone: str = "research_complete",
    findings: tuple[str, ...] = ("The evolution loop lives in localpilot/selfdev.py.",),
) -> int:
    cycle_id = _start_cycle(developer, workspace)
    monkeypatch.setattr(developer.github, "candidate_snapshot", lambda _workspace: _snapshot())
    developer._activate_checkpoint(
        cycle_id=cycle_id,
        task=TASK,
        branch=BRANCH,
        workspace=workspace,
        tools=CandidateTools(workspace),
        milestone=milestone,
        research_findings=findings,
        decisions=["Keep the handoff compact and versioned."],
        next_action="Implement the checkpoint store.",
        reusable_lessons=["Validate durable context before reuse."],
    )
    return cycle_id


def test_checkpoint_save_is_compact_structured_and_redacted(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = EvolutionCheckpoint.create(
        cycle_id=7,
        task=TASK,
        branch=BRANCH,
        workspace=tmp_path / "candidate",
        milestone="implementation",
        files_inspected=["localpilot/selfdev.py"],
        files_changed=["localpilot/checkpoint.py"],
        research_findings=["password=hunter2", "Use an atomic JSON replacement."],
        decisions=["Persist only reviewable engineering facts."],
        git_head=HEAD,
        git_state_digest="d" * 64,
        diff_status="1 changed path(s)",
        static_check_status="failed",
        static_check_failures=["localpilot/checkpoint.py: SyntaxError"],
        test_status="not run locally; GitHub CI required",
        test_failures=[],
        unresolved_questions=["Should old schema versions be migrated?"],
        next_action="Repair the syntax failure.",
        reusable_lessons=["Reject unknown schema versions."],
    )

    store.save(checkpoint)
    loaded = store.load()
    raw = store.path.read_text(encoding="utf-8")

    assert loaded == checkpoint
    assert "hunter2" not in raw
    assert "<redacted>" in raw
    assert "prompt" not in raw.lower()
    assert "transcript" not in raw.lower()
    assert "content" not in json.loads(raw)
    assert loaded.test_status == "not run locally; GitHub CI required"
    assert loaded.test_failures == ()
    assert len(raw) < 10_000


def test_version_one_checkpoint_migrates_without_losing_resume_identity(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    current = EvolutionCheckpoint.create(
        cycle_id=9,
        task=TASK,
        branch=BRANCH,
        workspace=tmp_path / "candidate",
        milestone="research_complete",
        git_head=HEAD,
        git_state_digest="d" * 64,
        next_action="Resume the same candidate.",
    )
    legacy = asdict(current)
    legacy["version"] = 1
    for name in ("evolution_class", "capability_target", "hypothesis", "evaluation_plan"):
        legacy.pop(name)
    store.path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = store.load()

    assert migrated is not None
    assert migrated.version == 2
    assert migrated.cycle_id == 9
    assert migrated.branch == BRANCH
    assert migrated.capability_target == TASK["title"]


def test_unstructured_model_text_is_not_used_as_durable_handoff():
    findings, decisions, questions, _next_action = SelfDeveloper._research_handoff(
        "private scratch analysis that did not follow the JSON contract"
    )
    summary, _lesson = SelfDeveloper._checkpoint_outcome(
        "private scratch analysis that did not follow the JSON contract",
        "Use structured facts.",
    )

    assert all("private scratch" not in item for item in findings + decisions + questions)
    assert "private scratch" not in summary


def test_valid_checkpoint_resumes_only_matching_git_and_task_state(tmp_path: Path, monkeypatch):
    developer, workspace = _developer(tmp_path)
    cycle_id = _save_checkpoint(developer, workspace, monkeypatch)
    developer._active_checkpoint = None
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda _branch: workspace)
    monkeypatch.setattr(developer.github, "candidate_snapshot", lambda _workspace: _snapshot())

    validated = developer._validated_checkpoint()

    assert validated is not None
    checkpoint, candidate, task, resolved_workspace = validated
    assert checkpoint.cycle_id == cycle_id
    assert candidate.branch == BRANCH
    assert task["acceptance"] == TASK["acceptance"]
    assert resolved_workspace == workspace.resolve()
    assert developer.audit.latest("selfdev_checkpoint_resume")["status"] == "succeeded"


def test_stale_checkpoint_is_rejected_and_removed(tmp_path: Path, monkeypatch):
    developer, workspace = _developer(tmp_path)
    _save_checkpoint(developer, workspace, monkeypatch)
    developer._active_checkpoint = None
    monkeypatch.setattr(developer.github, "worktree_for_branch", lambda _branch: workspace)
    monkeypatch.setattr(
        developer.github,
        "candidate_snapshot",
        lambda _workspace: _snapshot(digest="changed"),
    )

    assert developer._validated_checkpoint() is None
    assert developer.checkpoints.exists() is False
    event = developer.audit.latest("selfdev_checkpoint_resume")
    assert event["status"] == "rejected"
    assert "files changed" in event["reason"]


def test_resource_pause_preserves_handoff_for_next_invocation(tmp_path: Path, monkeypatch):
    developer, workspace = _developer(tmp_path)
    cycle_id = _save_checkpoint(developer, workspace, monkeypatch)
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda interval=0.05: ResourceState(
            0,
            10,
            50,
            False,
            "user returned",
            False,
            True,
            "user returned",
            "",
        ),
    )

    with pytest.raises(CyclePaused):
        developer._check_resources(False, BRANCH)

    resumed = SelfDeveloper(developer.config, developer.root)
    monkeypatch.setattr(resumed.github, "worktree_for_branch", lambda _branch: workspace)
    monkeypatch.setattr(resumed.github, "candidate_snapshot", lambda _workspace: _snapshot())
    validated = resumed._validated_checkpoint()
    assert validated is not None
    assert validated[0].research_findings == (
        "The evolution loop lives in localpilot/selfdev.py.",
    )
    assert "resource gate clears" in validated[0].next_action

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        resumed,
        "_select_developer_model",
        lambda: DeveloperModelSelection("developer", 1, 1.0, "test"),
    )
    monkeypatch.setattr(resumed.github, "reviewer_modified_test_paths", lambda *_args, **_kwargs: set())

    def continue_candidate(**kwargs):
        captured.update(kwargs)
        return EvolutionResult("paused", BRANCH, workspace, "paused")

    monkeypatch.setattr(resumed, "_continue_candidate", continue_candidate)
    result = resumed._resume_checkpoint_candidate(validated, force=False)

    assert result.status == "paused"
    assert captured["cycle_id"] == cycle_id
    assert captured["checkpoint"].research_findings[0].startswith("The evolution loop")


def test_active_broker_turn_pauses_background_inference_even_when_pc_is_idle(
    tmp_path: Path,
    monkeypatch,
):
    developer, _workspace = _developer(tmp_path)
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda interval=0.05: ResourceState(
            3600,
            10,
            50,
            True,
            "idle capacity available",
            True,
            True,
            "",
            "",
        ),
    )
    assert write_foreground_turns(
        developer.data_dir,
        (
            {
                "request_id": "request-1",
                "session_id": "session-1",
                "message_id": "message-1",
            },
        ),
    )

    with pytest.raises(CyclePaused, match="active foreground chat turn"):
        developer._check_resources(
            False,
            "capability-discovery",
            during_inference=True,
        )

    assert "active foreground chat turn" in developer.audit.latest("selfdev_paused")["reason"]


def test_active_foreground_turn_defers_before_sync_or_resource_sampling(tmp_path, monkeypatch):
    developer, _workspace = _developer(tmp_path)
    assert write_foreground_turns(
        developer.data_dir,
        ({"request_id": "request-early", "session_id": "session-1", "message_id": "message-1"},),
    )
    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: pytest.fail("foreground deferral must happen before repository sync"),
    )
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: pytest.fail("foreground deferral must happen before resource sampling"),
    )

    result = developer.run_once()

    assert result.status == "deferred"
    assert "active foreground chat turn" in result.summary


def test_terminal_completion_cleans_up_active_checkpoint(tmp_path: Path, monkeypatch):
    developer, workspace = _developer(tmp_path)
    _save_checkpoint(developer, workspace, monkeypatch, milestone="delivery")
    monkeypatch.setattr(
        developer,
        "_run_once",
        lambda force=False: EvolutionResult("candidate_ready", BRANCH, workspace, "ready", True),
    )

    result = developer.run_once(force=True)

    assert result.status == "candidate_ready"
    assert developer.checkpoints.exists() is False
    cleared = developer.audit.latest("selfdev_checkpoint_cleared")
    assert "candidate_ready" in cleared["reason"]
