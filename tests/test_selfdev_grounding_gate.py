from __future__ import annotations

import json
from pathlib import Path

from localpilot.config import Config
from localpilot.evolution import normalize_evolution_task
from localpilot.selfdev import CandidateTools, SelfDeveloper, parse_grounding_plan
from localpilot.study import RepositoryGroundingValidator


def _repository(root: Path) -> None:
    (root / "localpilot").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "localpilot" / "config.py").write_text(
        "class AgentConfig:\n    timeout: int = 20\n",
        encoding="utf-8",
    )
    (root / "localpilot" / "sample.py").write_text(
        "def helper():\n    return 1\n\n"
        "class Runner:\n"
        "    def execute(self):\n"
        "        return helper()\n\n"
        "def run():\n"
        "    return Runner().execute()\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_sample.py").write_text(
        "def test_run():\n    assert True\n",
        encoding="utf-8",
    )


def _plan(**updates):
    plan = {
        "referenced_symbols": ["localpilot.sample:Runner.execute"],
        "referenced_config_fields": ["agent.timeout"],
        "referenced_paths": ["localpilot/sample.py", "localpilot/config.py"],
        "required_test_contracts": ["test_run"],
        "integration_points": ["localpilot.sample:Runner.execute"],
        "expected_call_relationships": [
            ["localpilot.sample:Runner.execute", "helper"]
        ],
        "planned_subsystems": [],
        "new_runtime_paths": ["localpilot/new_feature.py"],
    }
    plan.update(updates)
    return plan


def _task():
    return normalize_evolution_task(
        {
            "id": "grounding-task",
            "title": "Add one grounded feature",
            "acceptance": ["Add a focused implementation and test."],
            "source": "backlog",
        }
    )


def _developer(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    _repository(workspace)
    config = Config()
    config.agent.data_dir = "data"
    config.selfdev.run_static_checks = False
    config.github.auto_push_candidates = False
    developer = SelfDeveloper(config, project)
    task = _task()
    cycle_id = developer.memory.start_cycle(
        task_id=task["id"],
        branch="localpilot/candidate-grounding-task",
        everyday_model="daily",
        developer_model="developer",
        workspace=workspace,
        is_worktree=False,
    )
    tools = CandidateTools(workspace)
    return developer, workspace, task, cycle_id, tools


def test_live_grounding_uses_repository_not_durable_memory(tmp_path: Path):
    _repository(tmp_path)

    report = RepositoryGroundingValidator(root=tmp_path).validate(_plan())

    assert report.grounded
    assert "symbol:localpilot.sample:Runner.execute" in report.evidence
    assert "call:localpilot.sample:Runner.execute->helper" in report.evidence


def test_live_grounding_reports_false_claims_with_evidence(tmp_path: Path):
    _repository(tmp_path)

    report = RepositoryGroundingValidator(root=tmp_path).validate(
        _plan(
            referenced_symbols=["localpilot.sample:ImaginaryRunner"],
            referenced_config_fields=["agent.magic"],
            referenced_paths=["localpilot/missing.py"],
            required_test_contracts=["test_imaginary"],
            integration_points=["localpilot.sample:missing_hook"],
            expected_call_relationships=[["localpilot.sample:run", "missing"]],
        )
    )

    assert not report.grounded
    assert {issue.code for issue in report.issues} == {
        "nonexistent_api",
        "nonexistent_config_field",
        "missing_file_or_command",
        "missing_test_contract",
        "wrong_integration_point",
        "call_graph_mismatch",
    }


def test_grounding_plan_parser_requires_every_claim_class():
    raw = json.dumps({"change_plan": _plan()})
    assert parse_grounding_plan(raw) == _plan()

    incomplete = _plan()
    incomplete.pop("integration_points")
    try:
        parse_grounding_plan(json.dumps({"change_plan": incomplete}))
    except ValueError as exc:
        assert "integration_points" in str(exc)
    else:
        raise AssertionError("Incomplete grounding plans must fail closed.")


def test_real_candidate_pipeline_blocks_writes_when_grounding_fails(
    tmp_path: Path, monkeypatch
):
    developer, workspace, task, cycle_id, tools = _developer(tmp_path)
    stages = []
    milestones = []
    monkeypatch.setattr(developer, "_activate_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(
        developer,
        "_checkpoint_milestone",
        lambda milestone, **kwargs: milestones.append(milestone),
    )

    def fake_stage(*, stage, **kwargs):
        stages.append(stage)
        if stage == "research":
            return json.dumps(
                {
                    "findings": ["Runner.execute is the integration point."],
                    "decisions": ["Add one small module."],
                    "unresolved_questions": [],
                    "next_action": "Ground the plan.",
                }
            )
        if stage == "grounding":
            return json.dumps(
                {
                    "change_plan": _plan(
                        integration_points=["localpilot.sample:ImaginaryRunner"]
                    )
                }
            )
        raise AssertionError("Write-capable implementation stage must not be reached.")

    monkeypatch.setattr(developer, "_tool_stage", fake_stage)

    result = developer._continue_candidate(
        force=True,
        cycle_id=cycle_id,
        task=task,
        branch="localpilot/candidate-grounding-task",
        workspace=workspace,
        is_worktree=False,
        developer_model="developer",
        tools=tools,
    )

    assert result.status == "failed"
    assert "GroundingGateError" in result.summary
    assert stages == ["research", "grounding"]
    assert "implementation" not in milestones
    assert tools.files_written == set()
    assert developer.audit.latest("selfdev_grounding_gate")["status"] == "rejected"


def test_real_candidate_pipeline_writes_only_after_grounding_passes(
    tmp_path: Path, monkeypatch
):
    developer, workspace, task, cycle_id, tools = _developer(tmp_path)
    stages = []
    milestones = []
    monkeypatch.setattr(developer, "_activate_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(
        developer,
        "_checkpoint_milestone",
        lambda milestone, **kwargs: milestones.append(milestone),
    )

    def fake_stage(*, stage, messages, **kwargs):
        stages.append(stage)
        if stage == "research":
            return json.dumps(
                {
                    "findings": ["Runner.execute calls helper."],
                    "decisions": ["Add one small module."],
                    "unresolved_questions": [],
                    "next_action": "Ground the plan.",
                }
            )
        if stage == "grounding":
            return json.dumps({"change_plan": _plan()})
        assert stage == "implementation"
        assert "Verified repository change plan" in messages[0]["content"]
        tools.write_project_file("localpilot/new_feature.py", "VALUE = 1\n")
        return json.dumps(
            {
                "summary": "Added the grounded feature.",
                "reusable_lesson": "Gate writes on live repository claims.",
                "evaluation_evidence": {
                    "metric": "focused test result",
                    "baseline_evidence": "feature absent",
                    "candidate_evidence": "candidate file added",
                    "result": "pending_ci",
                    "measurement_artifact": "tests/test_sample.py",
                },
            }
        )

    monkeypatch.setattr(developer, "_tool_stage", fake_stage)

    result = developer._continue_candidate(
        force=True,
        cycle_id=cycle_id,
        task=task,
        branch="localpilot/candidate-grounding-task",
        workspace=workspace,
        is_worktree=False,
        developer_model="developer",
        tools=tools,
    )

    assert result.status == "candidate_needs_work"
    assert stages == ["research", "grounding", "implementation"]
    assert milestones.index("grounding_complete") < milestones.index("implementation")
    assert (workspace / "localpilot" / "new_feature.py").is_file()
    assert developer.audit.latest("selfdev_grounding_gate")["status"] == "passed"
