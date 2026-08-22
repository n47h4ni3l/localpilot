"""Fail-closed regression coverage for the structurally incomplete PR #19."""

from pathlib import Path

import pytest

from localpilot.selfdev import (
    CandidateTools,
    ChangePlan,
    PlannedChange,
    _ALLOWED_SUFFIXES,
    apply_change_plan,
    candidate_write_integrity_failure,
)


def _tools(tmp_path: Path, **kwargs) -> CandidateTools:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    return CandidateTools(workspace, **kwargs)


def _plan(*changes: PlannedChange) -> ChangePlan:
    return ChangePlan(
        summary="candidate plan",
        reusable_lesson="validate before writing",
        changes=tuple(changes),
    )


def _change(path: str, content: str = "x") -> PlannedChange:
    return PlannedChange(path=path, content=content, reason="regression test")


def test_pr19_plan_is_rejected_before_precommit_config_is_written(tmp_path: Path):
    """Reproduce PR #19 with its real first file, not a protected .github path."""
    tools = _tools(tmp_path)
    plan = _plan(
        _change(
            ".pre-commit-config.yaml",
            "repos:\n  - repo: local\n    hooks:\n      - entry: scripts/pre_ci_review.sh\n",
        ),
        _change("scripts/pre_ci_review.sh", "#!/usr/bin/env bash\necho ok\n"),
    )

    with pytest.raises(ValueError, match=r"autonomous editing: \.sh"):
        apply_change_plan(plan, tools)

    assert not (tools.workspace / ".pre-commit-config.yaml").exists()
    assert not (tools.workspace / "scripts/pre_ci_review.sh").exists()
    assert tools.files_written == set()


@pytest.mark.parametrize(
    ("invalid_change", "protected_paths", "message"),
    [
        (_change("../escape.py"), (), "escapes candidate workspace"),
        (_change(".github/workflows/test.yml"), (), "Protected candidate path"),
        (_change("tests/reviewer.py"), ("tests/reviewer.py",), "read-only"),
        (_change("large.md", "x" * 1_000_001), (), "1 MB safety limit"),
    ],
)
def test_every_write_rule_is_preflighted_before_any_plan_write(
    tmp_path: Path,
    invalid_change: PlannedChange,
    protected_paths: tuple[str, ...],
    message: str,
):
    tools = _tools(tmp_path, protected_paths=protected_paths)
    plan = _plan(_change("README.md", "candidate\n"), invalid_change)

    with pytest.raises((PermissionError, ValueError), match=message):
        apply_change_plan(plan, tools)

    assert not (tools.workspace / "README.md").exists()
    assert tools.files_written == set()


def test_plan_preflight_accounts_for_existing_file_budget(tmp_path: Path):
    tools = _tools(tmp_path, max_files=2)
    tools.write_project_file("existing.py", "VALUE = 1\n")
    plan = _plan(
        _change("first.py", "VALUE = 2\n"),
        _change("second.py", "VALUE = 3\n"),
    )

    with pytest.raises(RuntimeError, match="file-write limit"):
        apply_change_plan(plan, tools)

    assert (tools.workspace / "existing.py").exists()
    assert not (tools.workspace / "first.py").exists()
    assert not (tools.workspace / "second.py").exists()


def test_rejected_direct_write_is_durable_for_the_cycle(tmp_path: Path):
    tools = _tools(tmp_path)
    tools.write_project_file(
        ".pre-commit-config.yaml",
        "repos:\n  - repo: local\n    hooks:\n      - entry: scripts/pre_ci_review.sh\n",
    )

    with pytest.raises(ValueError, match=r"autonomous editing: \.sh"):
        tools.write_project_file("scripts/pre_ci_review.sh", "echo ok\n")

    tools.write_project_file("scripts/pre_ci_review.ps1", "Write-Host 'ok'\n")
    failure = candidate_write_integrity_failure(tools)
    assert failure is not None
    assert "scripts/pre_ci_review.sh" in failure
    assert "Candidate delivery blocked" in failure


def test_static_checks_reject_missing_precommit_entry(tmp_path: Path):
    tools = _tools(tmp_path)
    tools.write_project_file(
        ".pre-commit-config.yaml",
        "repos:\n  - repo: local\n    hooks:\n      - entry: scripts/pre_ci_review.sh\n",
    )

    result = tools.run_candidate_static_checks()

    assert result.startswith("static_checks=failed")
    assert "hook entry references a missing file: scripts/pre_ci_review.sh" in result


def test_static_checks_accept_existing_human_owned_hook_script(tmp_path: Path):
    tools = _tools(tmp_path)
    script = tools.workspace / "scripts" / "pre_ci_review.sh"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    tools.write_project_file(
        ".pre-commit-config.yaml",
        "repos:\n  - repo: local\n    hooks:\n      - entry: scripts/pre_ci_review.sh\n",
    )

    assert tools.run_candidate_static_checks().startswith("static_checks=passed")


def test_allowed_suffixes_still_apply(tmp_path: Path):
    tools = _tools(tmp_path, max_files=len(_ALLOWED_SUFFIXES))
    changes = tuple(
        _change(f"generated/file{suffix}")
        for suffix in sorted(_ALLOWED_SUFFIXES - {".gitignore"})
    ) + (_change("generated/.gitignore"),)

    apply_change_plan(_plan(*changes), tools)

    assert len(tools.files_written) == len(_ALLOWED_SUFFIXES)
    assert candidate_write_integrity_failure(tools) is None
