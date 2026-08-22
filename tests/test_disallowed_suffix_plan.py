"""
Regression test for the PR #19 failure mode.

PR #19 ("self-review-repair") was rejected because the candidate referenced
scripts/pre_ci_review.sh, which write_project_file structurally cannot
create (.sh is not in _ALLOWED_SUFFIXES). Two gaps let that happen silently:

1. Developer-stage prompts never told the model which file types were
   actually writable, so it could plan a .sh file with no way to know that
   plan was doomed before trying it.
2. apply_change_plan() wrote each planned file one at a time with no
   upfront validation, so a plan with N changes could partially apply
   (e.g. write a workflow file that references a script) and then raise
   on the disallowed file, leaving a dangling reference on disk.

This test locks in the fix for (2): a change plan containing any
disallowed suffix must be rejected in full, before any file is written,
so the workspace never ends up in that half-applied state again.

Place this file under tests/ alongside the existing suite.
"""

import pytest

from localpilot.selfdev import (
    CandidateTools,
    ChangePlan,
    PlannedChange,
    _ALLOWED_SUFFIXES,
    apply_change_plan,
)


def _tools(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    return CandidateTools(workspace, max_files=8)


def test_plan_with_disallowed_suffix_is_rejected_before_any_write(tmp_path):
    tools = _tools(tmp_path)
    plan = ChangePlan(
        summary="Add a pre-CI review hook",
        reusable_lesson="",
        changes=(
            # This file is allowed and would normally be written first.
            PlannedChange(
                path=".github/workflows/pre_ci_review.yml",
                content="on: [pull_request]\n",
                reason="Wire the hook into CI",
            ),
            # This one is not — .sh is outside _ALLOWED_SUFFIXES.
            PlannedChange(
                path="scripts/pre_ci_review.sh",
                content="#!/usr/bin/env bash\necho ok\n",
                reason="The actual review script",
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"disallowed file type"):
        apply_change_plan(plan, tools)

    # The critical assertion: nothing from the plan should have landed on
    # disk. Before this fix, the .yml file above would have been written
    # successfully and only the .sh write would have raised, leaving a
    # workflow file that references a script that was never created —
    # exactly the structurally-incomplete candidate PR #19 shipped.
    assert not (tools.workspace / ".github/workflows/pre_ci_review.yml").exists()
    assert not (tools.workspace / "scripts/pre_ci_review.sh").exists()
    assert tools.files_written == set()


def test_plan_within_allowed_suffixes_still_applies_normally(tmp_path):
    tools = _tools(tmp_path)
    plan = ChangePlan(
        summary="Add a pre-CI review hook using an allowed script type",
        reusable_lesson="",
        changes=(
            PlannedChange(
                path="scripts/pre_ci_review.ps1",
                content="Write-Host 'ok'\n",
                reason="Windows-native equivalent of the .sh attempt",
            ),
        ),
    )

    results = apply_change_plan(plan, tools)

    assert len(results) == 1
    assert (tools.workspace / "scripts/pre_ci_review.ps1").exists()


@pytest.mark.parametrize("suffix", sorted(_ALLOWED_SUFFIXES - {".gitignore"}))
def test_every_currently_allowed_suffix_still_applies(tmp_path, suffix):
    """Guards against future edits to _ALLOWED_SUFFIXES silently breaking
    apply_change_plan's validation logic for suffixes it's supposed to
    permit."""
    tools = _tools(tmp_path)
    plan = ChangePlan(
        summary="s",
        reusable_lesson="",
        changes=(PlannedChange(path=f"generated/file{suffix}", content="x", reason="r"),),
    )
    apply_change_plan(plan, tools)
    assert (tools.workspace / f"generated/file{suffix}").exists()
