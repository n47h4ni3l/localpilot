from pathlib import Path

import pytest

from localpilot.github_integration import parse_reviewer_modified_test_paths
from localpilot.selfdev import CandidateTools


def test_reviewer_commit_protects_regression_test():
    log = """@@LOCALPILOT_COMMIT@@repair: Implement guarded operator
localpilot/operator.py
tests/test_operator.py

@@LOCALPILOT_COMMIT@@review: add timeout regression contract
tests/test_operator.py

@@LOCALPILOT_COMMIT@@candidate: Implement guarded operator
localpilot/operator.py
tests/test_operator.py
"""

    protected = parse_reviewer_modified_test_paths(log)

    assert protected == {"tests/test_operator.py"}


def test_autonomous_commits_do_not_protect_their_own_tests():
    log = """@@LOCALPILOT_COMMIT@@repair: Fix generated test
tests/test_generated.py

@@LOCALPILOT_COMMIT@@candidate: Initial implementation
tests/test_generated.py
"""

    assert parse_reviewer_modified_test_paths(log) == set()


def test_protected_test_can_be_read_but_not_written(tmp_path: Path):
    test_file = tmp_path / "tests" / "test_contract.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("assert expected is None\n", encoding="utf-8")

    production = tmp_path / "localpilot" / "operator.py"
    production.parent.mkdir(parents=True)
    production.write_text("value = -1\n", encoding="utf-8")

    tools = CandidateTools(
        tmp_path,
        protected_paths={"tests/test_contract.py"},
    )

    assert "expected is None" in tools.read_project_file(
        "tests/test_contract.py"
    )

    with pytest.raises(PermissionError, match="read-only"):
        tools.write_project_file(
            "tests/test_contract.py",
            "assert expected == -1\n",
        )

    tools.write_project_file(
        "localpilot/operator.py",
        "value = None\n",
    )

    assert production.read_text(encoding="utf-8") == "value = None\n"
