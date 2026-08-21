from pathlib import Path

import pytest

from localpilot.selfdev import CandidateTools


def test_candidate_cannot_escape_workspace(tmp_path: Path):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError):
        tools.write_project_file("../escape.py", "x = 1")


def test_candidate_cannot_write_git_metadata(tmp_path: Path):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError):
        tools.write_project_file(".git/config", "bad")


def test_candidate_write_limit(tmp_path: Path):
    tools = CandidateTools(tmp_path, max_files=1)
    tools.write_project_file("one.py", "x = 1")
    with pytest.raises(RuntimeError):
        tools.write_project_file("two.py", "x = 2")
