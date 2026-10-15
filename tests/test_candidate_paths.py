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


def test_python_syntax_guard_preserves_previous_valid_file(tmp_path: Path):
    original = "from __future__ import annotations\nVALUE = 3\n"
    path = tmp_path / "module.py"
    path.write_text(original, encoding="utf-8")
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError, match="Invalid Python candidate edit.*line 2"):
        tools.write_project_file("module.py", "class Broken:\n        okay: bool\n  invalid: bool\n")
    assert path.read_text(encoding="utf-8") == original
    assert not tools.files_written
    assert tools.write_count == 0
    assert tools.failed_write_attempts
    assert "module.py" in tools.failed_write_attempts[0]

    # A correct replacement for an already broken source is allowed.
    path.write_text("class Broken:\n  invalid: bool\n        wrong: bool\n", encoding="utf-8")
    tools.write_project_file("module.py", "class Fixed:\n    okay: bool\n")
    assert path.read_text(encoding="utf-8") == "class Fixed:\n    okay: bool\n"
