from pathlib import Path

import pytest

from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.repository import RepositoryReader


def test_repository_reader_lists_reads_and_searches_verified_source(tmp_path: Path):
    (tmp_path / "localpilot").mkdir()
    (tmp_path / "localpilot" / "agent.py").write_text(
        "class LocalPilotAgent:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("LocalPilot repository\n", encoding="utf-8")
    (tmp_path / "localpilot-data").mkdir()
    (tmp_path / "localpilot-data" / "private.txt").write_text("do not expose", encoding="utf-8")

    reader = RepositoryReader(tmp_path)

    tree = reader.list_repository_tree(depth=3)
    assert "localpilot/" in tree
    assert "agent.py" in tree
    assert "README.md" in tree
    assert "localpilot-data" not in tree

    content = reader.read_repository_file("localpilot/agent.py", 1, 20)
    assert "1: class LocalPilotAgent:" in content

    results = reader.search_repository("LocalPilotAgent")
    assert "localpilot/agent.py:1" in results
    assert "private.txt" not in results


def test_repository_reader_rejects_traversal_and_sensitive_files(tmp_path: Path):
    (tmp_path / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "signing.key").write_text("secret\n", encoding="utf-8")
    reader = RepositoryReader(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        reader.read_repository_file(str((tmp_path / "safe.py").resolve()))
    with pytest.raises(ValueError, match="escapes"):
        reader.read_repository_file("../outside.txt")
    with pytest.raises(ValueError, match="excluded"):
        reader.read_repository_file(".env")
    with pytest.raises(ValueError, match="excluded"):
        reader.read_repository_file("signing.key")

    tree = reader.list_repository_tree(depth=2)
    assert ".env" not in tree
    assert "signing.key" not in tree


def test_repository_reader_does_not_follow_escape_symlink(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside secret\n", encoding="utf-8")
    link = tmp_path / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform/account.")

    reader = RepositoryReader(tmp_path)
    tree = reader.list_repository_tree(depth=2)
    assert "outside-link.txt -> [symlink not followed]" in tree
    with pytest.raises(ValueError, match="escapes"):
        reader.read_repository_file("outside-link.txt")


def test_repository_reader_bounds_file_output(tmp_path: Path):
    lines = "".join(f"line {index}\n" for index in range(1, 700))
    (tmp_path / "large.txt").write_text(lines, encoding="utf-8")
    reader = RepositoryReader(tmp_path)

    result = reader.read_repository_file("large.txt", 1, 699)
    numbered = [line for line in result.splitlines() if line[:1].isdigit()]
    assert len(numbered) <= 300
    assert "301: line 301" not in result


def test_repository_dependency_inspection_uses_declared_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
requires-python = ">=3.11"
dependencies = ["ollama>=0.6", "rich>=13"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
localpilot = "localpilot.cli:main"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    reader = RepositoryReader(tmp_path)

    result = reader.inspect_project_dependencies()
    assert '"requires_python": ">=3.11"' in result
    assert '"ollama>=0.6"' in result
    assert '"pytest>=8"' in result
    assert '"localpilot": "localpilot.cli:main"' in result


def test_operator_registry_exposes_repository_senses_as_read_only(tmp_path: Path):
    tools = registry(tmp_path)
    expected = {
        "list_repository_tree",
        "read_repository_file",
        "search_repository",
        "inspect_project_dependencies",
        "get_repository_status",
    }
    assert expected <= tools.keys()
    assert all(tools[name].risk is RiskLevel.READ_ONLY for name in expected)

    legacy = registry()
    assert expected.isdisjoint(legacy.keys())
