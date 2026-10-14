from pathlib import Path
import shutil
import subprocess

import pytest

from localpilot.selfdev import CandidateTools
from localpilot.selfdev_frontend_baseline import approved_main_revision


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Baseline test")
    git(tmp_path, "config", "user.email", "baseline@example.invalid")
    path = tmp_path / "localpilot/webview/settings-updates.js"
    path.parent.mkdir(parents=True)
    path.write_text("window.pywebview.api.get_update_settings();\n", encoding="utf-8")
    (tmp_path / "feature.py").write_text("answer = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "approved main")
    sha = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", sha)
    return tmp_path, path, sha


def test_unchanged_approved_frontend_no_longer_blocks_python_candidate(repository):
    root, path, sha = repository
    assert approved_main_revision(root, remote="origin", branch="main") == sha
    git(root, "switch", "-c", "candidate")
    (root / "feature.py").write_text("answer = 2\n", encoding="utf-8")
    # The old/default standalone behaviour demonstrates the reported blocker.
    assert "disallowed direct native bridge" in CandidateTools(root).run_candidate_static_checks()
    tools = CandidateTools(root, trusted_main_revision=sha, existing_changed_paths=["feature.py"])
    result = tools.run_candidate_static_checks()
    assert result.startswith("static_checks=passed")
    assert "trusted_baseline_frontend_files=1" in result
    assert tools._frontend_baseline.revision == sha


@pytest.mark.parametrize("commit", [False, True])
def test_candidate_changes_are_never_their_own_trusted_baseline(repository, commit):
    root, path, sha = repository
    git(root, "switch", "-c", "candidate")
    path.write_text(path.read_text() + "eval('new candidate code');\n", encoding="utf-8")
    if commit:
        git(root, "add", ".")
        git(root, "commit", "-m", "candidate-authored commit")
    result = CandidateTools(root, trusted_main_revision=sha).run_candidate_static_checks()
    assert result.startswith("static_checks=failed") and "dynamic code execution" in result
    with pytest.raises(ValueError):
        CandidateTools(root, trusted_main_revision=sha, existing_changed_paths=[str(path.relative_to(root))])


def test_unknown_provenance_and_new_frontend_stay_strict(repository):
    root, path, sha = repository
    result = CandidateTools(root, trusted_main_revision="0" * 40).run_candidate_static_checks()
    assert "disallowed direct native bridge" in result
    new = path.with_name("unapproved.js")
    new.write_text("window.pywebview.api.unapproved();\n", encoding="utf-8")
    result = CandidateTools(root, trusted_main_revision=sha).run_candidate_static_checks()
    assert result.startswith("static_checks=failed") and "unapproved.js" in result


@pytest.mark.skipif(not shutil.which("node"), reason="JavaScript syntax parser unavailable")
def test_approved_exception_does_not_skip_javascript_syntax(repository):
    root, path, sha = repository
    path.write_text("window.pywebview.api.get_update_settings(); const = ;\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "invalid syntax fixture")
    sha = git(root, "rev-parse", "HEAD")
    result = CandidateTools(root, trusted_main_revision=sha).run_candidate_static_checks()
    assert "JavaScript syntax" in result


def test_approved_source_requires_clean_main_and_matching_upstream(repository):
    root, path, sha = repository
    path.write_text(path.read_text() + "// dirty\n", encoding="utf-8")
    assert approved_main_revision(root, remote="origin", branch="main") is None
    git(root, "add", ".")
    git(root, "commit", "-m", "unpublished main change")
    assert approved_main_revision(root, remote="origin", branch="main") is None
    git(root, "switch", "-c", "candidate")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert approved_main_revision(root, remote="origin", branch="main") is None


def test_git_replacement_cannot_substitute_approved_source(repository):
    root, path, sha = repository
    original = git(root, "rev-parse", f"{sha}:{path.relative_to(root).as_posix()}")
    path.write_text("eval('candidate code');\n", encoding="utf-8")
    substituted = git(root, "hash-object", "-w", str(path))
    git(root, "replace", original, substituted)
    result = CandidateTools(root, trusted_main_revision=sha).run_candidate_static_checks()
    assert result.startswith("static_checks=failed") and "dynamic code execution" in result
