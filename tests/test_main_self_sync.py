import subprocess
from pathlib import Path

from localpilot.config import Config, GitHubConfig
from localpilot.github_integration import CommandResult, GitHubIntegration, MainSyncResult
from localpilot.selfdev import SelfDeveloper


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> CommandResult:
    return CommandResult(returncode == 0, stdout, stderr, returncode)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _scripted_git(monkeypatch, integration: GitHubIntegration, steps):
    calls: list[tuple[list[str], Path | None, int]] = []

    def fake_run(args, cwd=None, timeout=30):
        calls.append((args, cwd, timeout))
        expected, response = steps.pop(0)
        assert args == expected
        return response

    monkeypatch.setattr(integration, "git_available", lambda: True)
    monkeypatch.setattr(integration, "_run", fake_run)
    return calls


def test_trusted_main_fast_forwards_with_argv_only(tmp_path: Path, monkeypatch):
    integration = GitHubIntegration(tmp_path, GitHubConfig())
    old = "1" * 40
    new = "2" * 40
    remote_ref = "refs/remotes/origin/main"
    steps = [
        (["git", "rev-parse", "--is-inside-work-tree"], _result("true")),
        (["git", "rev-parse", "--show-toplevel"], _result(str(tmp_path.resolve()))),
        (["git", "branch", "--show-current"], _result("main")),
        (["git", "status", "--porcelain", "--untracked-files=all"], _result()),
        (["git", "remote", "get-url", "origin"], _result("https://example.invalid/repo.git")),
        (
            [
                "git",
                "fetch",
                "--no-tags",
                "--prune",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ],
            _result(),
        ),
        (["git", "rev-parse", "--verify", "HEAD^{commit}"], _result(old)),
        (["git", "rev-parse", "--verify", f"{remote_ref}^{{commit}}"], _result(new)),
        (["git", "merge-base", "--is-ancestor", old, new], _result()),
        (["git", "status", "--porcelain", "--untracked-files=all"], _result()),
        (["git", "merge", "--ff-only", "--no-edit", new], _result()),
        (["git", "rev-parse", "--verify", "HEAD^{commit}"], _result(new)),
    ]
    calls = _scripted_git(monkeypatch, integration, steps)

    result = integration.sync_trusted_main()

    assert result.ok is True
    assert result.updated is True
    assert old in result.summary and new in result.summary
    assert steps == []
    assert all(isinstance(call[0], list) for call in calls)


def test_trusted_main_fast_forwards_a_real_checkout(tmp_path: Path):
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    remote.mkdir()
    source.mkdir()
    _git(remote, "init", "--bare")
    _git(source, "init")
    _git(source, "config", "user.name", "LocalPilot Test")
    _git(source, "config", "user.email", "localpilot@example.invalid")
    _git(source, "branch", "-M", "main")
    (source / "version.txt").write_text("one\n", encoding="utf-8")
    _git(source, "add", "version.txt")
    _git(source, "commit", "-m", "initial")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(checkout))
    old = _git(checkout, "rev-parse", "HEAD")

    (source / "version.txt").write_text("two\n", encoding="utf-8")
    _git(source, "add", "version.txt")
    _git(source, "commit", "-m", "update")
    _git(source, "push", "origin", "main")
    new = _git(source, "rev-parse", "HEAD")

    result = GitHubIntegration(checkout, GitHubConfig()).sync_trusted_main()

    assert result.ok is True
    assert result.updated is True
    assert _git(checkout, "rev-parse", "HEAD") == new
    assert old != new
    assert (checkout / "version.txt").read_text(encoding="utf-8") == "two\n"


def test_dirty_main_is_preserved_without_fetch(tmp_path: Path, monkeypatch):
    integration = GitHubIntegration(tmp_path, GitHubConfig())
    steps = [
        (["git", "rev-parse", "--is-inside-work-tree"], _result("true")),
        (["git", "rev-parse", "--show-toplevel"], _result(str(tmp_path.resolve()))),
        (["git", "branch", "--show-current"], _result("main")),
        (
            ["git", "status", "--porcelain", "--untracked-files=all"],
            _result(" M localpilot/selfdev.py"),
        ),
    ]
    calls = _scripted_git(monkeypatch, integration, steps)

    result = integration.sync_trusted_main()

    assert result.ok is False
    assert result.updated is False
    assert "uncommitted work" in result.summary
    assert steps == []
    assert not any(call[0][0:2] == ["git", "fetch"] for call in calls)


def test_candidate_branch_is_never_self_synced(tmp_path: Path, monkeypatch):
    integration = GitHubIntegration(tmp_path, GitHubConfig())
    steps = [
        (["git", "rev-parse", "--is-inside-work-tree"], _result("true")),
        (["git", "rev-parse", "--show-toplevel"], _result(str(tmp_path.resolve()))),
        (["git", "branch", "--show-current"], _result("localpilot/candidate-change")),
    ]
    calls = _scripted_git(monkeypatch, integration, steps)

    result = integration.sync_trusted_main()

    assert result.ok is False
    assert "Refusing self-sync" in result.summary
    assert steps == []
    assert not any(call[0][0:2] == ["git", "fetch"] for call in calls)


def test_divergent_main_is_not_merged(tmp_path: Path, monkeypatch):
    integration = GitHubIntegration(tmp_path, GitHubConfig())
    local = "3" * 40
    remote = "4" * 40
    steps = [
        (["git", "rev-parse", "--is-inside-work-tree"], _result("true")),
        (["git", "rev-parse", "--show-toplevel"], _result(str(tmp_path.resolve()))),
        (["git", "branch", "--show-current"], _result("main")),
        (["git", "status", "--porcelain", "--untracked-files=all"], _result()),
        (["git", "remote", "get-url", "origin"], _result("https://example.invalid/repo.git")),
        (
            [
                "git",
                "fetch",
                "--no-tags",
                "--prune",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ],
            _result(),
        ),
        (["git", "rev-parse", "--verify", "HEAD^{commit}"], _result(local)),
        (
            ["git", "rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
            _result(remote),
        ),
        (["git", "merge-base", "--is-ancestor", local, remote], _result(returncode=1)),
    ]
    calls = _scripted_git(monkeypatch, integration, steps)

    result = integration.sync_trusted_main()

    assert result.ok is False
    assert "ahead of or diverged" in result.summary
    assert steps == []
    assert not any(call[0][0:2] == ["git", "merge"] for call in calls)


def test_updated_build_stops_before_resource_or_candidate_work(tmp_path: Path, monkeypatch):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: MainSyncResult(True, True, "Trusted main was updated."),
    )
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: (_ for _ in ()).throw(AssertionError("resource gate should not run")),
    )

    result = developer.run_once()

    assert result.status == "updated"
    assert "next invocation" in result.summary


def test_failed_sync_stops_before_resource_or_candidate_work(tmp_path: Path, monkeypatch):
    config = Config()
    config.agent.data_dir = "data"
    developer = SelfDeveloper(config, tmp_path)
    monkeypatch.setattr(
        developer.github,
        "sync_trusted_main",
        lambda: MainSyncResult(False, False, "Main checkout is dirty."),
    )
    monkeypatch.setattr(
        developer.governor,
        "sample",
        lambda: (_ for _ in ()).throw(AssertionError("resource gate should not run")),
    )

    result = developer.run_once()

    assert result.status == "sync_blocked"
    assert result.summary == "Main checkout is dirty."
