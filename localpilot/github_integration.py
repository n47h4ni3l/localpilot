from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from localpilot.config import GitHubConfig


@dataclass(slots=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


class GitHubIntegration:
    """Git/GitHub adapter. `git` is required; `gh` is optional."""

    def __init__(self, project_root: str | Path, config: GitHubConfig) -> None:
        self.root = Path(project_root).resolve()
        self.config = config

    def _run(self, args: list[str], cwd: Path | None = None, timeout: int = 30) -> CommandResult:
        completed = subprocess.run(
            args,
            cwd=str(cwd or self.root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip(), completed.returncode)

    def git_available(self) -> bool:
        return shutil.which("git") is not None

    def gh_available(self) -> bool:
        return shutil.which("gh") is not None

    def is_git_repo(self) -> bool:
        if not self.git_available():
            return False
        return self._run(["git", "rev-parse", "--is-inside-work-tree"]).ok

    def status(self) -> str:
        if not self.is_git_repo():
            return "Not connected to a local Git repository yet."
        branch = self._run(["git", "branch", "--show-current"]).stdout or "(detached)"
        changes = self._run(["git", "status", "--short"]).stdout
        remote = self._run(["git", "remote", "get-url", self.config.remote])
        remote_text = remote.stdout if remote.ok else "(no remote configured)"
        return f"branch: {branch}\nremote: {remote_text}\nchanges:\n{changes or '(clean)'}"

    def clean_worktree(self) -> bool:
        return self.is_git_repo() and self._run(["git", "status", "--porcelain"]).stdout == ""

    def create_candidate_worktree(self, branch: str, destination: Path) -> CommandResult:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return self._run(["git", "worktree", "add", "-b", branch, str(destination), "HEAD"])

    def commit_all(self, worktree: Path, message: str) -> CommandResult:
        add = self._run(["git", "add", "-A"], cwd=worktree)
        if not add.ok:
            return add
        return self._run(["git", "commit", "-m", message], cwd=worktree)

    def push_branch(self, worktree: Path, branch: str) -> CommandResult:
        return self._run(["git", "push", "-u", self.config.remote, branch], cwd=worktree, timeout=120)

    def create_issue(self, title: str, body: str) -> CommandResult:
        if not self.gh_available():
            return CommandResult(False, "", "GitHub CLI (gh) is not installed.", 127)
        return self._run(["gh", "issue", "create", "--title", title, "--body", body], timeout=60)
