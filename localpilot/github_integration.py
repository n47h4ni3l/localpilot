from __future__ import annotations

import json
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


@dataclass(frozen=True, slots=True)
class CandidateLifecycle:
    validation_state: str
    merged: bool
    pull_request_url: str | None = None


def classify_check_rollup(checks: list[dict]) -> str:
    """Return a conservative aggregate for GitHub statusCheckRollup."""
    if not checks:
        return "pending"
    conclusions = {str(item.get("conclusion") or "").upper() for item in checks}
    statuses = {str(item.get("status") or "").upper() for item in checks}
    if conclusions & {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}:
        return "failed"
    if statuses & {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED"}:
        return "pending"
    terminal_ok = {"SUCCESS", "SKIPPED", "NEUTRAL"}
    return "passed" if conclusions and conclusions <= terminal_ok else "pending"


class GitHubIntegration:
    """Git/GitHub adapter. Every process uses argv and shell=False."""

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
            shell=False,
        )
        return CommandResult(completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip(), completed.returncode)

    def git_available(self) -> bool:
        return shutil.which("git") is not None

    def gh_available(self) -> bool:
        return shutil.which("gh") is not None

    def is_git_repo(self) -> bool:
        return self.git_available() and self._run(["git", "rev-parse", "--is-inside-work-tree"]).ok

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

    def commit_paths(self, worktree: Path, message: str, relative_paths: list[str]) -> CommandResult:
        if not relative_paths:
            return CommandResult(False, "", "No candidate paths supplied.", 2)
        add = self._run(["git", "add", "--", *sorted(set(relative_paths))], cwd=worktree)
        if not add.ok:
            return add
        return self._run(["git", "commit", "-m", message], cwd=worktree)

    def push_branch(self, worktree: Path, branch: str) -> CommandResult:
        return self._run(["git", "push", "-u", self.config.remote, branch], cwd=worktree, timeout=120)

    def create_issue(self, title: str, body: str) -> CommandResult:
        if not self.gh_available():
            return CommandResult(False, "", "GitHub CLI (gh) is not installed.", 127)
        return self._run(["gh", "issue", "create", "--title", title, "--body", body], timeout=60)

    def candidate_lifecycle(self, branch: str) -> CandidateLifecycle:
        """Observe PR/check state. This method never merges or promotes."""
        if not self.gh_available():
            return CandidateLifecycle("awaiting_pr", False)
        result = self._run(
            [
                "gh", "pr", "list", "--head", branch, "--state", "all", "--limit", "1",
                "--json", "url,mergedAt,statusCheckRollup",
            ],
            timeout=60,
        )
        if not result.ok:
            return CandidateLifecycle("awaiting_pr", False)
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return CandidateLifecycle("awaiting_pr", False)
        if not rows:
            return CandidateLifecycle("awaiting_pr", False)
        pr = rows[0]
        return CandidateLifecycle(
            classify_check_rollup(pr.get("statusCheckRollup") or []),
            bool(pr.get("mergedAt")),
            pr.get("url"),
        )

