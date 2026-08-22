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


def classify_workflow_run(status: str | None, conclusion: str | None) -> str:
    status_name = str(status or "").upper()
    conclusion_name = str(conclusion or "").upper()

    if status_name != "COMPLETED":
        return "pending"

    if conclusion_name in {"SUCCESS", "SKIPPED", "NEUTRAL"}:
        return "passed"

    if conclusion_name in {
        "FAILURE",
        "CANCELLED",
        "TIMED_OUT",
        "ACTION_REQUIRED",
        "STARTUP_FAILURE",
    }:
        return "failed"

    return "pending"


_AUTONOMOUS_COMMIT_PREFIXES = ("candidate: ", "repair: ")
_REVIEW_COMMIT_MARKER = "@@LOCALPILOT_COMMIT@@"


def parse_reviewer_modified_test_paths(log_text: str) -> set[str]:
    """Find tests changed by non-LocalPilot commits on a candidate branch."""
    protected: set[str] = set()
    reviewer_commit = False

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        if line.startswith(_REVIEW_COMMIT_MARKER):
            subject = line[len(_REVIEW_COMMIT_MARKER):].strip()
            reviewer_commit = not subject.startswith(_AUTONOMOUS_COMMIT_PREFIXES)
            continue

        if not reviewer_commit or not line:
            continue

        relative = Path(line).as_posix()

        if relative.startswith("tests/") and relative.endswith(".py"):
            protected.add(relative)

    return protected


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

    def branch_workflow_state(self, branch: str) -> CandidateLifecycle:
        """Observe the newest GitHub Actions run for a candidate branch."""
        if not self.gh_available():
            return CandidateLifecycle("awaiting_pr", False)

        result = self._run(
            [
                "gh", "run", "list",
                "--branch", branch,
                "--workflow", "tests",
                "--limit", "1",
                "--json", "databaseId,status,conclusion,url",
            ],
            timeout=60,
        )

        if not result.ok:
            return CandidateLifecycle("pending", False)

        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return CandidateLifecycle("pending", False)

        if not rows:
            return CandidateLifecycle("pending", False)

        row = rows[0]
        return CandidateLifecycle(
            classify_workflow_run(row.get("status"), row.get("conclusion")),
            False,
        )

    def failed_workflow_log(self, branch: str, max_chars: int = 16000) -> str:
        """Return a bounded failed-step log for the newest failed branch run."""
        if not self.gh_available():
            return "GitHub CLI is unavailable; CI failure details could not be retrieved."

        result = self._run(
            [
                "gh", "run", "list",
                "--branch", branch,
                "--workflow", "tests",
                "--limit", "1",
                "--json", "databaseId,status,conclusion",
            ],
            timeout=60,
        )

        if not result.ok:
            return result.stderr or result.stdout or "Could not query GitHub Actions."

        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return "GitHub Actions returned invalid JSON."

        if not rows:
            return "No GitHub Actions run was found for this candidate branch."

        row = rows[0]

        if classify_workflow_run(row.get("status"), row.get("conclusion")) != "failed":
            return "The newest GitHub Actions run is not failed."

        run_id = str(row.get("databaseId") or "")
        if not run_id:
            return "Failed workflow run did not include a run id."

        logs = self._run(
            ["gh", "run", "view", run_id, "--log-failed"],
            timeout=120,
        )

        log_text = logs.stdout if logs.ok else (logs.stderr or logs.stdout)
        limit = max(1000, min(int(max_chars), 50000))
        return (log_text or "No failed-step log was returned.")[-limit:]

    def reviewer_modified_test_paths(self, worktree: Path) -> set[str]:
        """Return reviewer-controlled test paths unique to this candidate branch."""
        fetched = self._run(
            ["git", "fetch", self.config.remote, "--prune"],
            cwd=worktree,
            timeout=120,
        )

        if not fetched.ok:
            raise RuntimeError(
                "Could not refresh Git history before determining protected "
                f"review tests: {fetched.stderr or fetched.stdout}"
            )

        base_ref = f"{self.config.remote}/{self.config.main_branch}"

        result = self._run(
            [
                "git",
                "log",
                f"{base_ref}..HEAD",
                f"--format={_REVIEW_COMMIT_MARKER}%s",
                "--name-only",
                "--no-renames",
            ],
            cwd=worktree,
            timeout=60,
        )

        if not result.ok:
            raise RuntimeError(
                "Could not inspect candidate history for reviewer tests: "
                f"{result.stderr or result.stdout}"
            )

        return parse_reviewer_modified_test_paths(result.stdout)

    def worktree_for_branch(self, branch: str) -> Path | None:
        result = self._run(["git", "worktree", "list", "--porcelain"])
        if not result.ok:
            return None

        current: Path | None = None
        wanted = f"branch refs/heads/{branch}"

        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current = Path(line[len("worktree "):])
            elif line == wanted and current is not None and current.exists():
                return current.resolve()

        return None

    def checkout_existing_branch_worktree(
        self,
        branch: str,
        destination: Path,
    ) -> CommandResult:
        """Restore an existing candidate branch without creating a new branch."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(["git", "worktree", "prune"])

        verify = self._run(
            ["git", "show-ref", "--verify", f"refs/heads/{branch}"]
        )

        if not verify.ok:
            fetched = self._run(
                ["git", "fetch", self.config.remote, f"{branch}:{branch}"],
                timeout=120,
            )
            if not fetched.ok:
                return fetched

        return self._run(
            ["git", "worktree", "add", str(destination), branch]
        )

    def candidate_lifecycle(self, branch: str) -> CandidateLifecycle:
        """Observe PR/check state. This method never merges or promotes."""
        if not self.gh_available():
            return CandidateLifecycle("awaiting_pr", False)

        result = self._run(
            [
                "gh", "pr", "list",
                "--head", branch,
                "--state", "all",
                "--limit", "1",
                "--json", "url,mergedAt,statusCheckRollup",
            ],
            timeout=60,
        )

        if not result.ok:
            return self.branch_workflow_state(branch)

        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self.branch_workflow_state(branch)

        if not rows:
            return self.branch_workflow_state(branch)

        pr = rows[0]
        checks = pr.get("statusCheckRollup") or []

        state = (
            classify_check_rollup(checks)
            if checks
            else self.branch_workflow_state(branch).validation_state
        )

        return CandidateLifecycle(
            state,
            bool(pr.get("mergedAt")),
            pr.get("url"),
        )

