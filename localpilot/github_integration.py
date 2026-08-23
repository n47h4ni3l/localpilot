from __future__ import annotations

import hashlib
import json
import re
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
class MainSyncResult:
    ok: bool
    updated: bool
    summary: str


@dataclass(frozen=True, slots=True)
class CandidateLifecycle:
    validation_state: str
    merged: bool
    pull_request_url: str | None = None
    pull_request_state: str = "unknown"
    remote_branch_exists: bool | None = None


@dataclass(frozen=True, slots=True)
class CandidatePullRequest:
    number: int
    branch: str
    url: str


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    branch: str
    head: str
    changed_paths: tuple[str, ...]
    state_digest: str


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
_SAFE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MANAGED_CANDIDATE_BRANCH = re.compile(
    r"^localpilot/candidate-[A-Za-z0-9][A-Za-z0-9._/-]*$"
)


def is_managed_candidate_branch(branch: str) -> bool:
    return bool(_MANAGED_CANDIDATE_BRANCH.fullmatch(str(branch)))


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

    def sync_trusted_main(self) -> MainSyncResult:
        """Fast-forward a clean, trusted main checkout without touching candidates."""
        remote = self.config.remote
        main_branch = self.config.main_branch
        if not _SAFE_REMOTE_NAME.fullmatch(remote):
            return MainSyncResult(False, False, "Configured Git remote name is unsafe.")
        if not _SAFE_BRANCH_NAME.fullmatch(main_branch):
            return MainSyncResult(False, False, "Configured main branch name is unsafe.")

        if not self.is_git_repo():
            return MainSyncResult(False, False, "Project root is not a Git checkout.")

        top_level = self._run(["git", "rev-parse", "--show-toplevel"])
        if not top_level.ok or Path(top_level.stdout).resolve() != self.root:
            return MainSyncResult(False, False, "Project root is not the Git checkout root.")

        branch = self._run(["git", "branch", "--show-current"])
        if not branch.ok or branch.stdout != main_branch:
            current = branch.stdout or "detached HEAD"
            return MainSyncResult(
                False,
                False,
                f"Refusing self-sync from {current}; trusted branch is {main_branch}.",
            )

        status = self._run(["git", "status", "--porcelain", "--untracked-files=all"])
        if not status.ok:
            return MainSyncResult(False, False, status.stderr or "Could not inspect the main checkout.")
        if status.stdout:
            return MainSyncResult(False, False, "Main checkout has uncommitted work; nothing was changed.")

        configured_remote = self._run(["git", "remote", "get-url", remote])
        if not configured_remote.ok:
            return MainSyncResult(False, False, f"Configured Git remote {remote} is unavailable.")

        remote_ref = f"refs/remotes/{remote}/{main_branch}"
        fetched = self._run(
            [
                "git",
                "fetch",
                "--no-tags",
                "--prune",
                remote,
                f"+refs/heads/{main_branch}:{remote_ref}",
            ],
            timeout=120,
        )
        if not fetched.ok:
            detail = fetched.stderr or fetched.stdout or "unknown fetch error"
            return MainSyncResult(False, False, f"Could not refresh trusted main: {detail}")

        head = self._run(["git", "rev-parse", "--verify", "HEAD^{commit}"])
        target = self._run(["git", "rev-parse", "--verify", f"{remote_ref}^{{commit}}"])
        if not head.ok or not target.ok:
            return MainSyncResult(False, False, "Could not resolve local and remote main commits.")
        if head.stdout == target.stdout:
            return MainSyncResult(True, False, f"Trusted {main_branch} is already current at {head.stdout}.")

        ancestor = self._run(["git", "merge-base", "--is-ancestor", head.stdout, target.stdout])
        if not ancestor.ok:
            return MainSyncResult(
                False,
                False,
                "Local main is ahead of or diverged from the remote; automatic sync was refused.",
            )

        still_clean = self._run(["git", "status", "--porcelain", "--untracked-files=all"])
        if not still_clean.ok or still_clean.stdout:
            return MainSyncResult(False, False, "Main checkout changed during sync; fast-forward was refused.")

        merged = self._run(["git", "merge", "--ff-only", "--no-edit", target.stdout], timeout=120)
        if not merged.ok:
            detail = merged.stderr or merged.stdout or "unknown fast-forward error"
            return MainSyncResult(False, False, f"Trusted main could not be fast-forwarded: {detail}")

        verified = self._run(["git", "rev-parse", "--verify", "HEAD^{commit}"])
        if not verified.ok or verified.stdout != target.stdout:
            return MainSyncResult(False, False, "Fast-forward verification failed; evolve was stopped.")
        return MainSyncResult(
            True,
            True,
            f"Trusted {main_branch} fast-forwarded from {head.stdout} to {target.stdout}.",
        )

    def create_candidate_worktree(self, branch: str, destination: Path) -> CommandResult:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return self._run(["git", "worktree", "add", "-b", branch, str(destination), "HEAD"])

    def create_policy_retry_worktree(
        self,
        branch: str,
        destination: Path,
    ) -> CommandResult:
        """Create a distinct retry branch only from clean, current trusted main."""
        if not is_managed_candidate_branch(branch):
            return CommandResult(False, "", "Retry branch is not a managed candidate.", 2)
        top_level = self._run(["git", "rev-parse", "--show-toplevel"])
        if not top_level.ok or Path(top_level.stdout).resolve() != self.root:
            return CommandResult(False, "", "Project root is not the Git checkout root.", 2)
        current = self._run(["git", "branch", "--show-current"])
        if not current.ok or current.stdout != self.config.main_branch:
            return CommandResult(
                False,
                "",
                f"Trusted checkout must be on {self.config.main_branch} before retry.",
                2,
            )
        clean = self._run(["git", "status", "--porcelain", "--untracked-files=all"])
        if not clean.ok or clean.stdout:
            return CommandResult(False, "", "Trusted main checkout must be clean before retry.", 2)
        head = self._run(["git", "rev-parse", "--verify", "HEAD^{commit}"])
        remote_main = self._run(
            [
                "git", "rev-parse", "--verify",
                f"refs/remotes/{self.config.remote}/{self.config.main_branch}^{{commit}}",
            ]
        )
        if not head.ok or not remote_main.ok or head.stdout != remote_main.stdout:
            return CommandResult(
                False,
                "",
                "Trusted main is not aligned with its fetched remote; pull with --ff-only first.",
                2,
            )
        if destination.exists():
            return CommandResult(False, "", "Retry worktree destination already exists.", 2)
        return self.create_candidate_worktree(branch, destination)

    def commit_paths(self, worktree: Path, message: str, relative_paths: list[str]) -> CommandResult:
        if not relative_paths:
            return CommandResult(False, "", "No candidate paths supplied.", 2)
        add = self._run(["git", "add", "--", *sorted(set(relative_paths))], cwd=worktree)
        if not add.ok:
            return add
        return self._run(["git", "commit", "-m", message], cwd=worktree)

    def push_branch(self, worktree: Path, branch: str) -> CommandResult:
        return self._run(["git", "push", "-u", self.config.remote, branch], cwd=worktree, timeout=120)

    def create_candidate_pull_request(
        self,
        worktree: Path,
        *,
        branch: str,
        title: str,
        body: str,
    ) -> CommandResult:
        """Present a candidate for review; this never merges or promotes it."""
        if not self.gh_available():
            return CommandResult(False, "", "GitHub CLI (gh) is not installed.", 127)
        if not _SAFE_BRANCH_NAME.fullmatch(branch):
            return CommandResult(False, "", "Candidate branch name is unsafe.", 2)
        return self._run(
            [
                "gh",
                "pr",
                "create",
                "--head",
                branch,
                "--base",
                self.config.main_branch,
                "--title",
                title[:240],
                "--body",
                body[:20_000],
            ],
            cwd=worktree,
            timeout=120,
        )

    def resolve_candidate_pull_request(self, pull_request_number: int) -> CandidatePullRequest:
        """Resolve an explicit PR number without changing GitHub state."""
        if isinstance(pull_request_number, bool) or int(pull_request_number) <= 0:
            raise ValueError("Pull request number must be a positive integer.")
        if not self.gh_available():
            raise RuntimeError("GitHub CLI (gh) is required to resolve a pull request.")
        result = self._run(
            [
                "gh",
                "pr",
                "view",
                str(int(pull_request_number)),
                "--json",
                "number,headRefName,url",
            ],
            timeout=60,
        )
        if not result.ok:
            detail = result.stderr or result.stdout or "GitHub returned no detail."
            raise RuntimeError(f"Could not resolve PR #{pull_request_number}: {detail}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GitHub returned invalid PR metadata.") from exc
        if not isinstance(value, dict):
            raise RuntimeError("GitHub returned invalid PR metadata.")
        number = value.get("number")
        branch = str(value.get("headRefName") or "")
        url = str(value.get("url") or "")
        if number != int(pull_request_number) or not url:
            raise RuntimeError("GitHub PR metadata did not match the requested PR.")
        if not is_managed_candidate_branch(branch):
            raise ValueError(
                f"PR #{pull_request_number} uses unrelated branch {branch or '(unknown)'}; "
                "only LocalPilot-managed candidate branches can be rejected."
            )
        return CandidatePullRequest(int(number), branch, url)

    def tracked_project_paths(self) -> set[str]:
        """Return only committed paths suitable for read-only discovery."""
        result = self._run(["git", "ls-files", "-z"])
        if not result.ok:
            return set()
        paths: set[str] = set()
        for raw in result.stdout.split("\0"):
            relative = Path(raw).as_posix()
            if relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts:
                paths.add(relative)
        return paths

    def candidate_changed_paths(self, worktree: Path) -> list[str]:
        """Return tracked and untracked candidate paths without executing code."""
        tracked = self._run(
            ["git", "diff", "--name-only", "-z", "HEAD", "--", "."],
            cwd=worktree,
        )
        untracked = self._run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=worktree,
        )
        if not tracked.ok or not untracked.ok:
            return []

        paths: set[str] = set()
        for value in (tracked.stdout, untracked.stdout):
            for raw in value.split("\0"):
                if not raw:
                    continue
                relative = Path(raw).as_posix()
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    continue
                paths.add(relative)
        return sorted(paths)

    def candidate_snapshot(self, worktree: Path) -> CandidateSnapshot:
        """Describe candidate identity and content without executing candidate code."""
        resolved = Path(worktree).resolve()
        top_level = self._run(["git", "rev-parse", "--show-toplevel"], cwd=resolved)
        branch = self._run(["git", "branch", "--show-current"], cwd=resolved)
        head = self._run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=resolved)
        if not top_level.ok or Path(top_level.stdout).resolve() != resolved:
            raise RuntimeError("Candidate workspace is not its Git worktree root.")
        if not branch.ok or not branch.stdout:
            raise RuntimeError("Candidate workspace has no attached branch.")
        if not head.ok or not head.stdout:
            raise RuntimeError("Candidate workspace HEAD could not be resolved.")

        changed_paths = self.candidate_changed_paths(resolved)
        digest = hashlib.sha256()
        digest.update(branch.stdout.encode("utf-8"))
        digest.update(b"\0")
        digest.update(head.stdout.encode("ascii", errors="replace"))
        for relative in changed_paths:
            digest.update(b"\0")
            digest.update(relative.encode("utf-8"))
            path = (resolved / relative).resolve()
            if path == resolved or resolved not in path.parents or not path.is_file():
                digest.update(b"<missing>")
                continue
            with path.open("rb") as handle:
                while chunk := handle.read(65536):
                    digest.update(chunk)

        return CandidateSnapshot(
            branch.stdout,
            head.stdout,
            tuple(changed_paths),
            digest.hexdigest(),
        )

    def branch_has_candidate_commit(self, worktree: Path) -> bool:
        result = self._run(
            [
                "git",
                "rev-list",
                "--count",
                f"{self.config.main_branch}..HEAD",
            ],
            cwd=worktree,
        )
        return result.ok and result.stdout.isdigit() and int(result.stdout) > 0

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

    def remote_candidate_branch_exists(self, branch: str) -> bool | None:
        """Return a definitive remote-branch answer, or None on query failure."""
        if not is_managed_candidate_branch(branch):
            return False
        result = self._run(
            [
                "git", "ls-remote", "--exit-code", "--heads",
                self.config.remote, f"refs/heads/{branch}",
            ],
            timeout=120,
        )
        if result.ok:
            return bool(result.stdout.strip())
        if result.returncode == 2:
            return False
        return None

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

    def reviewer_modified_test_paths(
        self,
        worktree: Path,
        *,
        refresh: bool = True,
    ) -> set[str]:
        """Return reviewer-controlled test paths unique to this candidate branch."""
        if refresh:
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

        base_ref = (
            f"{self.config.remote}/{self.config.main_branch}"
            if refresh
            else self.config.main_branch
        )

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

    def remove_candidate_worktree(
        self,
        branch: str,
        *,
        expected_workspace: str | Path | None = None,
    ) -> CommandResult:
        """Remove only a clean, registered candidate worktree; keep branch history."""
        if not is_managed_candidate_branch(branch):
            return CommandResult(False, "", "Branch is not a managed candidate.", 2)
        workspace = self.worktree_for_branch(branch)
        if workspace is None:
            return CommandResult(True, "No registered candidate worktree.", "", 0)
        if workspace == self.root:
            return CommandResult(False, "", "Refusing to remove the project root worktree.", 2)
        if expected_workspace is not None and workspace != Path(expected_workspace).resolve():
            return CommandResult(
                False,
                "",
                "Registered worktree does not match durable candidate memory.",
                2,
            )
        status = self._run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=workspace,
        )
        if not status.ok:
            return status
        if status.stdout:
            return CommandResult(
                False,
                "",
                "Candidate worktree has local changes; cleanup was skipped.",
                2,
            )
        removed = self._run(["git", "worktree", "remove", str(workspace)], timeout=60)
        if not removed.ok:
            return removed
        return CommandResult(
            True,
            "Removed clean local candidate worktree; branch and GitHub history were retained.",
            "",
            0,
        )

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
        if not is_managed_candidate_branch(branch):
            return CandidateLifecycle("pending", False, None, "unrelated", False)
        remote_branch_exists = self.remote_candidate_branch_exists(branch)
        if not self.gh_available():
            workflow = self.branch_workflow_state(branch)
            return CandidateLifecycle(
                workflow.validation_state,
                False,
                None,
                "unknown",
                remote_branch_exists,
            )

        result = self._run(
            [
                "gh", "pr", "list",
                "--head", branch,
                "--state", "all",
                "--limit", "20",
                "--json", "headRefName,url,state,mergedAt,statusCheckRollup",
            ],
            timeout=60,
        )

        if not result.ok:
            workflow = self.branch_workflow_state(branch)
            return CandidateLifecycle(
                workflow.validation_state, False, None, "unknown",
                remote_branch_exists,
            )

        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            workflow = self.branch_workflow_state(branch)
            return CandidateLifecycle(
                workflow.validation_state, False, None, "unknown",
                remote_branch_exists,
            )

        if not isinstance(rows, list):
            workflow = self.branch_workflow_state(branch)
            return CandidateLifecycle(
                workflow.validation_state, False, None, "unknown",
                remote_branch_exists,
            )

        matching = [
            row for row in rows
            if isinstance(row, dict) and str(row.get("headRefName") or "") == branch
        ]
        if not matching:
            workflow = self.branch_workflow_state(branch)
            return CandidateLifecycle(
                workflow.validation_state, False, None, "none",
                remote_branch_exists,
            )

        merged_pr = next((row for row in matching if row.get("mergedAt")), None)
        open_pr = next(
            (row for row in matching if str(row.get("state") or "").upper() == "OPEN"),
            None,
        )
        pr = merged_pr or open_pr or matching[0]
        checks = pr.get("statusCheckRollup") or []

        state = (
            classify_check_rollup(checks)
            if checks
            else self.branch_workflow_state(branch).validation_state
        )

        return CandidateLifecycle(
            state,
            merged_pr is not None,
            pr.get("url"),
            (
                "merged" if merged_pr is not None
                else "open" if open_pr is not None
                else "closed"
            ),
            remote_branch_exists,
        )

