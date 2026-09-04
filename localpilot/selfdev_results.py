"""Candidate-lifecycle result/error types and small stateless helpers for
the self-development pipeline.

Extracted verbatim (no logic changes) from selfdev.py: the dataclasses
and exceptions a self-dev cycle returns or raises (EvolutionResult,
CandidateRejectionResult/Error, CandidateRetryResult/Error,
GroundingGateError, CyclePaused), the write-budget constants
(_IGNORE_NAMES, _ALLOWED_SUFFIXES, _ALLOWED_SUFFIXES_NOTE), and a few
small helpers (classify_candidate_result, choose_next_task,
candidate_write_integrity_failure, _checkpoint_paths). Re-imported as
bare names into selfdev.py wherever still referenced there -- which, for
the result/error types especially, is throughout every candidate-lifecycle
method including the three large ones this pass doesn't touch.

candidate_write_integrity_failure and _checkpoint_paths take a
CandidateTools instance as a parameter for its type hint only, never at
runtime (from __future__ import annotations defers all annotations) --
CandidateTools stays in selfdev.py, so that import is guarded under
TYPE_CHECKING to avoid a circular import while keeping the hint precise
for static analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from localpilot.selfdev import CandidateTools

_IGNORE_NAMES = {".git", ".github", ".venv", "__pycache__", ".pytest_cache", "localpilot-data"}

_ALLOWED_SUFFIXES = {
    ".py", ".toml", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv",
    ".yml", ".yaml", ".ps1", ".gitignore", ".zip", ".html", ".css", ".js",
}

# Derived from _ALLOWED_SUFFIXES so prompt guidance can never drift out of
# sync with what CandidateTools.write_project_file actually enforces.
_ALLOWED_SUFFIXES_NOTE = (
    "Directories may be created freely inside the isolated candidate with "
    "create_project_directory; directories do not consume the file budget. "
    "Allowed file types for autonomous writes: "
    f"{', '.join(sorted(_ALLOWED_SUFFIXES))}. write_project_file rejects any "
    "other extension, including .sh — this project is Windows-first, so use "
    ".ps1 for scripts, not .sh. HTML, CSS, and JavaScript writes are confined "
    "to localpilot/webview and must pass local-resource, strict-CSP, DOM-sink, "
    "storage, navigation, and native-bridge validation. Any rejected write attempt blocks candidate "
    "delivery for the current cycle. Use create_zip for bounded, inert archives "
    "and download_candidate_resource for provenance-tracked HTTPS research/data. "
    "Resources are stored outside the repository, never executed, and remain "
    "subject to resource-governor and quota checks."
)


@dataclass(slots=True)
class EvolutionResult:
    status: str
    branch: str | None
    workspace: Path | None
    summary: str
    tests_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class CandidateRejectionResult:
    pull_request_number: int
    branch: str
    task_id: str
    reason: str
    already_rejected: bool
    checkpoint_cleared: bool
    worktree_cleanup: str


class CandidateRejectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateRetryResult:
    prior_cycle_id: int
    retry_cycle_id: int
    prior_branch: str
    branch: str
    task_id: str
    reason: str
    already_authorized: bool
    resume_mode: str


class CandidateRetryError(RuntimeError):
    pass


class GroundingGateError(RuntimeError):
    pass


class CyclePaused(RuntimeError):
    pass


def classify_candidate_result(files_written: int, checks_passed: bool | None) -> str:
    if files_written == 0:
        return "no_changes"
    return "candidate_ready" if checks_passed else "candidate_needs_work"


def choose_next_task(
    tasks: Iterable[dict[str, Any]],
    completed_task_ids: set[str],
    pending_task_ids: set[str] | None = None,
    rejected_task_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Select the next unfinished task without retrying terminal rejections."""
    pending = pending_task_ids or set()
    rejected = rejected_task_ids or set()
    for task in tasks:
        if task.get("status", "todo") != "todo":
            continue
        task_id = str(task.get("id"))
        if task_id in completed_task_ids or task_id in rejected:
            continue
        if task_id in pending:
            return None
        return task
    return None


def candidate_write_integrity_failure(tools: CandidateTools) -> str | None:
    """Return the fail-closed delivery reason for rejected write attempts."""
    if not tools.failed_write_attempts:
        return None
    attempts = "; ".join(tools.failed_write_attempts[:10])
    return (
        "Candidate delivery blocked because one or more autonomous write attempts "
        f"were rejected during this cycle: {attempts}"
    )


def _checkpoint_paths(tools: CandidateTools, paths: Iterable[Path]) -> list[str]:
    return sorted(path.relative_to(tools.workspace).as_posix() for path in paths)
