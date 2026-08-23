from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from localpilot.evolution import normalize_evolution_task


_MANAGED_CANDIDATE_BRANCH = re.compile(
    r"^localpilot/candidate-[A-Za-z0-9][A-Za-z0-9._/-]*$"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PendingCandidate:
    cycle_id: int
    task_id: str
    branch: str
    workspace: str | None = None
    local_repair_attempts: int = 0
    status: str = ""
    write_integrity_failure: str = ""
    human_authorized_retry: bool = False
    retry_of_cycle_id: int | None = None
    retry_reason: str = ""


@dataclass(frozen=True, slots=True)
class ManagedCandidate:
    cycle_id: int
    task_id: str
    branch: str
    workspace: str | None
    is_worktree: bool
    pushed: bool
    pull_request_url: str | None
    validation_state: str
    merged: bool
    status: str
    summary: str
    reusable_lesson: str
    local_repair_attempts: int
    write_integrity_failure: str
    rejection_reason: str
    rejection_prior_validation_state: str
    rejection_pull_request_number: int | None
    rejected_at: str | None
    failure_attribution: str
    policy_failure_reason: str
    human_authorized_retry: bool
    retry_of_cycle_id: int | None
    retry_reason: str
    retry_authorized_at: str | None
    retried_by_cycle_id: int | None


@dataclass(frozen=True, slots=True)
class CandidatePolicyRetry:
    prior_cycle_id: int
    retry_cycle_id: int
    task_id: str
    prior_branch: str
    branch: str
    reason: str
    already_authorized: bool


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    candidate: ManagedCandidate
    already_rejected: bool


@dataclass(frozen=True, slots=True)
class CapabilityExperiment:
    id: int
    task_id: str
    title: str
    evolution_class: str
    capability_target: str
    question: str
    observed_limitation: str
    evidence: tuple[str, ...]
    alternatives: tuple[str, ...]
    hypothesis: str
    metric: str
    baseline: str
    success_criterion: str
    measurement_method: str
    expected_complexity: str
    status: str
    cycle_id: int | None
    branch: str | None
    outcome: str
    before_evidence: str
    after_evidence: str
    reusable_lesson: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class MissionFrontier:
    task_id: str
    mission_alignment: str
    current_frontier: str
    why_high_leverage: str
    capability_unlocked: str
    next_frontier: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class KnowledgeFact:
    stage: str
    fact_key: str
    fact_type: str
    subject: str
    summary: str
    source_uri: str
    source_kind: str
    source_digest: str
    confidence: float
    last_verified_at: str
    relationships: tuple[str, ...]
    stale: bool


@dataclass(frozen=True, slots=True)
class StudyRun:
    id: int
    stage: str
    phase: str
    benchmark_version: str
    question_set_digest: str
    score: float
    correct: int
    total: int
    latency_ms: int
    resource_cost: dict
    errors: tuple[str, ...]
    transferable_lessons: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class CurriculumStageState:
    stage: str
    status: str
    baseline_score: float | None
    latest_score: float | None
    known_weak_areas: tuple[str, ...]
    next_lesson: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class PeerModelComparison:
    id: int
    subject_model: str
    peer_model: str
    subject_score: float
    peer_score: float
    subject_latency_ms: int
    peer_latency_ms: int
    resource_cost: dict
    transferable_lessons: tuple[str, ...]
    created_at: str


class LearningMemory:
    """Durable cycle outcomes and reusable lessons.

    The schema deliberately has no prompt, transcript, message, or reasoning
    column. It stores reviewable facts about what happened, not chain-of-thought.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS development_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    everyday_model TEXT NOT NULL,
                    developer_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    reusable_lesson TEXT NOT NULL DEFAULT '',
                    checks_passed INTEGER,
                    pushed INTEGER NOT NULL DEFAULT 0,
                    pull_request_url TEXT,
                    validation_state TEXT NOT NULL DEFAULT 'not_started',
                    merged INTEGER NOT NULL DEFAULT 0,
                    workspace TEXT,
                    is_worktree INTEGER NOT NULL DEFAULT 0,
                    local_repair_attempts INTEGER NOT NULL DEFAULT 0,
                    write_integrity_failure TEXT NOT NULL DEFAULT '',
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    rejection_prior_validation_state TEXT NOT NULL DEFAULT '',
                    rejection_pull_request_number INTEGER,
                    rejected_at TEXT,
                    failure_attribution TEXT NOT NULL DEFAULT '',
                    policy_failure_reason TEXT NOT NULL DEFAULT '',
                    human_authorized_retry INTEGER NOT NULL DEFAULT 0,
                    retry_of_cycle_id INTEGER,
                    retry_reason TEXT NOT NULL DEFAULT '',
                    retry_authorized_at TEXT,
                    retried_by_cycle_id INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS cycles_task_idx
                    ON development_cycles(task_id, id DESC);
                CREATE INDEX IF NOT EXISTS cycles_pending_idx
                    ON development_cycles(merged, validation_state, pushed);
                CREATE TABLE IF NOT EXISTS capability_map (
                    capability_key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    current_state TEXT NOT NULL DEFAULT '',
                    known_limitation TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capability_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    evolution_class TEXT NOT NULL,
                    capability_target TEXT NOT NULL,
                    question TEXT NOT NULL,
                    observed_limitation TEXT NOT NULL,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    alternatives_json TEXT NOT NULL DEFAULT '[]',
                    hypothesis TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    baseline TEXT NOT NULL,
                    success_criterion TEXT NOT NULL,
                    measurement_method TEXT NOT NULL,
                    expected_complexity TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'proposed',
                    cycle_id INTEGER,
                    branch TEXT,
                    outcome TEXT NOT NULL DEFAULT '',
                    before_evidence TEXT NOT NULL DEFAULT '',
                    after_evidence TEXT NOT NULL DEFAULT '',
                    reusable_lesson TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experiments_status_idx
                    ON capability_experiments(status, id DESC);
                CREATE TABLE IF NOT EXISTS mission_frontiers (
                    task_id TEXT PRIMARY KEY,
                    mission_alignment TEXT NOT NULL,
                    current_frontier TEXT NOT NULL,
                    why_high_leverage TEXT NOT NULL,
                    capability_unlocked TEXT NOT NULL,
                    next_frontier TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS mission_frontiers_updated_idx
                    ON mission_frontiers(updated_at DESC);
                CREATE TABLE IF NOT EXISTS knowledge_facts (
                    stage TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_digest TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_verified_at TEXT NOT NULL,
                    relationships_json TEXT NOT NULL DEFAULT '[]',
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(stage, fact_key)
                );
                CREATE INDEX IF NOT EXISTS knowledge_source_idx
                    ON knowledge_facts(source_uri, stale);
                CREATE INDEX IF NOT EXISTS knowledge_stage_idx
                    ON knowledge_facts(stage, stale, fact_type);
                CREATE TABLE IF NOT EXISTS study_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    benchmark_version TEXT NOT NULL,
                    question_set_digest TEXT NOT NULL,
                    score REAL NOT NULL,
                    correct INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    resource_cost_json TEXT NOT NULL DEFAULT '{}',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    transferable_lessons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS study_runs_stage_idx
                    ON study_runs(stage, id DESC);
                CREATE TABLE IF NOT EXISTS curriculum_stage_state (
                    stage TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'not_started',
                    baseline_run_id INTEGER,
                    latest_run_id INTEGER,
                    known_weak_areas_json TEXT NOT NULL DEFAULT '[]',
                    next_lesson TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS peer_model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_model TEXT NOT NULL,
                    peer_model TEXT NOT NULL,
                    subject_score REAL NOT NULL,
                    peer_score REAL NOT NULL,
                    subject_latency_ms INTEGER NOT NULL,
                    peer_latency_ms INTEGER NOT NULL,
                    resource_cost_json TEXT NOT NULL DEFAULT '{}',
                    transferable_lessons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(development_cycles)"
                ).fetchall()
            }
            migrations = {
                "workspace": "ALTER TABLE development_cycles ADD COLUMN workspace TEXT",
                "is_worktree": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "is_worktree INTEGER NOT NULL DEFAULT 0"
                ),
                "local_repair_attempts": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "local_repair_attempts INTEGER NOT NULL DEFAULT 0"
                ),
                "write_integrity_failure": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "write_integrity_failure TEXT NOT NULL DEFAULT ''"
                ),
                "rejection_reason": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "rejection_reason TEXT NOT NULL DEFAULT ''"
                ),
                "rejection_prior_validation_state": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "rejection_prior_validation_state TEXT NOT NULL DEFAULT ''"
                ),
                "rejection_pull_request_number": (
                    "ALTER TABLE development_cycles ADD COLUMN "
                    "rejection_pull_request_number INTEGER"
                ),
                "rejected_at": (
                    "ALTER TABLE development_cycles ADD COLUMN rejected_at TEXT"
                ),
                "failure_attribution": (
                    "ALTER TABLE development_cycles ADD COLUMN failure_attribution TEXT NOT NULL DEFAULT ''"
                ),
                "policy_failure_reason": (
                    "ALTER TABLE development_cycles ADD COLUMN policy_failure_reason TEXT NOT NULL DEFAULT ''"
                ),
                "human_authorized_retry": (
                    "ALTER TABLE development_cycles ADD COLUMN human_authorized_retry INTEGER NOT NULL DEFAULT 0"
                ),
                "retry_of_cycle_id": (
                    "ALTER TABLE development_cycles ADD COLUMN retry_of_cycle_id INTEGER"
                ),
                "retry_reason": (
                    "ALTER TABLE development_cycles ADD COLUMN retry_reason TEXT NOT NULL DEFAULT ''"
                ),
                "retry_authorized_at": (
                    "ALTER TABLE development_cycles ADD COLUMN retry_authorized_at TEXT"
                ),
                "retried_by_cycle_id": (
                    "ALTER TABLE development_cycles ADD COLUMN retried_by_cycle_id INTEGER"
                ),
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            experiment_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(capability_experiments)"
                ).fetchall()
            }
            if "title" not in experiment_columns:
                connection.execute(
                    "ALTER TABLE capability_experiments ADD COLUMN title TEXT NOT NULL DEFAULT 'Capability experiment'"
                )

    def start_cycle(
        self,
        *,
        task_id: str,
        branch: str,
        everyday_model: str,
        developer_model: str,
        workspace: str | Path | None = None,
        is_worktree: bool = False,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO development_cycles (
                    task_id, branch, everyday_model, developer_model, status,
                    workspace, is_worktree, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    task_id,
                    branch,
                    everyday_model,
                    developer_model,
                    str(Path(workspace).resolve()) if workspace is not None else None,
                    int(is_worktree),
                    _now(),
                ),
            )
            return int(cursor.lastrowid)

    def update_candidate_workspace(self, cycle_id: int, workspace: str | Path) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE development_cycles SET workspace = ? WHERE id = ?",
                (str(Path(workspace).resolve()), cycle_id),
            )

    def record_local_repair_attempt(
        self,
        cycle_id: int,
        *,
        check_result: str,
    ) -> int:
        """Durably count a model repair before it starts.

        Recording first makes a crash or power loss consume an attempt instead
        of silently creating an unbounded retry loop on the next invocation.
        """
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE development_cycles
                SET local_repair_attempts = local_repair_attempts + 1,
                    status = 'candidate_needs_work', checks_passed = 0,
                    summary = ?, finished_at = ?
                WHERE id = ?
                """,
                (check_result[:4000], _now(), cycle_id),
            )
            row = connection.execute(
                "SELECT local_repair_attempts FROM development_cycles WHERE id = ?",
                (cycle_id,),
            ).fetchone()
        return int(row["local_repair_attempts"]) if row is not None else 0

    def record_write_integrity_failure(self, cycle_id: int, detail: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE development_cycles SET write_integrity_failure = ? WHERE id = ?",
                (str(detail).strip()[:4000], cycle_id),
            )

    def clear_write_integrity_failure(self, cycle_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE development_cycles SET write_integrity_failure = '' WHERE id = ?",
                (cycle_id,),
            )

    def finish_cycle(
        self,
        cycle_id: int,
        *,
        status: str,
        summary: str,
        reusable_lesson: str,
        checks_passed: bool | None,
        pushed: bool,
        validation_state: str | None = None,
    ) -> None:
        next_validation_state = validation_state or (
            "awaiting_pr" if pushed else "not_started"
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE development_cycles
                SET status = ?, summary = ?, reusable_lesson = ?, checks_passed = ?,
                    pushed = ?, validation_state = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    summary[:4000],
                    reusable_lesson[:2000],
                    None if checks_passed is None else int(checks_passed),
                    int(pushed),
                    next_validation_state,
                    _now(),
                    cycle_id,
                ),
            )

    def pending_candidates(self) -> list[PendingCandidate]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, branch FROM development_cycles
                WHERE pushed = 1
                  AND validation_state != 'rejected_by_human'
                  AND NOT (merged = 1 AND validation_state = 'passed')
                  AND NOT (
                    status = 'policy_blocked'
                    AND retried_by_cycle_id IS NOT NULL
                  )
                ORDER BY id
                """
            ).fetchall()
        return [PendingCandidate(int(row["id"]), row["task_id"], row["branch"]) for row in rows]

    def local_candidates(self) -> list[PendingCandidate]:
        """Return unfinished, unpushed Git candidates that still own a worktree."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, branch, workspace,
                       local_repair_attempts, status, write_integrity_failure,
                       human_authorized_retry, retry_of_cycle_id, retry_reason
                FROM development_cycles
                WHERE pushed = 0
                  AND merged = 0
                  AND is_worktree = 1
                  AND status IN ('running', 'paused', 'candidate_needs_work')
                ORDER BY id
                """
            ).fetchall()
        return [
            PendingCandidate(
                int(row["id"]),
                str(row["task_id"]),
                str(row["branch"]),
                str(row["workspace"]) if row["workspace"] else None,
                int(row["local_repair_attempts"]),
                str(row["status"]),
                str(row["write_integrity_failure"]),
                bool(row["human_authorized_retry"]),
                int(row["retry_of_cycle_id"]) if row["retry_of_cycle_id"] is not None else None,
                str(row["retry_reason"]),
            )
            for row in rows
        ]

    def failed_candidates(self) -> list[PendingCandidate]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, branch, workspace,
                       local_repair_attempts, status
                FROM development_cycles
                WHERE pushed = 1
                  AND validation_state = 'failed'
                  AND merged = 0
                  AND NOT (
                    status = 'policy_blocked'
                    AND retried_by_cycle_id IS NOT NULL
                  )
                ORDER BY id
                """
            ).fetchall()

        return [
            PendingCandidate(
                int(row["id"]),
                str(row["task_id"]),
                str(row["branch"]),
                str(row["workspace"]) if row["workspace"] else None,
                int(row["local_repair_attempts"]),
                str(row["status"]),
            )
            for row in rows
        ]

    def update_candidate_review(
        self,
        cycle_id: int,
        *,
        validation_state: str,
        merged: bool,
        pull_request_url: str | None,
    ) -> None:
        allowed = {"awaiting_pr", "pending", "passed", "failed"}
        if validation_state not in allowed:
            raise ValueError(f"Unsupported validation state: {validation_state}")
        status = "merged" if merged and validation_state == "passed" else "candidate_pending_validation"
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT validation_state, status, retried_by_cycle_id
                FROM development_cycles WHERE id = ?
                """,
                (cycle_id,),
            ).fetchone()
            if existing is not None and existing["validation_state"] == "rejected_by_human":
                return
            if (
                existing is not None
                and existing["status"] == "policy_blocked"
                and existing["retried_by_cycle_id"] is not None
            ):
                return
            connection.execute(
                """
                UPDATE development_cycles
                SET validation_state = ?, merged = ?, pull_request_url = ?, status = ?
                WHERE id = ?
                """,
                (validation_state, int(merged), pull_request_url, status, cycle_id),
            )

    def completed_task_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT task_id FROM development_cycles
                WHERE merged = 1 AND validation_state = 'passed'
                """
            ).fetchall()
        return {str(row["task_id"]) for row in rows}

    def rejected_task_ids(self) -> set[str]:
        """Return tasks with an explicit, terminal human rejection."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT task_id FROM development_cycles
                WHERE validation_state = 'rejected_by_human'
                """
            ).fetchall()
        return {str(row["task_id"]) for row in rows}

    def pending_task_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT task_id FROM development_cycles
                WHERE (
                    pushed = 1
                    AND validation_state != 'rejected_by_human'
                    AND NOT (merged = 1 AND validation_state = 'passed')
                    AND NOT (
                        status = 'policy_blocked'
                        AND retried_by_cycle_id IS NOT NULL
                    )
                ) OR (
                    pushed = 0
                    AND is_worktree = 1
                    AND status IN (
                        'running', 'paused', 'candidate_ready',
                        'candidate_needs_work'
                    )
                )
                """
            ).fetchall()
        return {str(row["task_id"]) for row in rows}

    def has_outstanding_candidate(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM development_cycles
                WHERE NOT (merged = 1 AND validation_state = 'passed')
                  AND validation_state != 'rejected_by_human'
                  AND NOT (
                    status = 'policy_blocked'
                    AND retried_by_cycle_id IS NOT NULL
                  )
                  AND (
                    pushed = 1 OR (
                        pushed = 0 AND is_worktree = 1
                        AND status IN (
                            'running', 'paused', 'candidate_ready',
                            'candidate_needs_work'
                        )
                    )
                  )
                LIMIT 1
                """
            ).fetchone()
        return row is not None

    @staticmethod
    def _managed_candidate(row: sqlite3.Row | None) -> ManagedCandidate | None:
        if row is None:
            return None
        return ManagedCandidate(
            cycle_id=int(row["id"]),
            task_id=str(row["task_id"]),
            branch=str(row["branch"]),
            workspace=str(row["workspace"]) if row["workspace"] else None,
            is_worktree=bool(row["is_worktree"]),
            pushed=bool(row["pushed"]),
            pull_request_url=(
                str(row["pull_request_url"]) if row["pull_request_url"] else None
            ),
            validation_state=str(row["validation_state"]),
            merged=bool(row["merged"]),
            status=str(row["status"]),
            summary=str(row["summary"]),
            reusable_lesson=str(row["reusable_lesson"]),
            local_repair_attempts=int(row["local_repair_attempts"]),
            write_integrity_failure=str(row["write_integrity_failure"]),
            rejection_reason=str(row["rejection_reason"]),
            rejection_prior_validation_state=str(
                row["rejection_prior_validation_state"]
            ),
            rejection_pull_request_number=(
                int(row["rejection_pull_request_number"])
                if row["rejection_pull_request_number"] is not None
                else None
            ),
            rejected_at=str(row["rejected_at"]) if row["rejected_at"] else None,
            failure_attribution=str(row["failure_attribution"]),
            policy_failure_reason=str(row["policy_failure_reason"]),
            human_authorized_retry=bool(row["human_authorized_retry"]),
            retry_of_cycle_id=(
                int(row["retry_of_cycle_id"])
                if row["retry_of_cycle_id"] is not None else None
            ),
            retry_reason=str(row["retry_reason"]),
            retry_authorized_at=(
                str(row["retry_authorized_at"]) if row["retry_authorized_at"] else None
            ),
            retried_by_cycle_id=(
                int(row["retried_by_cycle_id"])
                if row["retried_by_cycle_id"] is not None else None
            ),
        )

    @staticmethod
    def _policy_retry_branch(prior_branch: str, prior_cycle_id: int) -> str:
        return f"{prior_branch}-policy-retry-{prior_cycle_id}"

    def authorize_policy_retry(
        self,
        identifier: str,
        *,
        reason: str,
        remote_branch_verified: bool = False,
        remote_merged: bool | None = None,
        pull_request_state: str = "unknown",
    ) -> CandidatePolicyRetry:
        """Link a new bounded retry cycle to one proven framework-policy failure."""
        clean_identifier = str(identifier).strip()
        clean_reason = str(reason).replace("\x00", " ").strip()[:1800]
        if not clean_identifier or not clean_reason:
            raise ValueError("Candidate identifier and human retry reason are required.")
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM development_cycles
                WHERE branch = ? OR task_id = ?
                ORDER BY id DESC
                """,
                (clean_identifier, clean_identifier),
            ).fetchall()
            if not rows:
                raise ValueError("No LocalPilot-managed candidate matches that branch or task.")
            row = rows[0]
            if bool(row["human_authorized_retry"]) and row["retry_of_cycle_id"] is not None:
                prior = connection.execute(
                    "SELECT branch FROM development_cycles WHERE id = ?",
                    (int(row["retry_of_cycle_id"]),),
                ).fetchone()
                return CandidatePolicyRetry(
                    int(row["retry_of_cycle_id"]), int(row["id"]), str(row["task_id"]),
                    str(prior["branch"]) if prior is not None else str(row["branch"]),
                    str(row["branch"]), str(row["retry_reason"]), True,
                )
            if row["retried_by_cycle_id"] is not None:
                retry = connection.execute(
                    "SELECT * FROM development_cycles WHERE id = ?",
                    (int(row["retried_by_cycle_id"]),),
                ).fetchone()
                if retry is not None:
                    return CandidatePolicyRetry(
                        int(row["id"]), int(retry["id"]), str(retry["task_id"]),
                        str(row["branch"]), str(retry["branch"]),
                        str(retry["retry_reason"]), True,
                    )
            branch = str(row["branch"])
            if not _MANAGED_CANDIDATE_BRANCH.fullmatch(branch):
                raise ValueError("Retry is limited to a LocalPilot-managed candidate branch.")
            if (
                str(row["validation_state"]) == "rejected_by_human"
                or str(row["status"]) == "rejected_by_human"
                or bool(row["rejected_at"])
            ):
                raise ValueError("Retry refused: the candidate was explicitly rejected by a human.")
            promoted = connection.execute(
                """
                SELECT 1 FROM development_cycles
                WHERE task_id = ? AND merged = 1 AND validation_state = 'passed'
                LIMIT 1
                """,
                (str(row["task_id"]),),
            ).fetchone()
            if bool(row["merged"]) or promoted is not None:
                raise ValueError("Retry refused: the candidate task is already merged or promoted.")
            pushed = bool(row["pushed"])
            if pushed:
                normalized_pr_state = str(pull_request_state).strip().lower()
                if (
                    not remote_branch_verified
                    or remote_merged is not False
                    or normalized_pr_state not in {"open", "closed", "none"}
                ):
                    raise ValueError(
                        "Retry refused: the pushed branch and unmerged pull-request state "
                        "could not be verified."
                    )
            elif (
                not bool(row["is_worktree"])
                or str(row["status"]) not in {"candidate_needs_work", "failed"}
            ):
                raise ValueError("Retry is limited to a failed local candidate.")
            evidence = (
                f"{row['summary']}\n{row['write_integrity_failure']}\n"
                f"{row['policy_failure_reason']}"
            ).lower()
            policy_markers = (
                "candidate delivery blocked", "file-write limit",
                "hard file ceiling", "disallowed file type", "file type is not allowed",
                "autonomous editing", "recovered candidate exceeds the file-write limit",
                "directory creation", "directory write", "framework policy",
            )
            if (
                str(row["failure_attribution"]) != "framework_policy"
                and not any(marker in evidence for marker in policy_markers)
            ):
                raise ValueError(
                    "Retry refused: durable evidence does not show a framework-policy write block."
                )
            other = connection.execute(
                """
                SELECT id FROM development_cycles
                WHERE id != ? AND validation_state != 'rejected_by_human'
                  AND NOT (merged = 1 AND validation_state = 'passed')
                  AND NOT (
                    status = 'policy_blocked'
                    AND retried_by_cycle_id IS NOT NULL
                  )
                  AND (pushed = 1 OR (
                    pushed = 0 AND is_worktree = 1
                    AND status IN ('running', 'paused', 'candidate_ready', 'candidate_needs_work')
                  ))
                LIMIT 1
                """,
                (int(row["id"]),),
            ).fetchone()
            if other is not None:
                raise ValueError("Retry refused while another nonterminal candidate exists.")
            retry_branch = (
                self._policy_retry_branch(branch, int(row["id"]))
                if pushed else branch
            )
            retry_workspace = None if pushed else (
                str(row["workspace"]) if row["workspace"] else None
            )
            cursor = connection.execute(
                """
                INSERT INTO development_cycles (
                    task_id, branch, everyday_model, developer_model, status,
                    summary, reusable_lesson, checks_passed, pushed,
                    validation_state, merged, workspace, is_worktree,
                    local_repair_attempts, write_integrity_failure,
                    human_authorized_retry, retry_of_cycle_id, retry_reason,
                    retry_authorized_at, started_at
                ) VALUES (?, ?, ?, ?, 'candidate_needs_work', ?, ?, 0, 0,
                          'not_started', 0, ?, 1, 0, '', 1, ?, ?, ?, ?)
                """,
                (
                    str(row["task_id"]), retry_branch, str(row["everyday_model"]),
                    str(row["developer_model"]),
                    "Human-authorized retry of framework-policy-blocked prior cycle "
                    f"{row['id']} on {branch}.",
                    "The prior candidate idea was not at fault; framework policy blocked its construction.",
                    retry_workspace,
                    int(row["id"]), clean_reason, now, now,
                ),
            )
            retry_id = int(cursor.lastrowid)
            connection.execute(
                """
                UPDATE development_cycles
                SET status = 'policy_blocked',
                    failure_attribution = 'framework_policy',
                    policy_failure_reason = ?, retried_by_cycle_id = ?,
                    reusable_lesson = ?
                WHERE id = ?
                """,
                (
                    clean_reason, retry_id,
                    self._append_lesson(
                        str(row["reusable_lesson"]),
                        "Failure attribution: framework policy was too restrictive; preserve the candidate objective and retry after policy correction.",
                    ),
                    int(row["id"]),
                ),
            )
            connection.execute(
                """
                UPDATE capability_experiments
                SET cycle_id = ?, branch = ?, status = 'candidate_active', updated_at = ?
                WHERE task_id = ?
                """,
                (retry_id, retry_branch, now, str(row["task_id"])),
            )
        return CandidatePolicyRetry(
            int(row["id"]), retry_id, str(row["task_id"]), branch,
            retry_branch, clean_reason, False,
        )

    def candidate_for_branch(self, branch: str) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM development_cycles WHERE branch = ? ORDER BY id DESC LIMIT 1",
                (str(branch),),
            ).fetchone()
        return self._managed_candidate(row)

    def candidate_for_identifier(self, identifier: str) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM development_cycles
                WHERE branch = ? OR task_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (str(identifier), str(identifier)),
            ).fetchone()
        return self._managed_candidate(row)

    def candidate_for_cycle(self, cycle_id: int) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM development_cycles WHERE id = ?",
                (int(cycle_id),),
            ).fetchone()
        return self._managed_candidate(row)

    def latest_rejected_candidate(self) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM development_cycles
                WHERE validation_state = 'rejected_by_human'
                ORDER BY rejected_at DESC, id DESC LIMIT 1
                """
            ).fetchone()
        return self._managed_candidate(row)

    def latest_policy_retry(self) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM development_cycles
                WHERE human_authorized_retry = 1
                ORDER BY retry_authorized_at DESC, id DESC LIMIT 1
                """
            ).fetchone()
        return self._managed_candidate(row)

    @staticmethod
    def _append_lesson(existing: str, lesson: str) -> str:
        # The newly observed lesson must survive the size bound; retain prior
        # evidence in the remaining space.
        parts = [part.strip() for part in (lesson, existing) if part and part.strip()]
        return "\n".join(dict.fromkeys(parts))[:2000]

    def reject_candidate(
        self,
        cycle_id: int,
        *,
        pull_request_number: int,
        pull_request_url: str,
        reason: str,
    ) -> CandidateRejection:
        """Atomically retain evidence and make one explicit rejection terminal."""
        clean_reason = str(reason).replace("\x00", " ").strip()[:1800]
        if not clean_reason:
            raise ValueError("A human rejection reason must not be empty.")
        if (
            isinstance(pull_request_number, bool)
            or int(pull_request_number) <= 0
        ):
            raise ValueError("Pull request number must be a positive integer.")
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM development_cycles WHERE id = ?",
                (int(cycle_id),),
            ).fetchone()
            candidate = self._managed_candidate(row)
            if candidate is None:
                raise ValueError(f"Development cycle {cycle_id} does not exist.")
            if candidate.validation_state == "rejected_by_human":
                return CandidateRejection(candidate, True)
            if candidate.merged:
                raise ValueError("A merged candidate cannot be rejected.")
            if not candidate.pushed:
                raise ValueError("Only a pushed candidate with a GitHub PR can be rejected.")

            rejection_lesson = f"Human rejection: {clean_reason}"
            cycle_lesson = self._append_lesson(
                str(row["reusable_lesson"]), rejection_lesson
            )
            connection.execute(
                """
                UPDATE development_cycles
                SET status = 'rejected_by_human',
                    validation_state = 'rejected_by_human',
                    rejection_reason = ?,
                    rejection_prior_validation_state = ?,
                    rejection_pull_request_number = ?,
                    pull_request_url = ?,
                    reusable_lesson = ?,
                    rejected_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    clean_reason,
                    candidate.validation_state,
                    int(pull_request_number),
                    str(pull_request_url)[:2000],
                    cycle_lesson,
                    now,
                    now,
                    int(cycle_id),
                ),
            )

            experiment = connection.execute(
                "SELECT outcome, reusable_lesson FROM capability_experiments WHERE task_id = ?",
                (candidate.task_id,),
            ).fetchone()
            if experiment is not None:
                outcome = self._append_lesson(
                    str(experiment["outcome"]), f"Rejected by human: {clean_reason}"
                )
                experiment_lesson = self._append_lesson(
                    str(experiment["reusable_lesson"]), rejection_lesson
                )
                connection.execute(
                    """
                    UPDATE capability_experiments
                    SET status = 'rejected_by_human', outcome = ?,
                        reusable_lesson = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    (outcome, experiment_lesson, now, candidate.task_id),
                )

            updated = connection.execute(
                "SELECT * FROM development_cycles WHERE id = ?",
                (int(cycle_id),),
            ).fetchone()
        result = self._managed_candidate(updated)
        if result is None:
            raise RuntimeError("Rejected candidate could not be reloaded.")
        return CandidateRejection(result, False)

    @staticmethod
    def _experiment(row: sqlite3.Row | None) -> CapabilityExperiment | None:
        if row is None:
            return None

        def values(name: str) -> tuple[str, ...]:
            try:
                item = json.loads(str(row[name] or "[]"))
            except json.JSONDecodeError:
                return ()
            return tuple(str(part) for part in item if isinstance(part, str)) if isinstance(item, list) else ()

        return CapabilityExperiment(
            id=int(row["id"]),
            task_id=str(row["task_id"]),
            title=str(row["title"]),
            evolution_class=str(row["evolution_class"]),
            capability_target=str(row["capability_target"]),
            question=str(row["question"]),
            observed_limitation=str(row["observed_limitation"]),
            evidence=values("evidence_json"),
            alternatives=values("alternatives_json"),
            hypothesis=str(row["hypothesis"]),
            metric=str(row["metric"]),
            baseline=str(row["baseline"]),
            success_criterion=str(row["success_criterion"]),
            measurement_method=str(row["measurement_method"]),
            expected_complexity=str(row["expected_complexity"]),
            status=str(row["status"]),
            cycle_id=int(row["cycle_id"]) if row["cycle_id"] is not None else None,
            branch=str(row["branch"]) if row["branch"] else None,
            outcome=str(row["outcome"]),
            before_evidence=str(row["before_evidence"]),
            after_evidence=str(row["after_evidence"]),
            reusable_lesson=str(row["reusable_lesson"]),
            updated_at=str(row["updated_at"]),
        )

    def record_experiment(self, task: dict) -> int:
        task = normalize_evolution_task(task)
        evaluation = task["evaluation"]
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO capability_experiments (
                    task_id, title, evolution_class, capability_target, question,
                    observed_limitation, evidence_json, alternatives_json,
                    hypothesis, metric, baseline, success_criterion,
                    measurement_method, expected_complexity, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    title = excluded.title,
                    evolution_class = excluded.evolution_class,
                    capability_target = excluded.capability_target,
                    question = excluded.question,
                    observed_limitation = excluded.observed_limitation,
                    evidence_json = excluded.evidence_json,
                    alternatives_json = excluded.alternatives_json,
                    hypothesis = excluded.hypothesis,
                    metric = excluded.metric,
                    baseline = excluded.baseline,
                    success_criterion = excluded.success_criterion,
                    measurement_method = excluded.measurement_method,
                    expected_complexity = excluded.expected_complexity,
                    updated_at = excluded.updated_at
                """,
                (
                    str(task["id"])[:200],
                    str(task["title"])[:1000],
                    str(task["evolution_class"])[:40],
                    str(task["capability_target"])[:1000],
                    str(task["question"])[:1000],
                    str(task["observed_limitation"])[:2000],
                    json.dumps(list(task.get("evidence") or [])[:12]),
                    json.dumps(list(task.get("alternatives") or [])[:12]),
                    str(task["hypothesis"])[:2000],
                    str(evaluation["metric"])[:1000],
                    str(evaluation["baseline"])[:1000],
                    str(evaluation["success_criterion"])[:1000],
                    str(evaluation["measurement_method"])[:1000],
                    str(task.get("expected_complexity") or "medium")[:20],
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT id FROM capability_experiments WHERE task_id = ?",
                (str(task["id"]),),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO capability_map (
                    capability_key, name, current_state, known_limitation,
                    evidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(capability_key) DO UPDATE SET
                    name = excluded.name,
                    known_limitation = excluded.known_limitation,
                    evidence = excluded.evidence,
                    updated_at = excluded.updated_at
                """,
                (
                    str(task["capability_target"]).lower()[:300],
                    str(task["capability_target"])[:1000],
                    "experiment proposed",
                    str(task["observed_limitation"])[:2000],
                    "; ".join(str(item) for item in (task.get("evidence") or [])[:8])[:2000],
                    now,
                ),
            )
        experiment_id = int(row["id"])
        self.record_mission_frontier(task)
        return experiment_id

    def experiment_for_task(self, task_id: str) -> CapabilityExperiment | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM capability_experiments WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        return self._experiment(row)

    def latest_experiment(self) -> CapabilityExperiment | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM capability_experiments ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return self._experiment(row)

    @staticmethod
    def _frontier(row: sqlite3.Row | None) -> MissionFrontier | None:
        if row is None:
            return None
        return MissionFrontier(
            task_id=str(row["task_id"]),
            mission_alignment=str(row["mission_alignment"]),
            current_frontier=str(row["current_frontier"]),
            why_high_leverage=str(row["why_high_leverage"]),
            capability_unlocked=str(row["capability_unlocked"]),
            next_frontier=str(row["next_frontier"]),
            updated_at=str(row["updated_at"]),
        )

    def record_mission_frontier(self, task: dict) -> None:
        task = normalize_evolution_task(task)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO mission_frontiers (
                    task_id, mission_alignment, current_frontier,
                    why_high_leverage, capability_unlocked, next_frontier,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    mission_alignment = excluded.mission_alignment,
                    current_frontier = excluded.current_frontier,
                    why_high_leverage = excluded.why_high_leverage,
                    capability_unlocked = excluded.capability_unlocked,
                    next_frontier = excluded.next_frontier,
                    updated_at = excluded.updated_at
                """,
                (
                    str(task["id"])[:200],
                    str(task["mission_alignment"])[:2000],
                    str(task["current_frontier"])[:2000],
                    str(task["why_high_leverage"])[:2000],
                    str(task["capability_unlocked"])[:2000],
                    str(task["next_frontier"])[:2000],
                    _now(),
                ),
            )

    def frontier_for_task(self, task_id: str) -> MissionFrontier | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mission_frontiers WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        return self._frontier(row)

    def latest_frontier(self) -> MissionFrontier | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mission_frontiers ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return self._frontier(row)

    def experiment_task(self, task_id: str) -> dict | None:
        experiment = self.experiment_for_task(task_id)
        if experiment is None:
            return None
        task = {
            "id": experiment.task_id,
            "title": experiment.title,
            "status": "todo",
            "source": "capability_discovery",
            "evolution_class": experiment.evolution_class,
            "capability_target": experiment.capability_target,
            "question": experiment.question,
            "observed_limitation": experiment.observed_limitation,
            "evidence": list(experiment.evidence),
            "alternatives": list(experiment.alternatives),
            "hypothesis": experiment.hypothesis,
            "expected_complexity": experiment.expected_complexity,
            "evaluation": {
                "metric": experiment.metric,
                "baseline": experiment.baseline,
                "success_criterion": experiment.success_criterion,
                "measurement_method": experiment.measurement_method,
            },
        }
        frontier = self.frontier_for_task(task_id)
        if frontier is not None:
            task.update(
                {
                    "mission_alignment": frontier.mission_alignment,
                    "current_frontier": frontier.current_frontier,
                    "why_high_leverage": frontier.why_high_leverage,
                    "capability_unlocked": frontier.capability_unlocked,
                    "next_frontier": frontier.next_frontier,
                }
            )
        return normalize_evolution_task(task)

    def attach_experiment_cycle(self, task_id: str, cycle_id: int, branch: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE capability_experiments
                SET cycle_id = ?, branch = ?, status = 'candidate_active', updated_at = ?
                WHERE task_id = ?
                """,
                (cycle_id, branch[:300], _now(), str(task_id)),
            )

    def update_experiment_outcome(
        self,
        task_id: str,
        *,
        status: str,
        outcome: str = "",
        before_evidence: str = "",
        after_evidence: str = "",
        reusable_lesson: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE capability_experiments
                SET status = ?, outcome = ?, before_evidence = ?, after_evidence = ?,
                    reusable_lesson = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    status[:80],
                    outcome[:2000],
                    before_evidence[:2000],
                    after_evidence[:2000],
                    reusable_lesson[:2000],
                    _now(),
                    str(task_id),
                ),
            )

    def update_experiment_review(
        self,
        task_id: str,
        *,
        validation_state: str,
        merged: bool,
    ) -> None:
        experiment = self.experiment_for_task(task_id)
        if experiment is None:
            return
        if experiment.status == "rejected_by_human":
            return
        if validation_state == "failed":
            status = "evaluation_failed"
        elif merged and validation_state == "passed":
            status = "validated"
        elif validation_state == "passed":
            status = "validated_pending_human_merge"
        else:
            status = "evaluation_pending"
        with self._connect() as connection:
            connection.execute(
                "UPDATE capability_experiments SET status = ?, updated_at = ? WHERE task_id = ?",
                (status, _now(), task_id),
            )
            if status == "validated":
                connection.execute(
                    """
                    INSERT INTO capability_map (
                        capability_key, name, current_state, known_limitation,
                        evidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(capability_key) DO UPDATE SET
                        name = excluded.name,
                        current_state = excluded.current_state,
                        known_limitation = excluded.known_limitation,
                        evidence = excluded.evidence,
                        updated_at = excluded.updated_at
                    """,
                    (
                        experiment.capability_target.lower()[:300],
                        experiment.capability_target[:1000],
                        (experiment.after_evidence or experiment.outcome or "Validated by CI and human merge")[:2000],
                        "",
                        (experiment.outcome or experiment.after_evidence)[:2000],
                        _now(),
                    ),
                )

    def discovery_context(self, limit: int = 8) -> dict:
        limit = max(1, min(int(limit), 20))
        with self._connect() as connection:
            capabilities = connection.execute(
                "SELECT * FROM capability_map ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            experiments = connection.execute(
                """
                SELECT evolution_class, capability_target, hypothesis, status,
                       outcome,
                       CASE WHEN status IN ('validated', 'rejected_by_human')
                           THEN reusable_lesson ELSE '' END
                           AS reusable_lesson,
                       updated_at
                FROM capability_experiments ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            frontiers = connection.execute(
                """
                SELECT task_id, mission_alignment, current_frontier,
                       why_high_leverage, capability_unlocked, next_frontier,
                       updated_at
                FROM mission_frontiers
                ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            cycles = connection.execute(
                """
                SELECT task_id, status, summary,
                       CASE WHEN (merged = 1 AND validation_state = 'passed')
                                  OR validation_state = 'rejected_by_human'
                           THEN reusable_lesson ELSE '' END AS reusable_lesson,
                       checks_passed, validation_state, merged,
                       rejection_reason, rejection_prior_validation_state
                FROM development_cycles ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "capabilities": [dict(row) for row in capabilities],
            "frontiers": [dict(row) for row in frontiers],
            "experiments": [dict(row) for row in experiments],
            "recent_cycles": [dict(row) for row in cycles],
        }

    def reusable_lessons(self, limit: int = 6) -> list[str]:
        limit = max(0, min(int(limit), 50))
        if not limit:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reusable_lesson FROM development_cycles
                WHERE reusable_lesson != ''
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row["reusable_lesson"]) for row in rows]

    @staticmethod
    def _knowledge_fact(row: sqlite3.Row) -> KnowledgeFact:
        return KnowledgeFact(
            stage=str(row["stage"]),
            fact_key=str(row["fact_key"]),
            fact_type=str(row["fact_type"]),
            subject=str(row["subject"]),
            summary=str(row["summary"]),
            source_uri=str(row["source_uri"]),
            source_kind=str(row["source_kind"]),
            source_digest=str(row["source_digest"]),
            confidence=float(row["confidence"]),
            last_verified_at=str(row["last_verified_at"]),
            relationships=tuple(json.loads(row["relationships_json"] or "[]")),
            stale=bool(row["stale"]),
        )

    def upsert_knowledge_fact(
        self,
        *,
        stage: str,
        fact_key: str,
        fact_type: str,
        subject: str,
        summary: str,
        source_uri: str,
        source_kind: str,
        source_digest: str,
        confidence: float,
        relationships: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Persist one concise, attributable fact, never source bodies or reasoning."""
        self.upsert_knowledge_facts(
            [
                {
                    "stage": stage,
                    "fact_key": fact_key,
                    "fact_type": fact_type,
                    "subject": subject,
                    "summary": summary,
                    "source_uri": source_uri,
                    "source_kind": source_kind,
                    "source_digest": source_digest,
                    "confidence": confidence,
                    "relationships": relationships,
                }
            ]
        )

    def upsert_knowledge_facts(self, facts: list[dict]) -> None:
        """Batch concise facts into one transaction for bounded study overhead."""
        if not facts:
            return
        with self._connect() as connection:
            for fact in facts:
                confidence = max(0.0, min(float(fact["confidence"]), 1.0))
                relationships = tuple(fact.get("relationships") or ())
                connection.execute(
                    """
                    UPDATE knowledge_facts SET stale = 1
                    WHERE source_uri = ? AND source_digest != ? AND stale = 0
                    """,
                    (str(fact["source_uri"]), str(fact["source_digest"])),
                )
                connection.execute(
                    """
                    INSERT INTO knowledge_facts (
                        stage, fact_key, fact_type, subject, summary, source_uri,
                        source_kind, source_digest, confidence, last_verified_at,
                        relationships_json, stale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(stage, fact_key) DO UPDATE SET
                        fact_type = excluded.fact_type,
                        subject = excluded.subject,
                        summary = excluded.summary,
                        source_uri = excluded.source_uri,
                        source_kind = excluded.source_kind,
                        source_digest = excluded.source_digest,
                        confidence = excluded.confidence,
                        last_verified_at = excluded.last_verified_at,
                        relationships_json = excluded.relationships_json,
                        stale = 0
                    """,
                    (
                        str(fact["stage"])[:40],
                        str(fact["fact_key"])[:500],
                        str(fact["fact_type"])[:80],
                        str(fact["subject"])[:500],
                        str(fact["summary"])[:1200],
                        str(fact["source_uri"])[:1000],
                        str(fact["source_kind"])[:80],
                        str(fact["source_digest"])[:128],
                        confidence,
                        _now(),
                        json.dumps(
                            [str(item)[:500] for item in relationships[:30]]
                        ),
                    ),
                )

    def invalidate_knowledge_source(
        self, source_uri: str, current_digest: str
    ) -> int:
        """Mark facts stale when their authoritative source content changed."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE knowledge_facts SET stale = 1
                WHERE source_uri = ? AND source_digest != ? AND stale = 0
                """,
                (source_uri, current_digest),
            )
            return int(cursor.rowcount)

    def knowledge_facts(
        self,
        *,
        stage: str | None = None,
        fact_type: str | None = None,
        include_stale: bool = False,
    ) -> list[KnowledgeFact]:
        clauses: list[str] = []
        values: list[object] = []
        if stage is not None:
            clauses.append("stage = ?")
            values.append(stage)
        if fact_type is not None:
            clauses.append("fact_type = ?")
            values.append(fact_type)
        if not include_stale:
            clauses.append("stale = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM knowledge_facts {where} ORDER BY stage, fact_key",
                values,
            ).fetchall()
        return [self._knowledge_fact(row) for row in rows]

    def knowledge_fact(self, stage: str, fact_key: str) -> KnowledgeFact | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_facts WHERE stage = ? AND fact_key = ?",
                (stage, fact_key),
            ).fetchone()
        return self._knowledge_fact(row) if row is not None else None

    @staticmethod
    def _study_run(row: sqlite3.Row | None) -> StudyRun | None:
        if row is None:
            return None
        return StudyRun(
            id=int(row["id"]),
            stage=str(row["stage"]),
            phase=str(row["phase"]),
            benchmark_version=str(row["benchmark_version"]),
            question_set_digest=str(row["question_set_digest"]),
            score=float(row["score"]),
            correct=int(row["correct"]),
            total=int(row["total"]),
            latency_ms=int(row["latency_ms"]),
            resource_cost=dict(json.loads(row["resource_cost_json"] or "{}")),
            errors=tuple(json.loads(row["errors_json"] or "[]")),
            transferable_lessons=tuple(
                json.loads(row["transferable_lessons_json"] or "[]")
            ),
            created_at=str(row["created_at"]),
        )

    def record_study_run(
        self,
        *,
        stage: str,
        phase: str,
        benchmark_version: str,
        question_set_digest: str,
        score: float,
        correct: int,
        total: int,
        latency_ms: int,
        resource_cost: dict,
        errors: list[str] | tuple[str, ...],
        transferable_lessons: list[str] | tuple[str, ...] = (),
    ) -> StudyRun:
        if phase not in {"baseline", "post_study"}:
            raise ValueError(f"Unsupported study phase: {phase}")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO study_runs (
                    stage, phase, benchmark_version, question_set_digest,
                    score, correct, total, latency_ms, resource_cost_json,
                    errors_json, transferable_lessons_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage,
                    phase,
                    benchmark_version,
                    question_set_digest,
                    float(score),
                    int(correct),
                    int(total),
                    max(0, int(latency_ms)),
                    json.dumps(resource_cost, sort_keys=True),
                    json.dumps(list(errors)[:100]),
                    json.dumps(list(transferable_lessons)[:30]),
                    _now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM study_runs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        result = self._study_run(row)
        assert result is not None
        return result

    def latest_study_run(
        self, stage: str, phase: str | None = None
    ) -> StudyRun | None:
        clause = "AND phase = ?" if phase else ""
        values: tuple[object, ...] = (stage, phase) if phase else (stage,)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM study_runs WHERE stage = ? {clause} ORDER BY id DESC LIMIT 1",
                values,
            ).fetchone()
        return self._study_run(row)

    def update_curriculum_state(
        self,
        *,
        stage: str,
        status: str,
        baseline_run_id: int | None,
        latest_run_id: int | None,
        known_weak_areas: list[str] | tuple[str, ...],
        next_lesson: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO curriculum_stage_state (
                    stage, status, baseline_run_id, latest_run_id,
                    known_weak_areas_json, next_lesson, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    status = excluded.status,
                    baseline_run_id = COALESCE(excluded.baseline_run_id, curriculum_stage_state.baseline_run_id),
                    latest_run_id = COALESCE(excluded.latest_run_id, curriculum_stage_state.latest_run_id),
                    known_weak_areas_json = excluded.known_weak_areas_json,
                    next_lesson = excluded.next_lesson,
                    updated_at = excluded.updated_at
                """,
                (
                    stage,
                    status,
                    baseline_run_id,
                    latest_run_id,
                    json.dumps(list(known_weak_areas)[:30]),
                    next_lesson[:1000],
                    _now(),
                ),
            )

    def curriculum_state(self, stage: str) -> CurriculumStageState:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.*,
                       baseline.score AS baseline_score,
                       latest.score AS latest_score
                FROM curriculum_stage_state AS s
                LEFT JOIN study_runs AS baseline ON baseline.id = s.baseline_run_id
                LEFT JOIN study_runs AS latest ON latest.id = s.latest_run_id
                WHERE s.stage = ?
                """,
                (stage,),
            ).fetchone()
        if row is None:
            return CurriculumStageState(stage, "not_started", None, None, (), "", "")
        return CurriculumStageState(
            stage=str(row["stage"]),
            status=str(row["status"]),
            baseline_score=(
                float(row["baseline_score"])
                if row["baseline_score"] is not None
                else None
            ),
            latest_score=(
                float(row["latest_score"])
                if row["latest_score"] is not None
                else None
            ),
            known_weak_areas=tuple(json.loads(row["known_weak_areas_json"] or "[]")),
            next_lesson=str(row["next_lesson"]),
            updated_at=str(row["updated_at"]),
        )

    def curriculum_context(self) -> dict:
        """Bounded study evidence for capability discovery and status output."""
        stages = [self.curriculum_state(name) for name in ("self", "qwen", "python")]
        facts = self.knowledge_facts()
        return {
            "stages": [
                {
                    "stage": item.stage,
                    "status": item.status,
                    "baseline_score": item.baseline_score,
                    "latest_score": item.latest_score,
                    "known_weak_areas": list(item.known_weak_areas[:8]),
                    "next_lesson": item.next_lesson,
                }
                for item in stages
            ],
            "verified_fact_counts": {
                name: sum(1 for fact in facts if fact.stage == name)
                for name in ("self", "qwen", "python")
            },
        }

    def record_peer_model_comparison(
        self,
        *,
        subject_model: str,
        peer_model: str,
        subject_score: float,
        peer_score: float,
        subject_latency_ms: int,
        peer_latency_ms: int,
        resource_cost: dict,
        transferable_lessons: list[str] | tuple[str, ...],
    ) -> PeerModelComparison:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO peer_model_comparisons (
                    subject_model, peer_model, subject_score, peer_score,
                    subject_latency_ms, peer_latency_ms, resource_cost_json,
                    transferable_lessons_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_model[:300],
                    peer_model[:300],
                    float(subject_score),
                    float(peer_score),
                    max(0, int(subject_latency_ms)),
                    max(0, int(peer_latency_ms)),
                    json.dumps(resource_cost, sort_keys=True),
                    json.dumps(list(transferable_lessons)[:30]),
                    _now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM peer_model_comparisons WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        assert row is not None
        return PeerModelComparison(
            id=int(row["id"]),
            subject_model=str(row["subject_model"]),
            peer_model=str(row["peer_model"]),
            subject_score=float(row["subject_score"]),
            peer_score=float(row["peer_score"]),
            subject_latency_ms=int(row["subject_latency_ms"]),
            peer_latency_ms=int(row["peer_latency_ms"]),
            resource_cost=dict(json.loads(row["resource_cost_json"] or "{}")),
            transferable_lessons=tuple(
                json.loads(row["transferable_lessons_json"] or "[]")
            ),
            created_at=str(row["created_at"]),
        )

    def latest_peer_model_comparison(self) -> PeerModelComparison | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM peer_model_comparisons ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return PeerModelComparison(
            id=int(row["id"]),
            subject_model=str(row["subject_model"]),
            peer_model=str(row["peer_model"]),
            subject_score=float(row["subject_score"]),
            peer_score=float(row["peer_score"]),
            subject_latency_ms=int(row["subject_latency_ms"]),
            peer_latency_ms=int(row["peer_latency_ms"]),
            resource_cost=dict(json.loads(row["resource_cost_json"] or "{}")),
            transferable_lessons=tuple(
                json.loads(row["transferable_lessons_json"] or "[]")
            ),
            created_at=str(row["created_at"]),
        )

    def schema_columns(self) -> set[str]:
        """Exposed for diagnostics/tests that enforce the no-reasoning contract."""
        with self._connect() as connection:
            rows = []
            for table in (
                "development_cycles",
                "capability_map",
                "capability_experiments",
                "mission_frontiers",
                "knowledge_facts",
                "study_runs",
                "curriculum_stage_state",
                "peer_model_comparisons",
            ):
                rows.extend(connection.execute(f"PRAGMA table_info({table})").fetchall())
        return {str(row["name"]) for row in rows}

