from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
class HumanLesson:
    id: int
    lesson: str
    topic: str
    source: str
    confidence: float
    active: bool
    created_at: str


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
class DurableLearning:
    learning_key: str
    learning_type: str
    subject: str
    summary: str
    source_uri: str
    source_kind: str
    source_digest: str
    provenance: str
    confidence: float
    last_verified_at: str
    stale: bool


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    mode: str = "lexical"
    embedding_model: str = ""
    candidate_count: int = 0
    cache_hits: int = 0
    indexed_facts: int = 0
    semantic_candidates: int = 0
    latency_ms: int = 0
    error_type: str = ""


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

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_model: str = "",
        semantic_weight: float = 12.0,
        semantic_min_similarity: float = 0.2,
        embedding_batch_size: int = 64,
        embedding_migration_limit: int = 512,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.embedding_model = str(embedding_model).strip()
        self.semantic_weight = max(0.0, float(semantic_weight))
        self.semantic_min_similarity = max(
            -1.0, min(float(semantic_min_similarity), 1.0)
        )
        self.embedding_batch_size = max(1, min(int(embedding_batch_size), 256))
        self.embedding_migration_limit = max(
            1, min(int(embedding_migration_limit), 5000)
        )
        self._embedding_session_error = ""
        self.last_retrieval_diagnostics = RetrievalDiagnostics()
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
                CREATE TABLE IF NOT EXISTS human_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT 'general',
                    source TEXT NOT NULL DEFAULT 'owner',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS human_lessons_unique_idx
                    ON human_lessons(topic, lesson, source);
                CREATE INDEX IF NOT EXISTS human_lessons_active_idx
                    ON human_lessons(active, id DESC);
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
                CREATE TABLE IF NOT EXISTS durable_learnings (
                    learning_key TEXT PRIMARY KEY,
                    learning_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_digest TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_verified_at TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    CHECK (learning_type IN (
                        'heuristic', 'question', 'selfdev_hypothesis', 'opinion'
                    ))
                );
                CREATE INDEX IF NOT EXISTS durable_learning_source_idx
                    ON durable_learnings(source_uri, stale);
                CREATE INDEX IF NOT EXISTS durable_learning_type_idx
                    ON durable_learnings(learning_type, stale);
                CREATE TABLE IF NOT EXISTS knowledge_fact_embeddings (
                    stage TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(stage, fact_key, embedding_model),
                    FOREIGN KEY(stage, fact_key)
                        REFERENCES knowledge_facts(stage, fact_key)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS knowledge_embedding_model_idx
                    ON knowledge_fact_embeddings(embedding_model, stage);
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

    @staticmethod
    def _human_lesson(row: sqlite3.Row | None) -> HumanLesson | None:
        if row is None:
            return None
        return HumanLesson(
            id=int(row["id"]),
            lesson=str(row["lesson"]),
            topic=str(row["topic"]),
            source=str(row["source"]),
            confidence=float(row["confidence"]),
            active=bool(row["active"]),
            created_at=str(row["created_at"]),
        )

    def record_human_lesson(
        self,
        lesson: str,
        *,
        topic: str = "general",
        source: str = "owner",
        confidence: float = 1.0,
    ) -> HumanLesson:
        """Persist explicit human teaching, never an inferred transcript or hidden reasoning."""
        clean_lesson = " ".join(str(lesson).replace("\x00", " ").split())[:2000]
        clean_topic = " ".join(str(topic).replace("\x00", " ").split())[:120] or "general"
        clean_source = " ".join(str(source).replace("\x00", " ").split())[:80] or "owner"
        if not clean_lesson:
            raise ValueError("A human teaching must not be empty.")
        bounded_confidence = max(0.0, min(float(confidence), 1.0))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO human_lessons (
                    lesson, topic, source, confidence, active, created_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                """,
                (clean_lesson, clean_topic, clean_source, bounded_confidence, _now()),
            )
            connection.execute(
                """
                UPDATE human_lessons SET confidence = ?, active = 1
                WHERE lesson = ? AND topic = ? AND source = ?
                """,
                (bounded_confidence, clean_lesson, clean_topic, clean_source),
            )
            row = connection.execute(
                """
                SELECT * FROM human_lessons
                WHERE lesson = ? AND topic = ? AND source = ?
                """,
                (clean_lesson, clean_topic, clean_source),
            ).fetchone()
        result = self._human_lesson(row)
        if result is None:
            raise RuntimeError("Human teaching could not be persisted.")
        return result

    @staticmethod
    def _lesson_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_]{3,}", str(value).lower())
            if token not in {"the", "and", "for", "with", "that", "this", "from"}
        }

    def human_lessons(
        self,
        limit: int = 6,
        *,
        query: str | None = None,
    ) -> list[HumanLesson]:
        limit = max(0, min(int(limit), 50))
        if not limit:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM human_lessons
                WHERE active = 1
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        lessons = [self._human_lesson(row) for row in rows]
        items = [item for item in lessons if item is not None]
        query_tokens = self._lesson_tokens(query or "")
        if query_tokens:
            items.sort(
                key=lambda item: (
                    len(query_tokens & self._lesson_tokens(f"{item.topic} {item.lesson}")),
                    item.id,
                ),
                reverse=True,
            )
        return items[:limit]

    def human_lesson_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM human_lessons WHERE active = 1"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

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
        teachings = self.human_lessons(limit=limit)
        return {
            "capabilities": [dict(row) for row in capabilities],
            "frontiers": [dict(row) for row in frontiers],
            "experiments": [dict(row) for row in experiments],
            "recent_cycles": [dict(row) for row in cycles],
            "human_teachings": [
                {
                    "id": item.id,
                    "topic": item.topic,
                    "lesson": item.lesson,
                    "source": item.source,
                    "confidence": item.confidence,
                    "created_at": item.created_at,
                }
                for item in teachings
            ],
            "library_learnings": self.library_learning_context(limit=limit),
        }

    def reusable_lessons(
        self,
        limit: int = 6,
        *,
        query: str | None = None,
    ) -> list[str]:
        """Return bounded validated lessons, teachings, and relevant library learning."""
        limit = max(0, min(int(limit), 50))
        if not limit:
            return []
        human_limit = min(limit, max(1, (limit + 1) // 2))
        human = self.human_lessons(human_limit, query=query)
        remaining = max(0, limit - len(human))
        library_limit = max(1, (remaining + 1) // 2) if remaining else 0
        library = self.library_learning_context(query=query, limit=library_limit)
        remaining = max(0, remaining - len(library))
        rows = []
        if remaining:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT reusable_lesson FROM development_cycles
                    WHERE reusable_lesson != ''
                    ORDER BY id DESC LIMIT ?
                    """,
                    (remaining,),
                ).fetchall()
        result = [f"Human teaching [{item.topic}]: {item.lesson}" for item in human]
        result.extend(
            f"Library {item['learning_type']} [{item['source_uri']}]: {item['summary']}"
            for item in library
        )
        result.extend(str(row["reusable_lesson"]) for row in rows)
        return result

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

    @staticmethod
    def _durable_learning(row: sqlite3.Row) -> DurableLearning:
        return DurableLearning(
            learning_key=str(row["learning_key"]),
            learning_type=str(row["learning_type"]),
            subject=str(row["subject"]),
            summary=str(row["summary"]),
            source_uri=str(row["source_uri"]),
            source_kind=str(row["source_kind"]),
            source_digest=str(row["source_digest"]),
            provenance=str(row["provenance"]),
            confidence=float(row["confidence"]),
            last_verified_at=str(row["last_verified_at"]),
            stale=bool(row["stale"]),
        )

    def upsert_durable_learning(
        self,
        *,
        learning_key: str,
        learning_type: str,
        subject: str,
        summary: str,
        source_uri: str,
        source_kind: str,
        source_digest: str,
        provenance: str,
        confidence: float,
    ) -> None:
        """Persist one verified, explicitly non-factual learning record."""
        allowed = {"heuristic", "question", "selfdev_hypothesis", "opinion"}
        chosen_type = str(learning_type).strip().casefold()
        if chosen_type not in allowed:
            raise ValueError(f"Unsupported durable learning type: {learning_type}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO durable_learnings (
                    learning_key, learning_type, subject, summary, source_uri,
                    source_kind, source_digest, provenance, confidence,
                    last_verified_at, stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(learning_key) DO UPDATE SET
                    learning_type = excluded.learning_type,
                    subject = excluded.subject,
                    summary = excluded.summary,
                    source_uri = excluded.source_uri,
                    source_kind = excluded.source_kind,
                    source_digest = excluded.source_digest,
                    provenance = excluded.provenance,
                    confidence = excluded.confidence,
                    last_verified_at = excluded.last_verified_at,
                    stale = 0
                """,
                (
                    str(learning_key)[:500],
                    chosen_type,
                    str(subject)[:500],
                    str(summary)[:1200],
                    str(source_uri)[:1000],
                    str(source_kind)[:80],
                    str(source_digest)[:128],
                    str(provenance)[:500],
                    max(0.0, min(float(confidence), 1.0)),
                    _now(),
                ),
            )

    def invalidate_library_source(self, source_path: str, current_digest: str) -> int:
        """Stale every prior learning from revised bytes of one library source."""
        path = str(source_path).split("#", 1)[0]
        escaped = path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        prefix = f"library://{escaped}#%"
        with self._connect() as connection:
            fact_cursor = connection.execute(
                """
                UPDATE knowledge_facts SET stale = 1
                WHERE source_uri LIKE ? ESCAPE '\\'
                  AND source_digest != ? AND stale = 0
                """,
                (prefix, str(current_digest)),
            )
            learning_cursor = connection.execute(
                """
                UPDATE durable_learnings SET stale = 1
                WHERE source_uri LIKE ? ESCAPE '\\'
                  AND source_digest != ? AND stale = 0
                """,
                (prefix, str(current_digest)),
            )
            return int(fact_cursor.rowcount) + int(learning_cursor.rowcount)

    def durable_learnings(
        self,
        *,
        learning_type: str | None = None,
        include_stale: bool = False,
    ) -> list[DurableLearning]:
        clauses: list[str] = []
        values: list[object] = []
        if learning_type is not None:
            clauses.append("learning_type = ?")
            values.append(str(learning_type))
        if not include_stale:
            clauses.append("stale = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM durable_learnings {where} "
                "ORDER BY last_verified_at DESC, learning_key",
                values,
            ).fetchall()
        return [self._durable_learning(row) for row in rows]

    def search_durable_learnings(
        self,
        query: str,
        *,
        limit: int = 6,
        include_stale: bool = True,
    ) -> list[DurableLearning]:
        """Return bounded lexical matches without collapsing them into facts."""
        limit = max(0, min(int(limit), 12))
        query_tokens = self._knowledge_tokens(query)
        if not limit or not query_tokens:
            return []
        scored: list[tuple[int, float, str, DurableLearning]] = []
        for item in self.durable_learnings(include_stale=include_stale):
            fields = self._knowledge_tokens(
                " ".join((item.learning_type, item.subject, item.summary, item.provenance))
            )
            overlap = len(query_tokens & fields)
            if overlap < 2 and item.subject.casefold() not in str(query).casefold():
                continue
            scored.append(
                (
                    overlap - (2 if item.stale else 0),
                    item.confidence,
                    item.last_verified_at,
                    item,
                )
            )
        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [row[3] for row in scored[:limit]]

    def library_learning_context(
        self,
        *,
        query: str | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Expose concise current library learning to self-development discovery."""
        limit = max(0, min(int(limit), 20))
        if not limit:
            return []
        query_text = str(query or "library learning self development capability")
        query_tokens = self._knowledge_tokens(query_text)
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for fact in self.knowledge_facts(stage="library", include_stale=False):
            tokens = self._knowledge_tokens(
                " ".join((fact.fact_type, fact.subject, fact.summary))
            )
            overlap = len(query_tokens & tokens) if query else 1
            if query and overlap < 1:
                continue
            rows.append(
                (
                    overlap,
                    fact.last_verified_at,
                    {
                        "learning_type": fact.fact_type.removeprefix("library_"),
                        "subject": fact.subject,
                        "summary": fact.summary,
                        "source_uri": fact.source_uri,
                        "source_digest": fact.source_digest,
                        "confidence": fact.confidence,
                        "objective_fact": True,
                    },
                )
            )
        for item in self.durable_learnings(include_stale=False):
            tokens = self._knowledge_tokens(
                " ".join((item.learning_type, item.subject, item.summary))
            )
            overlap = len(query_tokens & tokens) if query else 1
            if query and overlap < 1:
                continue
            rows.append(
                (
                    overlap,
                    item.last_verified_at,
                    {
                        "learning_type": item.learning_type,
                        "subject": item.subject,
                        "summary": item.summary,
                        "source_uri": item.source_uri,
                        "source_digest": item.source_digest,
                        "confidence": item.confidence,
                        "objective_fact": False,
                    },
                )
            )
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [row[2] for row in rows[:limit]]

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

    @staticmethod
    def _knowledge_tokens(value: str) -> set[str]:
        """Tokenize bounded fact metadata for deterministic in-process retrieval."""
        stop_words = {
            "about", "after", "again", "also", "before", "being", "could",
            "from", "have", "into", "localpilot", "more", "only", "should",
            "that", "their", "there", "these", "they", "this", "using",
            "what", "when", "where", "which", "with", "would", "your",
        }
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
        text = re.sub(r"[_:./\\>\-]+", " ", text).lower()
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", text)
            if token not in stop_words
        }
        expansions = {
            "architecture": {"owner", "symbol"},
            "candidate": {"confinement", "reviewer"},
            "dependency": {"pyproject", "toml"},
            "developer": {"development", "selfdev"},
            "development": {"developer", "selfdev"},
            "durable": {"learning", "memory"},
            "learning": {"memory", "study"},
            "memory": {"learning", "study"},
            "operator": {"agent"},
            "safety": {"confinement", "governor", "policy", "reviewer"},
        }
        for token in tuple(tokens):
            tokens.update(expansions.get(token, ()))
        return tokens

    @staticmethod
    def _embedding_document(fact: KnowledgeFact) -> str:
        """Stable semantic text; provenance remains on the returned fact, not in vectors."""
        return json.dumps(
            {
                "stage": fact.stage,
                "fact_type": fact.fact_type,
                "subject": fact.subject,
                "summary": fact.summary,
                "relationships": list(fact.relationships),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _embedding_digest(cls, fact: KnowledgeFact) -> str:
        return hashlib.sha256(
            cls._embedding_document(fact).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
        if not left or len(left) != len(right):
            return None
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return None
        return max(-1.0, min(dot / (left_norm * right_norm), 1.0))

    @staticmethod
    def _validated_embeddings(
        values: Any,
        expected: int,
    ) -> list[list[float]]:
        if not isinstance(values, (list, tuple)) or len(values) != expected:
            raise ValueError("Embedding provider returned an unexpected vector count.")
        vectors: list[list[float]] = []
        dimensions: int | None = None
        for value in values:
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("Embedding provider returned an empty vector.")
            vector = [float(item) for item in value]
            if not all(math.isfinite(item) for item in vector):
                raise ValueError("Embedding provider returned a non-finite vector.")
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("Embedding provider returned inconsistent dimensions.")
            vectors.append(vector)
        return vectors

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_provider is None or not self.embedding_model:
            raise RuntimeError("Semantic retrieval is not configured.")
        return self._validated_embeddings(self.embedding_provider(texts), len(texts))

    def _cached_fact_embeddings(
        self,
        facts: list[KnowledgeFact],
    ) -> tuple[dict[tuple[str, str], list[float]], list[KnowledgeFact]]:
        wanted = {(fact.stage, fact.fact_key): fact for fact in facts}
        cached: dict[tuple[str, str], list[float]] = {}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT stage, fact_key, content_digest, dimensions, embedding_json
                FROM knowledge_fact_embeddings
                WHERE embedding_model = ?
                """,
                (self.embedding_model,),
            ).fetchall()
        for row in rows:
            key = (str(row["stage"]), str(row["fact_key"]))
            fact = wanted.get(key)
            if fact is None or str(row["content_digest"]) != self._embedding_digest(fact):
                continue
            try:
                vector = [float(item) for item in json.loads(row["embedding_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                not vector
                or len(vector) != int(row["dimensions"])
                or not all(math.isfinite(item) for item in vector)
            ):
                continue
            cached[key] = vector
        missing = [
            fact for fact in facts if (fact.stage, fact.fact_key) not in cached
        ]
        return cached, missing

    def _index_fact_embeddings(
        self,
        facts: list[KnowledgeFact],
        cached: dict[tuple[str, str], list[float]],
    ) -> int:
        indexed = 0
        pending = facts[: self.embedding_migration_limit]
        for offset in range(0, len(pending), self.embedding_batch_size):
            batch = pending[offset : offset + self.embedding_batch_size]
            vectors = self._embed_texts(
                [self._embedding_document(fact) for fact in batch]
            )
            with self._connect() as connection:
                for fact, vector in zip(batch, vectors):
                    connection.execute(
                        """
                        INSERT INTO knowledge_fact_embeddings (
                            stage, fact_key, embedding_model, content_digest,
                            dimensions, embedding_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(stage, fact_key, embedding_model) DO UPDATE SET
                            content_digest = excluded.content_digest,
                            dimensions = excluded.dimensions,
                            embedding_json = excluded.embedding_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            fact.stage,
                            fact.fact_key,
                            self.embedding_model,
                            self._embedding_digest(fact),
                            len(vector),
                            json.dumps(vector, separators=(",", ":")),
                            _now(),
                        ),
                    )
                    cached[(fact.stage, fact.fact_key)] = vector
                    indexed += 1
        return indexed

    def _semantic_scores(
        self,
        query: str,
        facts: list[KnowledgeFact],
    ) -> tuple[dict[tuple[str, str], float], int, int, str]:
        if (
            self.embedding_provider is None
            or not self.embedding_model
            or self._embedding_session_error
        ):
            return {}, 0, 0, self._embedding_session_error
        try:
            query_vector = self._embed_texts([query])[0]
            cached, missing = self._cached_fact_embeddings(facts)
            cache_hits = len(cached)
            indexed = self._index_fact_embeddings(missing, cached) if missing else 0
            scores: dict[tuple[str, str], float] = {}
            for fact in facts:
                vector = cached.get((fact.stage, fact.fact_key))
                if vector is None:
                    continue
                similarity = self._cosine_similarity(query_vector, vector)
                if similarity is not None:
                    scores[(fact.stage, fact.fact_key)] = similarity
            return scores, cache_hits, indexed, ""
        except Exception as exc:
            self._embedding_session_error = type(exc).__name__
            return {}, 0, 0, self._embedding_session_error

    def knowledge_embedding_count(self, model: str | None = None) -> int:
        """Expose migration/index coverage without storing or returning query vectors."""
        chosen = str(model or self.embedding_model).strip()
        if not chosen:
            return 0
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM knowledge_fact_embeddings WHERE embedding_model = ?",
                (chosen,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def search_knowledge_facts(
        self,
        query: str,
        *,
        stage: str | None = None,
        limit: int = 8,
        include_stale: bool = True,
    ) -> list[KnowledgeFact]:
        """Return bounded hybrid lexical/semantic results with lexical fallback."""
        started = time.monotonic()
        limit = max(0, min(int(limit), 12))
        query_tokens = self._knowledge_tokens(query)
        if not limit or not query_tokens:
            self.last_retrieval_diagnostics = RetrievalDiagnostics(
                latency_ms=int((time.monotonic() - started) * 1000)
            )
            return []

        query_text = " ".join(str(query).lower().split())
        requests_test_evidence = bool(
            query_tokens & {"pytest", "regression", "test", "tests"}
        )
        scored: list[tuple[float, int, KnowledgeFact]] = []
        candidates: list[KnowledgeFact] = []
        for fact in self.knowledge_facts(stage=stage, include_stale=include_stale):
            if fact.source_uri.startswith("repo://tests/") and not requests_test_evidence:
                continue
            candidates.append(fact)
            fields = {
                "key": self._knowledge_tokens(fact.fact_key),
                "type": self._knowledge_tokens(fact.fact_type),
                "subject": self._knowledge_tokens(fact.subject),
                "summary": self._knowledge_tokens(fact.summary),
                "relationships": self._knowledge_tokens(" ".join(fact.relationships)),
                "stage": self._knowledge_tokens(fact.stage),
            }
            matched = query_tokens & set().union(*fields.values())
            exact_subject = bool(
                fact.subject.strip()
                and fact.subject.strip().lower() in query_text
            )
            explicit_stage = fact.stage.lower() in query_tokens
            if len(matched) < 2 and not exact_subject and not explicit_stage:
                continue
            quality = {
                "owner": 100,
                "symbol": 5,
                "verified_lesson": 4,
                "config_field": 3,
                "file": 2,
                "test_contract": 1,
                "call_relationship": -1,
                "import": -4,
            }.get(fact.fact_type, 0)
            source_quality = (
                50 if fact.source_uri == "repo://ARCHITECTURE.md"
                else -7 if fact.source_uri.startswith("repo://tests/")
                else 0
            )
            source_hint = (
                100
                if "dependency" in query_tokens and "pyproject" in fields["subject"]
                else 0
            )
            score = (
                7 * len(query_tokens & fields["subject"])
                + 5 * len(query_tokens & fields["key"])
                + 4 * len(query_tokens & fields["type"])
                + 3 * len(query_tokens & fields["summary"])
                + 2 * len(query_tokens & fields["relationships"])
                + 2 * len(query_tokens & fields["stage"])
                + (10 if exact_subject else 0)
                + (120 if explicit_stage else 0)
                + float(fact.confidence)
                + quality
                + source_quality
                + source_hint
                - (4 if fact.stale else 0)
            )
            scored.append((score, len(matched), fact))

        semantic_scores, cache_hits, indexed_facts, semantic_error = (
            self._semantic_scores(query, candidates)
        )
        semantic_candidates = 0
        if semantic_scores:
            semantic_ranks = {
                key: rank
                for rank, (key, similarity) in enumerate(
                    sorted(
                        semantic_scores.items(),
                        key=lambda item: (item[1], item[0]),
                        reverse=True,
                    ),
                    start=1,
                )
                if similarity >= self.semantic_min_similarity
            }

            def semantic_bonus(key: tuple[str, str]) -> float:
                similarity = semantic_scores.get(key)
                rank = semantic_ranks.get(key)
                if similarity is None or rank is None:
                    return 0.0
                span = max(0.0001, 1.0 - self.semantic_min_similarity)
                normalized = max(
                    0.0,
                    (similarity - self.semantic_min_similarity) / span,
                )
                return self.semantic_weight * (1.0 / rank + normalized)

            rescored: list[tuple[float, int, KnowledgeFact]] = []
            lexical_keys: set[tuple[str, str]] = set()
            for score, matched_count, fact in scored:
                key = (fact.stage, fact.fact_key)
                lexical_keys.add(key)
                rescored.append((score + semantic_bonus(key), matched_count, fact))
            for fact in candidates:
                key = (fact.stage, fact.fact_key)
                similarity = semantic_scores.get(key)
                if key in lexical_keys or similarity is None:
                    continue
                if similarity < self.semantic_min_similarity:
                    continue
                semantic_candidates += 1
                score = (
                    semantic_bonus(key)
                    + float(fact.confidence)
                    - (4 if fact.stale else 0)
                )
                rescored.append((score, 0, fact))
            scored = rescored

        mode = "lexical"
        if semantic_scores:
            mode = "hybrid"
        elif semantic_error:
            mode = "lexical_fallback"
        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            mode=mode,
            embedding_model=self.embedding_model if self.embedding_provider else "",
            candidate_count=len(candidates),
            cache_hits=cache_hits,
            indexed_facts=indexed_facts,
            semantic_candidates=semantic_candidates,
            latency_ms=int((time.monotonic() - started) * 1000),
            error_type=semantic_error,
        )

        scored.sort(
            key=lambda item: (
                item[0], item[1], item[2].confidence,
                item[2].last_verified_at, item[2].fact_key,
            ),
            reverse=True,
        )

        # Prefer distinct subjects first so a broad question gets coverage instead
        # of eight near-identical symbols from one module.
        selected: list[KnowledgeFact] = []
        deferred: list[KnowledgeFact] = []
        subjects: set[tuple[str, str]] = set()
        sources: set[str] = set()
        source_diversity_target = min(limit, 4)
        for _, _, fact in scored:
            subject_key = (fact.stage, fact.subject.lower())
            if (
                len(selected) >= source_diversity_target
                or subject_key in subjects
                or fact.source_uri in sources
            ):
                deferred.append(fact)
                continue
            selected.append(fact)
            subjects.add(subject_key)
            sources.add(fact.source_uri)
            if len(selected) == limit:
                return selected
        still_deferred: list[KnowledgeFact] = []
        for fact in deferred:
            subject_key = (fact.stage, fact.subject.lower())
            if subject_key in subjects:
                still_deferred.append(fact)
                continue
            selected.append(fact)
            subjects.add(subject_key)
            if len(selected) == limit:
                return selected
        for fact in still_deferred:
            selected.append(fact)
            if len(selected) == limit:
                break
        return selected

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
                "durable_learnings",
                "knowledge_fact_embeddings",
                "study_runs",
                "curriculum_stage_state",
                "peer_model_comparisons",
            ):
                rows.extend(connection.execute(f"PRAGMA table_info({table})").fetchall())
        return {str(row["name"]) for row in rows}

