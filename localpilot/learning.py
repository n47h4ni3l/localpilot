from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS cycles_task_idx
                    ON development_cycles(task_id, id DESC);
                CREATE INDEX IF NOT EXISTS cycles_pending_idx
                    ON development_cycles(merged, validation_state, pushed);
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
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)

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
                WHERE pushed = 1 AND NOT (merged = 1 AND validation_state = 'passed')
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
                       local_repair_attempts, status
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

    def pending_task_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT task_id FROM development_cycles
                WHERE (
                    pushed = 1
                    AND NOT (merged = 1 AND validation_state = 'passed')
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

    def schema_columns(self) -> set[str]:
        """Exposed for diagnostics/tests that enforce the no-reasoning contract."""
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(development_cycles)").fetchall()
        return {str(row["name"]) for row in rows}

