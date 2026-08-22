from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from localpilot.evolution import normalize_evolution_task


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
    rejection_reason: str
    rejection_prior_validation_state: str
    rejection_pull_request_number: int | None
    rejected_at: str | None


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
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    rejection_prior_validation_state TEXT NOT NULL DEFAULT '',
                    rejection_pull_request_number INTEGER,
                    rejected_at TEXT,
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
            existing = connection.execute(
                "SELECT validation_state FROM development_cycles WHERE id = ?",
                (cycle_id,),
            ).fetchone()
            if existing is not None and existing["validation_state"] == "rejected_by_human":
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
        )

    def candidate_for_branch(self, branch: str) -> ManagedCandidate | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM development_cycles WHERE branch = ? ORDER BY id DESC LIMIT 1",
                (str(branch),),
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

    def schema_columns(self) -> set[str]:
        """Exposed for diagnostics/tests that enforce the no-reasoning contract."""
        with self._connect() as connection:
            rows = []
            for table in (
                "development_cycles",
                "capability_map",
                "capability_experiments",
                "mission_frontiers",
            ):
                rows.extend(connection.execute(f"PRAGMA table_info({table})").fetchall())
        return {str(row["name"]) for row in rows}

