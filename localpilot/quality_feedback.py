"""Human-verified outcomes and optional positive coaching.

No model is authorized to grade itself. This separate append-only store never
updates model weights or imports protected evaluation responses.
"""
from __future__ import annotations

import json
import re
import sqlite3
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ("correctness", "evidence", "safety", "efficiency", "initiative")
_BENCHMARK_PATTERN = re.compile(
    r"(?:held.?out|benchmark|eval[_/-]|paired[_/-]|scaffold[_/-]sanity|"
    r"lp-paired|test[_/-]fixture|grading[_/-])", re.IGNORECASE
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#-]{2,191}$")
_TOPIC = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{1,47}$")


@dataclass(frozen=True, slots=True)
class FeedbackEntry:
    id: int
    task_id: str
    model: str
    topic: str
    scores: dict[str, int]
    outcome: str
    evidence_ref: str
    review_note: str
    created_at: str

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(CATEGORIES)

    @property
    def eligible_for_coaching(self) -> bool:
        return (
            self.outcome == "verified_success"
            and self.mean_score >= 3.4
            and all(self.scores[dimension] >= 3 for dimension in
                    ("correctness", "evidence", "safety"))
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_text(name: str, value: str, *, minimum: int = 1, maximum: int = 350) -> str:
    text = str(value).strip()
    if not minimum <= len(text) <= maximum or any(ord(c) < 32 for c in text):
        raise ValueError(f"{name} must be {minimum}–{maximum} printable characters")
    return text


class QualityFeedbackStore:
    """Application-level append-only audit ledger, isolated from LearningMemory.

    This API requires a human attestation on every write. It does not claim
    cryptographic proof of reviewer identity; database administrators retain
    ultimate control of local files.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK (kind IN ('rating','approve','revoke')),
                    subject_id INTEGER,
                    task_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS feedback_events_task
                    ON feedback_events(task_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS feedback_one_rating_per_task
                    ON feedback_events(task_id) WHERE kind='rating';
                CREATE INDEX IF NOT EXISTS feedback_events_subject
                    ON feedback_events(subject_id, id);

                -- Observed CI/merge evidence is not a human quality grade and
                -- cannot unlock positive coaching by itself.
                CREATE TABLE IF NOT EXISTS feedback_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL CHECK(source='local_development_cycles'),
                    source_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source,source_id)
                );
                CREATE TRIGGER IF NOT EXISTS feedback_observations_no_update
                    BEFORE UPDATE ON feedback_observations
                    BEGIN SELECT RAISE(ABORT, 'quality feedback observations are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS feedback_observations_no_delete
                    BEFORE DELETE ON feedback_observations
                    BEGIN SELECT RAISE(ABORT, 'quality feedback observations are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS feedback_events_no_update
                    BEFORE UPDATE ON feedback_events
                    BEGIN SELECT RAISE(ABORT, 'quality feedback events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS feedback_events_no_delete
                    BEFORE DELETE ON feedback_events
                    BEGIN SELECT RAISE(ABORT, 'quality feedback events are append-only'); END;
                """
            )

    def _connection(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        return db

    @staticmethod
    def _check_production_id(value: str) -> str:
        token = str(value).strip()
        if not _TOKEN.fullmatch(token) or _BENCHMARK_PATTERN.search(token):
            raise ValueError("Use a production task ID, never a benchmark/evaluation identifier")
        return token

    def record(
        self, *,
        task_id: str,
        model: str,
        topic: str,
        scores: dict[str, int],
        outcome: str,
        evidence_ref: str,
        review_note: str,
        human_attested: bool = False,
    ) -> FeedbackEntry:
        if human_attested is not True:
            raise PermissionError("An independent human must explicitly attest to verifying this outcome")
        task_id = self._check_production_id(task_id)
        model = _short_text("model", model, maximum=100)
        topic = _short_text("topic", topic, maximum=48)
        if not _TOPIC.fullmatch(topic):
            raise ValueError("topic must be a short descriptive category")
        if outcome not in ("verified_success", "partial", "failed"):
            raise ValueError("outcome must be verified_success, partial, or failed")
        if set(scores) != set(CATEGORIES):
            raise ValueError(f"scores must contain exactly: {', '.join(CATEGORIES)}")
        if any(type(scores[key]) is not int or not 0 <= scores[key] <= 4 for key in CATEGORIES):
            raise ValueError("Each dimension must be an integer between 0 and 4")
        evidence_ref = _short_text("evidence_ref", evidence_ref, minimum=6, maximum=192)
        if not _TOKEN.fullmatch(evidence_ref):
            raise ValueError("evidence_ref must be an opaque, non-sensitive evidence ID or digest")
        review_note = _short_text("review_note", review_note, minimum=12, maximum=350)
        payload = {
            "model": model, "topic": topic, "scores": scores, "outcome": outcome,
            "evidence_ref": evidence_ref, "review_note": review_note,
            "reviewer": "owner_attested", "origin": "production",
        }
        with self._connection() as connection:
            if connection.execute(
                "SELECT 1 FROM feedback_events WHERE kind='rating' AND task_id=?",
                (task_id,),
            ).fetchone():
                raise ValueError("Task already has a rating; do not overwrite or double-count")
            cursor = connection.execute(
                "INSERT INTO feedback_events(kind,subject_id,task_id,payload_json,created_at) "
                "VALUES ('rating',NULL,?,?,?)",
                (task_id, json.dumps(payload, sort_keys=True), _utc_now()),
            )
            entry_id = int(cursor.lastrowid)
        entry = self.get(entry_id)
        assert entry is not None
        return entry

    def get(self, rating_id: int) -> FeedbackEntry | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_events WHERE id=? AND kind='rating'", (rating_id,)
            ).fetchone()
        if not row:
            return None
        info = json.loads(row["payload_json"])
        return FeedbackEntry(
            int(row["id"]), str(row["task_id"]), info["model"], info["topic"],
            info["scores"], info["outcome"], info["evidence_ref"],
            info["review_note"], str(row["created_at"]),
        )

    def approve_coaching(self, rating_id: int, *, lesson: str, human_attested: bool = False) -> int:
        if human_attested is not True:
            raise PermissionError("Coaching requires a separate, explicit human approval")
        entry = self.get(rating_id)
        if not entry:
            raise ValueError("Unknown feedback rating")
        if not entry.eligible_for_coaching:
            raise ValueError("Only independently verified strong production outcomes are eligible")
        lesson = _short_text("coaching lesson", lesson, minimum=24, maximum=300)
        # The lesson is general procedural guidance, not a purported new fact.
        if _BENCHMARK_PATTERN.search(lesson):
            raise ValueError("Protected benchmark references cannot become coaching")
        return self._append("approve", entry, {"lesson": lesson, "reviewer": "owner_attested"})

    def revoke_coaching(self, rating_id: int, *, reason: str, human_attested: bool = False) -> int:
        if human_attested is not True:
            raise PermissionError("Revocation must be explicitly authorized by the human owner")
        entry = self.get(rating_id)
        if not entry:
            raise ValueError("Unknown feedback rating")
        return self._append(
            "revoke", entry, {"reason": _short_text("reason", reason, minimum=8, maximum=200)}
        )

    def _append(self, kind: str, entry: FeedbackEntry, payload: dict) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO feedback_events(kind,subject_id,task_id,payload_json,created_at) "
                "VALUES (?,?,?,?,?)",
                (kind, entry.id, entry.task_id, json.dumps(payload, sort_keys=True), _utc_now()),
            )
            return int(cursor.lastrowid)

    def approved_lessons(self, *, limit: int = 3) -> list[tuple[str, str]]:
        """Return only currently approved, unrevoked production coaching."""
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT r.id AS rating_id, r.payload_json AS rating_payload,
                       a.kind AS latest_kind, a.payload_json AS latest_payload
                FROM feedback_events r
                JOIN feedback_events a
                  ON a.id = (
                      SELECT MAX(x.id) FROM feedback_events x
                      WHERE x.subject_id = r.id AND x.kind IN ('approve','revoke')
                  )
                WHERE r.kind='rating'
                ORDER BY a.id DESC
                """
            ).fetchall()
        output: list[tuple[str, str]] = []
        for row in rows:
            if row["latest_kind"] != "approve":
                continue
            original = json.loads(row["rating_payload"])
            if original.get("origin") != "production":
                continue
            entry = self.get(int(row["rating_id"]))
            if entry is None or not entry.eligible_for_coaching:
                continue
            guidance = json.loads(row["latest_payload"])["lesson"]
            output.append((entry.topic, guidance))
            if len(output) >= limit:
                break
        return output

    def sync_verified_development_outcomes(self, learning_db: str | Path) -> int:
        """Capture objective CI/merge evidence automatically, *without* assigning a score.

        The learning ledger is maintained by LocalPilot's GitHub validation
        workflow. A merged PR with passed checks establishes a delivery event,
        not correctness, safety, generalization or a reward-worthy outcome.
        Protected benchmark IDs are excluded; recorded observations are never
        included by approved_lessons() or used to trigger model promotion.
        """
        source_path = Path(learning_db).resolve()
        if not source_path.is_file():
            return 0
        # Open without write access: the feedback observer cannot mutate the
        # existing training/learning lifecycle or its provenance.
        uri = "file:" + quote(str(source_path).replace("\\", "/"), safe="/:") + "?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as source:
                source.row_factory = sqlite3.Row
                rows = source.execute(
                    """
                    SELECT id, task_id, branch, developer_model, pull_request_url,
                           checks_passed, pushed, validation_state, merged,
                           local_repair_attempts, write_integrity_failure,
                           rejection_reason
                    FROM development_cycles
                    WHERE merged=1 AND checks_passed=1 AND pushed=1
                      AND validation_state='passed' AND pull_request_url IS NOT NULL
                    ORDER BY id
                    """
                ).fetchall()
        except sqlite3.DatabaseError:
            # Older or partial databases are not evidence; do not guess.
            return 0
        added = 0
        with self._connection() as destination:
            for row in rows:
                try:
                    task_id = self._check_production_id(row["task_id"])
                except ValueError:
                    continue
                link = str(row["pull_request_url"] or "")
                if not re.fullmatch(
                    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[1-9][0-9]*",
                    link,
                ):
                    continue
                # Prior local write-integrity/rejection warnings aren't silently
                # converted into a positive outcome.
                if str(row["write_integrity_failure"] or "").strip() or str(row["rejection_reason"] or "").strip():
                    continue
                payload = {
                    "evidence_type": "ci_passed_and_pr_merged",
                    "model": str(row["developer_model"] or "")[:100],
                    "branch": str(row["branch"] or "")[:192],
                    "pull_request_url": link,
                    "local_repair_attempts": int(row["local_repair_attempts"] or 0),
                    "checks_passed": True,
                    "merged": True,
                    "scope": "delivery_only_not_quality_score",
                }
                cursor = destination.execute(
                    """
                    INSERT OR IGNORE INTO feedback_observations(
                        source,source_id,task_id,payload_json,created_at
                    ) VALUES ('local_development_cycles',?,?,?,?)
                    """,
                    (int(row["id"]), task_id, json.dumps(payload, sort_keys=True), _utc_now()),
                )
                added += int(cursor.rowcount == 1)
        return added

    def recent_observations(self, *, limit: int = 20) -> list[dict]:
        """Read objective observations, separate from human-assessed ratings."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,source_id,task_id,payload_json,created_at "
                "FROM feedback_observations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": int(row["id"]), "source_id": int(row["source_id"]),
                "task_id": str(row["task_id"]), "created_at": str(row["created_at"]),
                **json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def recent(self, *, limit: int = 10) -> list[FeedbackEntry]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM feedback_events WHERE kind='rating' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [e for row in rows if (e := self.get(int(row["id"]))) is not None]
