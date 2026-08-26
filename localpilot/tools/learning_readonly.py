import os
import sqlite3
from pathlib import Path

from localpilot.config import load_config


class LearningMemoryReader:
    """Bounded read-only inspection of LocalPilot's durable knowledge facts."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        configured = os.environ.get("LOCALPILOT_CONFIG")
        config_path = Path(configured) if configured else self.project_root / "localpilot.toml"
        self.config = load_config(config_path)
        data_dir = (self.project_root / self.config.agent.data_dir).resolve()
        database = (data_dir / self.config.selfdev.learning_database).resolve()
        try:
            database.relative_to(data_dir)
        except ValueError as exc:
            raise ValueError("Learning database must remain inside the configured data directory.") from exc
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        if not self.database.is_file():
            raise FileNotFoundError("LocalPilot durable learning database does not exist yet.")
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _clean_filter(value: str, limit: int) -> str:
        return " ".join(str(value).replace("\x00", " ").split())[:limit]

    def get_learning_memory_summary(
        self,
        stage: str = "",
        fact_type: str = "",
        sample_limit: int = 8,
        source_limit: int = 8,
    ) -> dict:
        """Summarize current/stale durable facts without exposing raw SQL or mutating memory."""
        clean_stage = self._clean_filter(stage, 40)
        clean_type = self._clean_filter(fact_type, 80)
        sample_limit = max(0, min(int(sample_limit), 12))
        source_limit = max(0, min(int(source_limit), 12))

        clauses = []
        values: list[object] = []
        if clean_stage:
            clauses.append("stage = ?")
            values.append(clean_stage)
        if clean_type:
            clauses.append("fact_type = ?")
            values.append(clean_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as connection:
            total_row = connection.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN stale = 0 THEN 1 ELSE 0 END) AS current_count,
                       SUM(CASE WHEN stale = 1 THEN 1 ELSE 0 END) AS stale_count
                FROM knowledge_facts {where}
                """,
                values,
            ).fetchone()
            type_rows = connection.execute(
                f"""
                SELECT fact_type,
                       COUNT(*) AS total,
                       SUM(CASE WHEN stale = 0 THEN 1 ELSE 0 END) AS current_count,
                       SUM(CASE WHEN stale = 1 THEN 1 ELSE 0 END) AS stale_count
                FROM knowledge_facts {where}
                GROUP BY fact_type
                ORDER BY total DESC, fact_type
                LIMIT 40
                """,
                values,
            ).fetchall()
            stage_rows = connection.execute(
                f"""
                SELECT stage,
                       COUNT(*) AS total,
                       SUM(CASE WHEN stale = 0 THEN 1 ELSE 0 END) AS current_count,
                       SUM(CASE WHEN stale = 1 THEN 1 ELSE 0 END) AS stale_count
                FROM knowledge_facts {where}
                GROUP BY stage
                ORDER BY total DESC, stage
                LIMIT 20
                """,
                values,
            ).fetchall()
            source_rows = []
            if source_limit:
                source_rows = connection.execute(
                    f"""
                    SELECT source_uri, source_kind,
                           COUNT(*) AS total,
                           SUM(CASE WHEN stale = 0 THEN 1 ELSE 0 END) AS current_count,
                           SUM(CASE WHEN stale = 1 THEN 1 ELSE 0 END) AS stale_count
                    FROM knowledge_facts {where}
                    GROUP BY source_uri, source_kind
                    ORDER BY stale_count DESC, total DESC, source_uri
                    LIMIT ?
                    """,
                    [*values, source_limit],
                ).fetchall()
            stale_rows = []
            if sample_limit:
                stale_where = f"{where} {'AND' if where else 'WHERE'} stale = 1"
                stale_rows = connection.execute(
                    f"""
                    SELECT stage, fact_key, fact_type, subject, source_uri, source_kind,
                           source_digest, confidence, last_verified_at
                    FROM knowledge_facts {stale_where}
                    ORDER BY last_verified_at DESC, stage, fact_key
                    LIMIT ?
                    """,
                    [*values, sample_limit],
                ).fetchall()

        total = int(total_row["total"] or 0) if total_row is not None else 0
        current = int(total_row["current_count"] or 0) if total_row is not None else 0
        stale = int(total_row["stale_count"] or 0) if total_row is not None else 0
        return {
            "filters": {
                "stage": clean_stage or None,
                "fact_type": clean_type or None,
            },
            "counts": {
                "total": total,
                "current": current,
                "stale": stale,
            },
            "by_stage": [
                {
                    "stage": str(row["stage"]),
                    "total": int(row["total"] or 0),
                    "current": int(row["current_count"] or 0),
                    "stale": int(row["stale_count"] or 0),
                }
                for row in stage_rows
            ],
            "by_fact_type": [
                {
                    "fact_type": str(row["fact_type"]),
                    "total": int(row["total"] or 0),
                    "current": int(row["current_count"] or 0),
                    "stale": int(row["stale_count"] or 0),
                }
                for row in type_rows
            ],
            "top_sources": [
                {
                    "source_uri": str(row["source_uri"])[:1000],
                    "source_kind": str(row["source_kind"])[:80],
                    "total": int(row["total"] or 0),
                    "current": int(row["current_count"] or 0),
                    "stale": int(row["stale_count"] or 0),
                }
                for row in source_rows
            ],
            "stale_samples": [
                {
                    "stage": str(row["stage"])[:40],
                    "fact_key": str(row["fact_key"])[:500],
                    "fact_type": str(row["fact_type"])[:80],
                    "subject": str(row["subject"])[:500],
                    "source_uri": str(row["source_uri"])[:1000],
                    "source_kind": str(row["source_kind"])[:80],
                    "source_digest": str(row["source_digest"])[:128],
                    "confidence": float(row["confidence"]),
                    "last_verified_at": str(row["last_verified_at"])[:80],
                }
                for row in stale_rows
            ],
            "staleness": {
                "rule": "A knowledge fact is marked stale when its authoritative source URI is observed with a different source digest.",
                "invalidation_history_available": False,
                "history_note": "The current schema stores stale state and last verification time, but not a separate invalidated_at timestamp or per-row invalidation event, so exact invalidation chronology cannot be reconstructed from the learning database alone.",
            },
        }
