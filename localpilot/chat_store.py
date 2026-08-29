from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conversation_title(content: str, *, limit: int = 64) -> str:
    """Derive a stable, private title locally from the first user message."""
    normalized = " ".join(str(content).split())
    if not normalized:
        return "New conversation"
    if len(normalized) <= limit:
        return normalized
    shortened = normalized[: limit + 1].rsplit(" ", 1)[0].rstrip(".,;:!? -")
    if not shortened:
        shortened = normalized[:limit].rstrip()
    return shortened + "…"


class ChatStore:
    """Durable UI/session history kept separate from LocalPilot learning memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('streaming', 'complete', 'error')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_messages_session_id
                    ON chat_messages(session_id, id);
                CREATE TABLE IF NOT EXISTS chat_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_events_session_id
                    ON chat_events(session_id, id);
                """
            )

    def create_session(self, title: str = "New conversation") -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title.strip()[:120] or "New conversation", timestamp, timestamp),
            )
        return self.session(session_id)

    def session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown chat session: {session_id}")
        return dict(row)

    def sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_or_create_session(self) -> dict[str, Any]:
        sessions = self.sessions(limit=1)
        return sessions[0] if sessions else self.create_session()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        status: str = "complete",
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat history accepts only visible user and assistant messages")
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_messages(session_id, role, content, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, str(content), status, timestamp, timestamp),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ?, "
                "title = CASE WHEN ? = 'user' AND title = 'New conversation' THEN ? ELSE title END "
                "WHERE id = ?",
                (timestamp, role, _conversation_title(content), session_id),
            )
            message_id = int(cursor.lastrowid)
        return self.message(message_id)

    def message(self, message_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, session_id, role, content, status, created_at, updated_at "
                "FROM chat_messages WHERE id = ?",
                (int(message_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown chat message: {message_id}")
        return dict(row)

    def update_message(self, message_id: int, content: str, *, status: str) -> dict[str, Any]:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_messages SET content = ?, status = ?, updated_at = ? WHERE id = ?",
                (str(content), status, timestamp, int(message_id)),
            )
        return self.message(message_id)

    def messages(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id, role, content, status, created_at, updated_at "
                "FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def completed_history(self, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in self.messages(session_id)
            if item["status"] == "complete" and str(item["content"]).strip()
        ]

    def fail_streaming_messages(self, reason: str) -> list[dict[str, Any]]:
        """Close records abandoned by a broker/process restart."""
        timestamp = _now()
        marker = f"[LocalPilot answer interrupted: {str(reason).strip()}]"
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id FROM chat_messages WHERE status = 'streaming' ORDER BY id"
            ).fetchall()
            message_ids = [int(row["id"]) for row in rows]
            session_ids = {str(row["session_id"]) for row in rows}
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                connection.execute(
                    f"UPDATE chat_messages SET content = ?, status = 'error', updated_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (marker, timestamp, *message_ids),
                )
                connection.executemany(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                    ((timestamp, session_id) for session_id in session_ids),
                )
        return [self.message(message_id) for message_id in message_ids]

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        timestamp = _now()
        encoded = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_events(session_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, event_type, encoded, timestamp),
            )
            event_id = int(cursor.lastrowid)
        return {
            "id": event_id,
            "session_id": session_id,
            "type": event_type,
            "payload": json.loads(encoded),
            "created_at": timestamp,
        }

    def events_after(
        self,
        after_id: int,
        *,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, session_id, event_type, payload_json, created_at "
            "FROM chat_events WHERE id > ?"
        )
        params: list[Any] = [max(0, int(after_id))]
        if session_id:
            query += " AND (session_id IS NULL OR session_id = ?)"
            params.append(session_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "session_id": row["session_id"],
                "type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
