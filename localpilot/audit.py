from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SENSITIVE_FRAGMENTS = ("password", "passwd", "secret", "api_key", "apikey", "credential")


def _sensitive_key(key: str) -> bool:
    """Redact credential-bearing fields without hiding harmless token metrics."""
    normalized = str(key).strip().lower()
    if any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS):
        return True
    # Credential names commonly end in singular "token". Metrics such as
    # context_tokens, prompt_token_count, or generated_tokens remain visible.
    return normalized == "token" or normalized.endswith("_token") or normalized.endswith("token")


def _scrub(value: Any, key: str = "") -> Any:
    if _sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {k: _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value[:50]]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…"
    return value


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_scrub(fields),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def latest(self, event: str | None = None) -> dict[str, Any] | None:
        """Return the newest valid audit row, optionally filtered by event."""
        if not self.path.exists():
            return None
        latest_row: dict[str, Any] | None = None
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(row, dict):
                    continue
                if event is None or row.get("event") == event:
                    latest_row = row
        return latest_row

    def recent(self, event: str | None = None, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return a bounded newest-first view of valid audit rows."""
        if not self.path.exists():
            return []
        bounded_limit = max(1, min(int(limit), 100))
        rows: list[dict[str, Any]] = []
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(row, dict):
                    continue
                if event is None or row.get("event") == event:
                    rows.append(row)
                    if len(rows) > bounded_limit:
                        rows.pop(0)
        rows.reverse()
        return rows
