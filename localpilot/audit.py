from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SENSITIVE = ("password", "passwd", "token", "secret", "api_key", "apikey", "credential")


def _scrub(value: Any, key: str = "") -> Any:
    if any(word in key.lower() for word in _SENSITIVE):
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

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_scrub(fields),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
