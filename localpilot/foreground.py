from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil


FOREGROUND_TURNS_FILENAME = "foreground-turns.json"


def write_foreground_turns(
    data_dir: str | Path,
    requests: Iterable[dict[str, Any]],
) -> bool:
    """Publish broker-owned active-turn metadata without message content."""
    destination = Path(data_dir).resolve() / FOREGROUND_TURNS_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    active = [
        {
            key: str(item.get(key) or "")
            for key in ("request_id", "session_id", "message_id")
        }
        for item in requests
    ]
    try:
        broker_started_at = psutil.Process(os.getpid()).create_time()
    except (psutil.Error, OSError):
        broker_started_at = None
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "broker_pid": os.getpid(),
        "broker_started_at": broker_started_at,
        "active_requests": active,
    }
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for attempt in range(5):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, destination)
            return True
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(0.02)
    return False


def active_foreground_turns(data_dir: str | Path) -> tuple[dict[str, str], ...]:
    """Return active turns only while the publishing broker process is still current."""
    path = Path(data_dir).resolve() / FOREGROUND_TURNS_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        broker_pid = int(payload.get("broker_pid") or 0)
        recorded_start = float(payload.get("broker_started_at") or 0.0)
        process = psutil.Process(broker_pid)
        if recorded_start and abs(process.create_time() - recorded_start) > 1.0:
            return ()
        requests = payload.get("active_requests")
    except (OSError, ValueError, TypeError, json.JSONDecodeError, psutil.Error):
        return ()
    if not isinstance(requests, list):
        return ()
    return tuple(
        {
            key: str(item.get(key) or "")
            for key in ("request_id", "session_id", "message_id")
        }
        for item in requests
        if isinstance(item, dict) and str(item.get("request_id") or "").strip()
    )
