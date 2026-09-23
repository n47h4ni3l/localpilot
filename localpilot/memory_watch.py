from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from localpilot.systemsense_watch import (
    SystemSenseWatchIntent,
    is_systemsense_process_investigation_request,
    is_systemsense_watch_report_request,
    parse_systemsense_watch_request,
)


@dataclass(frozen=True, slots=True)
class MemoryWatchIntent:
    expires_at: datetime
    label: str


def parse_memory_watch_request(
    prompt: str,
    *,
    now: datetime | None = None,
) -> MemoryWatchIntent | None:
    """Compatibility wrapper for the original RAM-watch entry point."""
    intent = parse_systemsense_watch_request(prompt, now=now)
    if intent is None or intent.profile != "memory":
        return None
    return MemoryWatchIntent(expires_at=intent.expires_at, label=intent.label)


def is_memory_watch_report_request(prompt: str) -> bool:
    return is_systemsense_watch_report_request(prompt) and any(
        term in str(prompt).casefold()
        for term in ("ram", "memory", "memory watch", "ram watch")
    )


def is_memory_watch_process_investigation_request(prompt: str) -> bool:
    return is_systemsense_process_investigation_request(prompt) and any(
        term in str(prompt).casefold()
        for term in ("ram", "memory", "memory watch", "ram watch")
    )


__all__ = [
    "MemoryWatchIntent",
    "SystemSenseWatchIntent",
    "parse_memory_watch_request",
    "is_memory_watch_report_request",
    "is_memory_watch_process_investigation_request",
]
