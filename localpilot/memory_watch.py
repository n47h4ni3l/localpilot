from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class MemoryWatchIntent:
    expires_at: datetime
    label: str


_START_VERBS = re.compile(
    r"\b(?:monitor|watch|track|keep an eye on|keep watching|observe|log|record)\b",
    re.IGNORECASE,
)
_MEMORY_TERMS = re.compile(
    r"\b(?:ram|memory|memory usage|memory use|memory pressure)\b",
    re.IGNORECASE,
)
_REPORT_TERMS = re.compile(
    r"\b(?:what did|what has|what was|what is|results?|report|findings?|find|found|"
    r"show me|status|happened|using|used|spikes?|spiking|culprit|consumer)\b",
    re.IGNORECASE,
)
_WATCH_TERMS = re.compile(
    r"\b(?:monitor|watch|tracking|tracked|memory watch|ram watch)\b",
    re.IGNORECASE,
)


def parse_memory_watch_request(
    prompt: str,
    *,
    now: datetime | None = None,
) -> MemoryWatchIntent | None:
    """Recognize explicit owner requests to start passive RAM monitoring.

    The parser is intentionally conservative: it only returns an intent when
    both an active monitoring verb and an explicit memory/RAM subject are
    present. Ordinary questions about current memory usage remain normal
    diagnostic requests.
    """
    text = " ".join(str(prompt or "").strip().split())
    if not text or not _START_VERBS.search(text) or not _MEMORY_TERMS.search(text):
        return None

    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()

    lowered = text.casefold()
    if re.search(r"\b(?:today|for the rest of today|until tonight|this afternoon|this evening)\b", lowered):
        end = current.replace(hour=23, minute=59, second=59, microsecond=0)
        if end <= current:
            end = current + timedelta(hours=1)
        return MemoryWatchIntent(expires_at=end, label="today")

    hours_match = re.search(
        r"\b(?:for|over|next)\s+(\d{1,2}(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
        lowered,
    )
    if hours_match:
        hours = max(0.25, min(float(hours_match.group(1)), 24.0))
        return MemoryWatchIntent(
            expires_at=current + timedelta(hours=hours),
            label=f"{hours:g} hours",
        )

    minutes_match = re.search(
        r"\b(?:for|over|next)\s+(\d{1,3})\s*(?:minutes?|mins?)\b",
        lowered,
    )
    if minutes_match:
        minutes = max(5, min(int(minutes_match.group(1)), 24 * 60))
        return MemoryWatchIntent(
            expires_at=current + timedelta(minutes=minutes),
            label=f"{minutes} minutes",
        )

    return MemoryWatchIntent(
        expires_at=current + timedelta(hours=8),
        label="8 hours",
    )


def is_memory_watch_report_request(prompt: str) -> bool:
    """Recognize follow-up questions that should be grounded in watch history."""
    text = " ".join(str(prompt or "").strip().split())
    if not text or not _MEMORY_TERMS.search(text):
        return False
    return bool(_WATCH_TERMS.search(text) and _REPORT_TERMS.search(text))
