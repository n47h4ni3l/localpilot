from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class SystemSenseWatchIntent:
    profile: str
    expires_at: datetime
    label: str


_START_VERBS = re.compile(
    r"\b(?:monitor|watch|track|keep an eye on|keep watching|observe|log|record)\b",
    re.IGNORECASE,
)

_PROFILE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("memory", re.compile(r"\b(?:ram|memory|memory usage|memory use|memory pressure)\b", re.I)),
    ("gpu", re.compile(r"\b(?:gpu|vram|graphics(?: card)?|video memory)\b", re.I)),
    ("network", re.compile(r"\b(?:network|internet|upload|download|bandwidth|connection traffic)\b", re.I)),
    ("storage", re.compile(r"\b(?:disk|storage|ssd|drive|read(?:ing)?|writ(?:e|ing)|i/o|io)\b", re.I)),
    ("cpu", re.compile(r"\b(?:cpu|processor|core usage|cpu usage)\b", re.I)),
)

_SYSTEM_TERMS = re.compile(
    r"\b(?:system|computer|pc|machine|freeze|freezing|hang|hanging|slowdown|"
    r"slow down|stutter|stuttering|performance spike|resource spike)\b",
    re.IGNORECASE,
)

_REPORT_TERMS = re.compile(
    r"\b(?:what did|what has|what was|what is|results?|report|findings?|find|found|"
    r"show me|status|happened|using|used|spikes?|spiking|culprit|consumer)\b",
    re.IGNORECASE,
)

_WATCH_TERMS = re.compile(
    r"\b(?:monitor|watch|tracking|tracked|systemsense watch|resource watch|"
    r"memory watch|ram watch|gpu watch|network watch|disk watch|cpu watch)\b",
    re.IGNORECASE,
)


def _duration(text: str, *, now: datetime) -> tuple[datetime, str]:
    lowered = text.casefold()
    if re.search(
        r"\b(?:today|for the rest of today|until tonight|this afternoon|this evening)\b",
        lowered,
    ):
        end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        if end <= now:
            end = now + timedelta(hours=1)
        return end, "today"

    hours_match = re.search(
        r"\b(?:for|over|next)\s+(\d{1,2}(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
        lowered,
    )
    if hours_match:
        hours = max(0.25, min(float(hours_match.group(1)), 24.0))
        return now + timedelta(hours=hours), f"{hours:g} hours"

    minutes_match = re.search(
        r"\b(?:for|over|next)\s+(\d{1,3})\s*(?:minutes?|mins?)\b",
        lowered,
    )
    if minutes_match:
        minutes = max(5, min(int(minutes_match.group(1)), 24 * 60))
        return now + timedelta(minutes=minutes), f"{minutes} minutes"

    return now + timedelta(hours=8), "8 hours"


def parse_systemsense_watch_request(
    prompt: str,
    *,
    now: datetime | None = None,
) -> SystemSenseWatchIntent | None:
    """Recognize explicit owner requests for a bounded SystemSense resource watch."""
    text = " ".join(str(prompt or "").strip().split())
    if not text or not _START_VERBS.search(text):
        return None

    profile = None
    for name, pattern in _PROFILE_PATTERNS:
        if pattern.search(text):
            profile = name
            break
    if profile is None and _SYSTEM_TERMS.search(text):
        profile = "system"
    if profile is None:
        return None

    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    expires_at, label = _duration(text, now=current)
    return SystemSenseWatchIntent(profile=profile, expires_at=expires_at, label=label)


def is_systemsense_watch_report_request(prompt: str) -> bool:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return False
    if _WATCH_TERMS.search(text) and _REPORT_TERMS.search(text):
        return True
    # Follow-ups can omit the word "watch", but require clearly retrospective
    # wording so a normal "what is my RAM usage?" current-state question is not
    # accidentally forced onto historical watch evidence.
    historical = re.search(
        r"\b(?:what did|what has|results?|findings?|what happened|found|"
        r"spikes?|spiked|culprit|consumer|used|during the watch)\b",
        text,
        re.IGNORECASE,
    )
    return bool(
        historical
        and any(pattern.search(text) for _, pattern in _PROFILE_PATTERNS)
    )


def is_systemsense_process_investigation_request(prompt: str) -> bool:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return False
    investigation = re.search(
        r"\b(?:investigate|research|identify|determine|figure out|look into|"
        r"what is|what's|what does|why is|why does|where did|what started)\b",
        text,
        re.IGNORECASE,
    )
    process_subject = re.search(
        r"(?:\bprocess(?:es)?\b|\bpid\b|\bconsumer(?:s)?\b|\bculprit(?:s)?\b|"
        r"\b[a-z0-9_.-]+\.exe\b)",
        text,
        re.IGNORECASE,
    )
    resource_context = any(pattern.search(text) for _, pattern in _PROFILE_PATTERNS)
    historical_context = re.search(
        r"\b(?:that|those|used|spiked|during|watch|monitor|culprit|consumer)\b",
        text,
        re.IGNORECASE,
    )
    watch_context = _WATCH_TERMS.search(text) or (
        resource_context and historical_context
    )
    return bool(investigation and process_subject and watch_context)
