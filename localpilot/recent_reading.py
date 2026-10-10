"""Bounded progress evidence for questions about the assistant's recent reading."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from localpilot.background_reading import BackgroundReadingNotes

_MAX_METADATA_COUNT = 1_000_000_000


def asks_about_recent_reading(prompt: str) -> bool:
    text = " ".join(str(prompt).lower().split())
    activity = r"(?:read|reading|researched|researching|studied|studying)"
    self_activity = re.search(
        rf"\byou(?:['’]ve)? (?:have |been |have been )?{activity}\b|"
        rf"\b(?:have|did|are|were) you(?: been)? {activity}\b|"
        r"\byour (?:recent |latest |last )?(?:reading|research|study)\b",
        text,
    )
    recent = re.search(r"\b(?:recent(?:ly)?|lately|latest|last|so far)\b", text)
    ongoing = re.search(
        r"\b(?:have you been|are you|you['’]ve been|you have been) "
        r"(?:reading|researching|studying)\b", text,
    )
    return bool(self_activity and (recent or ongoing))


def recent_reading_context(data_dir: str | Path) -> str:
    """Read existing notes without starting reading or persisting new learning.

    Topic-ranked facts establish source knowledge, not the order of reading.
    Only progress metadata enters this context; excerpts and model reflections
    remain in the existing private note store.
    """
    try:
        note = BackgroundReadingNotes(data_dir).latest()
    except (OSError, ValueError, TypeError):
        return ""
    record = None
    if note:
        record = {
            key: str(note[key])[:500]
            for key in ("timestamp", "source_path", "citation_start", "citation_end")
            if isinstance(note.get(key), str)
        }
        for key in ("page_start", "page_end", "passage_start", "passage_end",
                    "passages_read", "chars_read"):
            value = note.get(key)
            if (isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value <= _MAX_METADATA_COUNT):
                record[key] = value
        progress = note.get("progress")
        if isinstance(progress, dict):
            clean_progress = {}
            for key in ("passages_read", "total_passages"):
                value = progress.get(key)
                if (isinstance(value, int) and not isinstance(value, bool)
                        and 0 <= value <= _MAX_METADATA_COUNT):
                    clean_progress[key] = value
            percent = progress.get("percent")
            if (isinstance(percent, (int, float)) and not isinstance(percent, bool)
                    and 0 <= percent <= 100 and math.isfinite(percent)):
                clean_progress["percent"] = percent
            if isinstance(progress.get("completed"), bool):
                clean_progress["completed"] = progress["completed"]
            record["progress"] = clean_progress
    return json.dumps({
        "kind": "recent_background_reading_record",
        "scope": "latest recorded background reading activity, not topic-ranked knowledge",
        "latest_record": record,
    }, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
