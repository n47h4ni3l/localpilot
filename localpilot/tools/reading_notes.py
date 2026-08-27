from __future__ import annotations

import json
from collections import deque
from pathlib import Path

_MAX_NOTES = 12
_MAX_OUTPUT_CHARS = 12_000
_MAX_REFLECTION_CHARS = 1800


class LibraryReadingNotesReader:
    """Bounded read-only access to LocalPilot's private autonomous reading notes.

    These notes are provisional reflections, not knowledge facts. The tool
    exists so future conversations can recall what LocalPilot actually read
    while the owner was away without silently promoting those reflections to
    factual authority.
    """

    def __init__(self, project_root: str | Path, data_dir: str) -> None:
        self.path = Path(project_root).resolve() / data_dir / "library-reading-notes.jsonl"

    def get_recent_library_reading_notes(self, limit: int = 5) -> str:
        limit = max(1, min(int(limit), _MAX_NOTES))
        if not self.path.exists():
            return "No autonomous local-library reading notes have been recorded yet."

        rows: deque[dict] = deque(maxlen=limit)
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    if row.get("kind") != "background_library_reading":
                        continue
                    rows.append(row)
        except OSError as exc:
            return f"Autonomous library reading notes are unavailable: {type(exc).__name__}."

        if not rows:
            return "No autonomous local-library reading notes have been recorded yet."

        output = [
            "Recent autonomous library reading notes (provisional reflections; not durable knowledge facts):"
        ]
        for row in reversed(rows):
            timestamp = str(row.get("timestamp") or "unknown time")[:80]
            citation = str(row.get("citation") or "uncited")[:600]
            query = str(row.get("query") or "")[:300]
            reflection = str(row.get("reflection") or "").strip()[:_MAX_REFLECTION_CHARS]
            output.append(
                f"- {timestamp}\n"
                f"  Citation: {citation}\n"
                f"  Selection theme: {query or '(not recorded)'}\n"
                f"  Note: {reflection or '(no reflection recorded)'}"
            )
        return "\n".join(output)[:_MAX_OUTPUT_CHARS]
