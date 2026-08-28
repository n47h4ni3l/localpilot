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
            "Recent autonomous library reading sessions (bounded sections only; provisional "
            "reflections; not durable knowledge facts):"
        ]
        for row in reversed(rows):
            timestamp = str(row.get("timestamp") or "unknown time")[:80]
            source_path = str(row.get("source_path") or "")[:600]
            citation_start = str(
                row.get("citation_start") or row.get("citation") or "uncited"
            )[:600]
            citation_end = str(row.get("citation_end") or citation_start)[:600]
            progress = row.get("progress") if isinstance(row.get("progress"), dict) else {}
            opinion = str(
                row.get("provisional_opinion") or row.get("reflection") or ""
            ).strip()[:_MAX_REFLECTION_CHARS]
            questions = row.get("questions_raised")
            if not isinstance(questions, list):
                questions = []
            learning = (
                row.get("durable_learning")
                if isinstance(row.get("durable_learning"), dict)
                else {}
            )
            persisted = (
                learning.get("persisted")
                if isinstance(learning.get("persisted"), list)
                else []
            )
            learned_types = sorted(
                {
                    str(item.get("learning_type") or "")
                    for item in persisted
                    if isinstance(item, dict) and item.get("learning_type")
                }
            )
            learning_text = (
                f"{learning.get('persisted_count', 0)} persisted"
                f" ({', '.join(learned_types) or 'none'}); "
                f"{learning.get('corrected_count', 0)} corrected; "
                f"{learning.get('rejected_count', 0)} rejected"
            )
            if source_path and row.get("citation_start"):
                completed = bool(progress.get("completed"))
                progress_text = (
                    f"{progress.get('passages_read', '?')}/{progress.get('total_passages', '?')} "
                    f"indexed passages ({progress.get('percent', '?')}%); "
                    + ("source complete" if completed else "source not complete")
                )
                intent = (
                    "follow a related source"
                    if row.get("follow_related_source")
                    else "continue this source"
                    if row.get("wants_to_continue")
                    else "choose afresh next session"
                )
                output.append(
                    f"- {timestamp}\n"
                    f"  Source: {source_path}\n"
                    f"  Section actually read: {citation_start} through {citation_end}\n"
                    f"  Progress: {progress_text}\n"
                    f"  Provisional opinion: {opinion or '(none recorded)'}\n"
                    f"  Questions: {'; '.join(str(item)[:400] for item in questions[:3]) or '(none recorded)'}\n"
                    f"  Durable learning: {learning_text}\n"
                    f"  Next preference: {intent}"
                )
            else:
                # Preserve bounded visibility for notes written by the earlier
                # passage-retrieval implementation without overstating them.
                query = str(row.get("query") or "")[:300]
                output.append(
                    f"- {timestamp}\n"
                    f"  Citation: {citation_start} (legacy bounded passage)\n"
                    f"  Legacy selection theme: {query or '(not recorded)'}\n"
                    f"  Provisional note: {opinion or '(none recorded)'}"
                )
        return "\n".join(output)[:_MAX_OUTPUT_CHARS]
