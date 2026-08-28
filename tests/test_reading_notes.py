from __future__ import annotations

import json
from pathlib import Path

from localpilot.config import Config
from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.reading_notes import LibraryReadingNotesReader


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_empty_reading_notes_are_clear(tmp_path):
    reader = LibraryReadingNotesReader(tmp_path, "data")
    assert reader.get_recent_library_reading_notes() == (
        "No autonomous local-library reading notes have been recorded yet."
    )


def test_reader_returns_newest_notes_with_provisional_authority_label(tmp_path):
    reader = LibraryReadingNotesReader(tmp_path, "data")
    _write(
        reader.path,
        [
            {
                "kind": "background_library_reading",
                "timestamp": "2026-08-27T10:00:00+00:00",
                "citation": "library://Old.pdf#page=1&passage=1",
                "query": "old theme",
                "reflection": "old note",
            },
            {
                "kind": "background_library_reading",
                "timestamp": "2026-08-27T11:00:00+00:00",
                "citation": "library://New.pdf#page=2&passage=3",
                "query": "new theme",
                "reflection": "new note",
            },
        ],
    )
    text = reader.get_recent_library_reading_notes(limit=1)
    assert "provisional reflections; not durable knowledge facts" in text
    assert "library://New.pdf#page=2&passage=3" in text
    assert "new note" in text
    assert "Old.pdf" not in text


def test_reader_caps_requested_note_count(tmp_path):
    reader = LibraryReadingNotesReader(tmp_path, "data")
    _write(
        reader.path,
        [
            {
                "kind": "background_library_reading",
                "timestamp": str(index),
                "citation": f"library://Book.pdf#page={index + 1}&passage=1",
                "query": "theme",
                "reflection": "note",
            }
            for index in range(30)
        ],
    )
    text = reader.get_recent_library_reading_notes(limit=999)
    assert text.count("Citation:") == 12


def test_reader_reports_exact_section_and_progress_without_inflation(tmp_path):
    reader = LibraryReadingNotesReader(tmp_path, "data")
    _write(
        reader.path,
        [
            {
                "kind": "background_library_reading",
                "timestamp": "2026-08-29T03:00:00+00:00",
                "source_path": "Systems Book.pdf",
                "citation_start": "library://Systems Book.pdf#page=10&passage=1",
                "citation_end": "library://Systems Book.pdf#page=12&passage=2",
                "progress": {
                    "passages_read": 18,
                    "total_passages": 90,
                    "percent": 20.0,
                    "completed": False,
                },
                "provisional_opinion": "The section offers a useful feedback-loop framing.",
                "questions_raised": ["How could the framing be evaluated?"],
                "durable_learning": {
                    "persisted_count": 2,
                    "corrected_count": 1,
                    "rejected_count": 1,
                    "persisted": [
                        {"learning_type": "source_concept"},
                        {"learning_type": "heuristic"},
                    ],
                },
                "wants_to_continue": True,
                "follow_related_source": False,
            }
        ],
    )

    text = reader.get_recent_library_reading_notes()

    assert "bounded sections only" in text
    assert "Section actually read: library://Systems Book.pdf#page=10&passage=1 through" in text
    assert "18/90 indexed passages (20.0%); source not complete" in text
    assert "Durable learning: 2 persisted (heuristic, source_concept); 1 corrected; 1 rejected" in text
    assert "Next preference: continue this source" in text
    assert "read the book" not in text.casefold()
    assert "afternoon" not in text.casefold()


def test_library_enabled_registry_exposes_reading_notes_as_read_only(tmp_path):
    config = Config()
    config.library.enabled = True
    config.library.root = str(tmp_path / "library")
    (tmp_path / "library").mkdir()
    tools = registry(tmp_path, config=config)
    assert "get_recent_library_reading_notes" in tools
    assert tools["get_recent_library_reading_notes"].risk == RiskLevel.READ_ONLY


def test_library_disabled_registry_does_not_expose_reading_notes(tmp_path):
    config = Config()
    config.library.enabled = False
    tools = registry(tmp_path, config=config)
    assert "get_recent_library_reading_notes" not in tools
