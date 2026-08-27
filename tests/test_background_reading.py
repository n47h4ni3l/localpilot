from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from localpilot.background_reading import (
    BackgroundLibraryReader,
    BackgroundReadingNotes,
    _owner_allows_background_reading,
)
from localpilot.config import Config


class FakeMemory:
    def __init__(self, lessons):
        self.lessons = lessons

    def human_lessons(self, limit=50):
        return self.lessons[:limit]


class FakeGovernor:
    def __init__(self, allowed=True, reason="idle capacity available"):
        self.allowed = allowed
        self.reason = reason
        self.calls = 0

    def sample(self):
        self.calls += 1
        return SimpleNamespace(background_allowed=self.allowed, reason=self.reason)


class FakeLibrary:
    def __init__(self):
        self.search_calls = 0
        self.read_calls = []

    def search_library(self, query, max_results=6):
        self.search_calls += 1
        return (
            f"Local library search: {query!r}\n\n"
            "[1] library://First Book.pdf#page=4&passage=2\nagent planning excerpt\n\n"
            "[2] library://Second Book.pdf#page=8&passage=1\nreflection excerpt"
        )

    def read_library_passage(self, path, page=1, start_passage=1, max_passages=3):
        self.read_calls.append((path, page, start_passage, max_passages))
        return (
            f"Library source: library://{path}#page={page}\n\n"
            f"Passage {start_passage}:\nUseful bounded source text."
        )


class FakeAudit:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


def _permission_lesson():
    return SimpleNamespace(
        active=True,
        topic="library",
        lesson=(
            "You have standing permission to choose and read material from your owner-managed "
            "local library whenever doing so may help. When autonomous background reading "
            "capability is available, you may use it while I am away."
        ),
    )


def _config() -> Config:
    config = Config()
    config.library.enabled = True
    config.library.max_search_results = 8
    return config


def _reader(tmp_path: Path, *, memory=None, governor=None, library=None, now=None):
    config = _config()
    notes = BackgroundReadingNotes(tmp_path / "data")
    audit = FakeAudit()
    instance = BackgroundLibraryReader(
        config,
        tmp_path,
        memory=memory or FakeMemory([_permission_lesson()]),
        governor=governor or FakeGovernor(),
        library=library or FakeLibrary(),
        notes=notes,
        audit=audit,
        now=now or (lambda: datetime(2026, 8, 27, 12, 0, tzinfo=UTC)),
        reflector=lambda query, citation, passage: (
            "Interesting: useful agent idea.\nQuestion: how does it transfer?\n"
            "Relevance: candidate context only."
        ),
    )
    return instance, notes, audit


def test_background_reading_requires_explicit_standing_permission(tmp_path):
    memory = FakeMemory(
        [SimpleNamespace(active=True, topic="library", lesson="I have some books in the library.")]
    )
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, memory=memory, library=library)
    assert reader.run_once()["status"] == "permission_missing"
    assert library.search_calls == 0


def test_permission_recognizes_owner_background_library_lesson():
    assert _owner_allows_background_reading(FakeMemory([_permission_lesson()])) is True


def test_disabled_library_never_reads(tmp_path):
    config = _config()
    config.library.enabled = False
    library = FakeLibrary()
    reader = BackgroundLibraryReader(
        config,
        tmp_path,
        memory=FakeMemory([_permission_lesson()]),
        governor=FakeGovernor(),
        library=library,
        notes=BackgroundReadingNotes(tmp_path / "data"),
        audit=FakeAudit(),
        reflector=lambda *_: "unused",
    )
    assert reader.run_once()["status"] == "disabled"
    assert library.search_calls == 0


def test_resource_gate_blocks_background_reading(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(
        tmp_path,
        governor=FakeGovernor(False, "user idle 20s < 600s"),
        library=library,
    )
    result = reader.run_once()
    assert result == {"status": "deferred", "reason": "user idle 20s < 600s"}
    assert library.search_calls == 0


def test_successful_read_is_bounded_and_saved_as_note_not_knowledge(tmp_path):
    # FakeMemory deliberately exposes only human_lessons. If the reader tried to
    # promote a knowledge_fact this test would fail with AttributeError.
    library = FakeLibrary()
    reader, notes, audit = _reader(tmp_path, library=library)
    result = reader.run_once()

    assert result["status"] == "read"
    assert result["citation"] == "library://First Book.pdf#page=4&passage=2"
    assert library.read_calls == [("First Book.pdf", 4, 2, 3)]
    latest = notes.latest()
    assert latest is not None
    assert latest["authority"] == "reading_note_only_not_durable_knowledge"
    assert latest["citation"] == result["citation"]
    assert "Interesting:" in latest["reflection"]
    assert audit.rows[-1][0] == "background_library_read"


def test_cooldown_prevents_rapid_repeat_reads(tmp_path):
    library = FakeLibrary()
    now = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    reader, _, _ = _reader(tmp_path, library=library, now=lambda: now)
    assert reader.run_once()["status"] == "read"
    second = reader.run_once()
    assert second["status"] == "cooldown"
    assert second["seconds_remaining"] > 0
    assert library.search_calls == 1


def test_seen_passage_is_skipped_when_a_later_candidate_is_novel(tmp_path):
    reader, notes, _ = _reader(tmp_path)
    notes.save_state(
        {
            "last_read_at": "",
            "query_cursor": 0,
            "seen_citations": ["library://First Book.pdf#page=4&passage=2"],
        }
    )
    result = reader.run_once()
    assert result["citation"] == "library://Second Book.pdf#page=8&passage=1"
