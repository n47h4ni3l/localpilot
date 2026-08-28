from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from localpilot.background_reading import BackgroundLibraryReader, BackgroundReadingNotes
from localpilot.config import Config


class ForbiddenMemory:
    def __getattr__(self, name):
        raise AssertionError(f"Background reading must not access LearningMemory.{name}")


class FakeGovernor:
    def __init__(self, *allowed: bool, reason="resource gate closed"):
        self.allowed = list(allowed or (True,))
        self.reason = reason
        self.calls = 0

    def sample(self):
        value = self.allowed[min(self.calls, len(self.allowed) - 1)]
        self.calls += 1
        return SimpleNamespace(background_allowed=value, reason=self.reason)


class Clock:
    def __init__(self):
        self.value = datetime(2026, 8, 29, 0, 0, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self):
        self.value += timedelta(hours=2)


class FakeLibrary:
    def __init__(self):
        first = [
            (1 + (index // 3), 1 + (index % 3), (f"first-{index} " * 80).strip())
            for index in range(8)
        ]
        second = [
            (1, index + 1, (f"second-{index} " * 60).strip()) for index in range(3)
        ]
        self.sections = {"First Book.pdf": first, "Second Book.pdf": second}
        self.source_calls = 0
        self.read_calls: list[tuple[str, int, int, int, int]] = []

    def list_indexed_sources(self, max_sources=200):
        self.source_calls += 1
        return [
            {
                "path": path,
                "kind": "pdf",
                "page_count": max(page for page, _, _ in rows),
                "passage_count": len(rows),
                "source_digest": f"digest-{path}",
                "opening_excerpt": rows[0][2][:200],
            }
            for path, rows in self.sections.items()
        ][:max_sources]

    def read_progressive_section(
        self, path, *, page=1, passage=1, max_passages=6, max_chars=9000
    ):
        self.read_calls.append((path, page, passage, max_passages, max_chars))
        rows = self.sections[path]
        eligible = [row for row in rows if (row[0], row[1]) >= (page, passage)]
        selected = []
        chars = 0
        for row in eligible[:max_passages]:
            if selected and chars + len(row[2]) > max_chars:
                break
            selected.append(row)
            chars += len(row[2])
        next_row = eligible[len(selected)] if len(selected) < len(eligible) else None
        start = selected[0]
        end = selected[-1]
        return {
            "available": True,
            "source_path": path,
            "source_digest": f"digest-{path}",
            "page_count": max(row[0] for row in rows),
            "total_passages": len(rows),
            "start_page": start[0],
            "start_passage": start[1],
            "end_page": end[0],
            "end_passage": end[1],
            "passages_read": len(selected),
            "chars_read": chars,
            "text": "\n\n".join(row[2] for row in selected),
            "completed": next_row is None,
            "next_page": next_row[0] if next_row else None,
            "next_passage": next_row[1] if next_row else None,
        }


class FakeAudit:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


def _config() -> Config:
    config = Config()
    config.library.enabled = True
    return config


def _chooser(sources, state, recent_notes):
    current = state.get("current_source")
    if current:
        return {
            "action": "continue",
            "source_path": current,
            "reason": "Continue from the saved progressive cursor.",
            "interest": "Continue the current question.",
        }
    progress = state.get("source_progress", {})
    source = next(item for item in sources if not progress.get(item["path"], {}).get("completed"))
    return {
        "action": "switch",
        "source_path": source["path"],
        "reason": "Choose a novel indexed source.",
        "interest": "Explore this source.",
    }


def _reflection(*_):
    return {
        "provisional_opinion": "The section suggests a useful but unverified design idea.",
        "questions_raised": ["Would the idea survive a focused evaluation?"],
        "wants_to_continue": True,
        "follow_related_source": False,
        "related_interest": "evaluation and reliability",
    }


def _reader(
    tmp_path: Path,
    *,
    config=None,
    governor=None,
    library=None,
    clock=None,
    chooser=_chooser,
    reflector=_reflection,
):
    notes = BackgroundReadingNotes(tmp_path / "data")
    audit = FakeAudit()
    instance = BackgroundLibraryReader(
        config or _config(),
        tmp_path,
        memory=ForbiddenMemory(),
        governor=governor or FakeGovernor(),
        library=library or FakeLibrary(),
        notes=notes,
        audit=audit,
        now=clock or Clock(),
        chooser=chooser,
        reflector=reflector,
    )
    return instance, notes, audit


def test_no_owner_permission_gate_or_learning_memory_access(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, library=library)

    assert reader.run_once()["status"] == "read"
    assert library.source_calls == 1


def test_disabled_library_never_reads(tmp_path):
    config = _config()
    config.library.enabled = False
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, config=config, library=library)

    assert reader.run_once() == {"status": "disabled"}
    assert library.source_calls == 0


def test_resource_gate_defers_before_library_work(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(
        tmp_path,
        governor=FakeGovernor(False, reason="user is active"),
        library=library,
    )

    assert reader.run_once() == {"status": "deferred", "reason": "user is active"}
    assert library.source_calls == 0


def test_resources_are_rechecked_before_selection_and_reflection(tmp_path):
    selection_reader, _, _ = _reader(
        tmp_path / "selection", governor=FakeGovernor(True, False, reason="capacity changed")
    )
    assert selection_reader.run_once()["status"] == "deferred_before_selection"

    reflection_library = FakeLibrary()
    reflection_reader, notes, _ = _reader(
        tmp_path / "reflection",
        governor=FakeGovernor(True, True, False, reason="capacity changed"),
        library=reflection_library,
    )
    assert reflection_reader.run_once()["status"] == "deferred_before_reflection"
    assert notes.latest() is None
    assert notes.state()["source_progress"] == {}


def test_progressive_cursor_advances_across_idle_cycles_without_page_four_repeat(tmp_path):
    clock = Clock()
    library = FakeLibrary()
    reader, notes, _ = _reader(tmp_path, library=library, clock=clock)

    first = reader.run_once()
    clock.advance()
    second = reader.run_once()

    assert first["citation_start"] == "library://First Book.pdf#page=1&passage=1"
    assert second["citation_start"] == "library://First Book.pdf#page=3&passage=1"
    assert library.read_calls[:2] == [
        ("First Book.pdf", 1, 1, 6, 9000),
        ("First Book.pdf", 3, 1, 6, 9000),
    ]
    assert all(call[1] != 4 for call in library.read_calls)
    assert notes.state()["source_progress"]["First Book.pdf"]["completed"] is True


def test_completed_source_switches_to_a_novel_source_and_is_not_reread(tmp_path):
    clock = Clock()
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, library=library, clock=clock)

    assert reader.run_once()["source_path"] == "First Book.pdf"
    clock.advance()
    assert reader.run_once()["source_path"] == "First Book.pdf"
    clock.advance()
    third = reader.run_once()

    assert third["source_path"] == "Second Book.pdf"
    assert [call[0] for call in library.read_calls] == [
        "First Book.pdf",
        "First Book.pdf",
        "Second Book.pdf",
    ]


def test_section_budget_and_structured_note_persistence_are_honest(tmp_path):
    library = FakeLibrary()
    reader, notes, audit = _reader(tmp_path, library=library)
    result = reader.run_once()
    latest = notes.latest()

    assert result["status"] == "read"
    assert result["progress"] == {
        "passages_read": 6,
        "total_passages": 8,
        "percent": 75.0,
        "completed": False,
        "next_page": 3,
        "next_passage": 1,
    }
    assert "Read one bounded section" in result["activity_summary"]
    assert "6/8 indexed passages" in result["activity_summary"]
    assert "read the book" not in result["activity_summary"].casefold()
    assert "afternoon" not in result["activity_summary"].casefold()
    assert latest is not None
    assert latest["citation_start"].endswith("page=1&passage=1")
    assert latest["citation_end"].endswith("page=2&passage=3")
    assert latest["passages_read"] <= 6
    assert latest["chars_read"] <= 9000
    assert latest["provisional_opinion"]
    assert latest["questions_raised"]
    assert latest["wants_to_continue"] is True
    assert latest["authority"] == "reading_note_only_not_durable_knowledge"
    assert audit.rows[-1][0] == "background_library_read"


def test_background_reader_has_no_knowledge_fact_promotion_path(tmp_path):
    # ForbiddenMemory raises on every attribute access, including any attempted
    # knowledge-fact write. A complete reading cycle proves the path stays apart.
    reader, _, _ = _reader(tmp_path)
    assert reader.run_once()["status"] == "read"


def test_cooldown_prevents_rapid_repeat_reads(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, library=library)

    assert reader.run_once()["status"] == "read"
    second = reader.run_once()
    assert second["status"] == "cooldown"
    assert second["seconds_remaining"] > 0
    assert len(library.read_calls) == 1
