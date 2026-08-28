from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from localpilot.background_reading import BackgroundLibraryReader, BackgroundReadingNotes
from localpilot.agent import LocalPilotAgent
from localpilot.config import Config
from localpilot.learning import LearningMemory


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
        self.digests = {path: f"digest-{path}" for path in self.sections}
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
                "source_digest": self.digests[path],
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
            "source_digest": self.digests[path],
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


def _empty_extractor(*_):
    return {"candidates": []}


def _learning_candidates(*_):
    return {
        "candidates": [
            {
                "candidate_id": "claim",
                "type": "source_claim",
                "subject": "Feedback loops",
                "summary": "Uncertainty can create expanding research loops.",
                "dedupe_key": "uncertainty expanding research loops",
            },
            {
                "candidate_id": "heuristic",
                "type": "heuristic",
                "subject": "Research loop diagnosis",
                "summary": "Stop all research as soon as context grows.",
                "dedupe_key": "diagnose reinforcing research loops",
            },
            {
                "candidate_id": "question",
                "type": "question",
                "subject": "Research efficiency",
                "summary": "Does context growth preserve answer accuracy?",
                "dedupe_key": "context growth answer accuracy question",
            },
            {
                "candidate_id": "hypothesis",
                "type": "selfdev_hypothesis",
                "subject": "Bounded research",
                "summary": "Detecting reinforcing loops may reduce tool count without lowering accuracy.",
                "dedupe_key": "detect loops reduce tools preserve accuracy",
            },
            {
                "candidate_id": "unsupported",
                "type": "source_claim",
                "subject": "Universal certainty",
                "summary": "All research loops always reduce intelligence.",
                "dedupe_key": "all loops reduce intelligence",
            },
        ]
    }


def _learning_verifications(_citation, _digest, passage, _candidates):
    evidence = " ".join(passage.split()[:3])
    return {
        "verifications": [
            {
                "candidate_id": "claim",
                "verdict": "supported",
                "reason": "The section describes the loop.",
                "confidence": 0.86,
                "evidence_quote": evidence,
            },
            {
                "candidate_id": "heuristic",
                "verdict": "corrected",
                "reason": "The absolute instruction overstates the passage.",
                "confidence": 0.78,
                "evidence_quote": evidence,
                "corrected_subject": "Research loop diagnosis",
                "corrected_summary": (
                    "When research keeps expanding, test whether uncertainty is reinforcing "
                    "context growth."
                ),
            },
            {
                "candidate_id": "question",
                "verdict": "supported",
                "reason": "This is a grounded follow-up question.",
                "confidence": 0.72,
                "evidence_quote": evidence,
            },
            {
                "candidate_id": "hypothesis",
                "verdict": "supported",
                "reason": "The passage motivates a measurable self-development test.",
                "confidence": 0.74,
                "evidence_quote": evidence,
            },
            {
                "candidate_id": "unsupported",
                "verdict": "rejected",
                "reason": "The passage does not support a universal claim.",
                "confidence": 0.1,
                "evidence_quote": evidence,
            },
        ]
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
    memory=None,
    extractor=_empty_extractor,
    verifier=None,
):
    notes = BackgroundReadingNotes(tmp_path / "data")
    audit = FakeAudit()
    instance = BackgroundLibraryReader(
        config or _config(),
        tmp_path,
        memory=memory or LearningMemory(tmp_path / "data" / "learning.sqlite3"),
        governor=governor or FakeGovernor(),
        library=library or FakeLibrary(),
        notes=notes,
        audit=audit,
        now=clock or Clock(),
        chooser=chooser,
        reflector=reflector,
        extractor=extractor,
        verifier=verifier,
    )
    return instance, notes, audit


def test_no_owner_permission_gate_and_learning_bridge_runs_by_default(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, library=library)

    assert reader.run_once()["status"] == "read"
    assert library.source_calls == 2


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
    assert latest["authority"] == "private_note_with_verified_typed_learning_bridge"
    assert latest["durable_learning"]["candidate_count"] == 0
    assert audit.rows[-1][0] == "background_library_read"


def test_empty_extraction_does_not_invent_authoritative_knowledge(tmp_path):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    reader, _, _ = _reader(tmp_path, memory=memory)

    assert reader.run_once()["status"] == "read"
    assert memory.knowledge_facts() == []
    assert memory.durable_learnings() == []


def test_cooldown_prevents_rapid_repeat_reads(tmp_path):
    library = FakeLibrary()
    reader, _, _ = _reader(tmp_path, library=library)

    assert reader.run_once()["status"] == "read"
    second = reader.run_once()
    assert second["status"] == "cooldown"
    assert second["seconds_remaining"] > 0
    assert len(library.read_calls) == 1


def test_verified_reading_creates_durable_typed_learning_with_audit_evidence(tmp_path):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    reader, notes, audit = _reader(
        tmp_path,
        memory=memory,
        extractor=_learning_candidates,
        verifier=_learning_verifications,
    )

    result = reader.run_once()
    facts = memory.knowledge_facts(stage="library")
    typed = memory.durable_learnings()

    assert result["status"] == "read"
    assert result["durable_learning"]["persisted_count"] == 4
    assert result["durable_learning"]["rejected_count"] == 1
    assert result["durable_learning"]["corrected_count"] == 1
    assert len(facts) == 1
    assert facts[0].fact_type == "library_claim"
    assert facts[0].summary.startswith("First Book.pdf argues:")
    assert facts[0].source_uri == (
        "library://First Book.pdf#page=1&passage=1&end_page=2&end_passage=3"
    )
    assert facts[0].source_digest == "digest-First Book.pdf"
    assert facts[0].source_kind == "verified_library_passage"
    assert facts[0].last_verified_at
    assert "provenance:autonomous_library_reading" in facts[0].relationships
    assert {item.learning_type for item in typed} == {
        "heuristic",
        "question",
        "selfdev_hypothesis",
    }
    assert all(item.source_digest == facts[0].source_digest for item in typed)
    heuristic = next(item for item in typed if item.learning_type == "heuristic")
    assert heuristic.summary.startswith("When research keeps expanding")
    assert "Stop all research" not in heuristic.summary
    assert not any(
        fact.subject == "Universal certainty"
        for fact in memory.knowledge_facts(include_stale=True)
    )
    learning_audit = next(
        row for row in audit.rows if row[0] == "background_library_learning"
    )
    assert learning_audit[1]["rejected"][0]["reason"].startswith(
        "The passage does not support"
    )
    assert learning_audit[1]["authority"] == (
        "learning_only_no_action_or_code_authority"
    )
    assert notes.latest()["durable_learning"]["persisted_count"] == 4


def test_repeated_reading_updates_stable_learning_keys_without_multiplication(tmp_path):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    clock = Clock()
    reader, _, _ = _reader(
        tmp_path,
        memory=memory,
        clock=clock,
        extractor=_learning_candidates,
        verifier=_learning_verifications,
    )

    first = reader.run_once()
    first_keys = {
        item.fact_key for item in memory.knowledge_facts(stage="library")
    } | {item.learning_key for item in memory.durable_learnings()}
    clock.advance()
    second = reader.run_once()
    second_keys = {
        item.fact_key for item in memory.knowledge_facts(stage="library")
    } | {item.learning_key for item in memory.durable_learnings()}

    assert first["durable_learning"]["persisted_count"] == 4
    assert second["durable_learning"]["persisted_count"] == 4
    assert {
        item["outcome"] for item in second["durable_learning"]["persisted"]
    } == {"updated"}
    assert first_keys == second_keys
    assert len(second_keys) == 4


def test_revised_source_bytes_stale_unreverified_prior_learning(tmp_path):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    clock = Clock()
    library = FakeLibrary()
    calls = 0

    def candidates_once(*_):
        nonlocal calls
        calls += 1
        return _learning_candidates() if calls == 1 else {"candidates": []}

    reader, _, _ = _reader(
        tmp_path,
        memory=memory,
        library=library,
        clock=clock,
        extractor=candidates_once,
        verifier=_learning_verifications,
    )
    assert reader.run_once()["durable_learning"]["persisted_count"] == 4

    library.digests["First Book.pdf"] = "revised-source-digest"
    library.sections["First Book.pdf"][0] = (1, 1, "revised passage bytes")
    clock.advance()
    second = reader.run_once()

    assert second["durable_learning"]["stale_invalidated"] == 4
    assert memory.knowledge_facts(stage="library") == []
    assert memory.durable_learnings() == []
    assert all(
        item.stale
        for item in memory.knowledge_facts(stage="library", include_stale=True)
    )
    assert all(item.stale for item in memory.durable_learnings(include_stale=True))


def test_operator_and_selfdev_retrieve_learning_without_flattening_epistemic_types(
    tmp_path,
):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    reader, _, _ = _reader(
        tmp_path,
        memory=memory,
        extractor=_learning_candidates,
        verifier=_learning_verifications,
    )
    reader.run_once()

    config = _config()
    library_root = tmp_path / "operator-library"
    library_root.mkdir()
    config.library.root = str(library_root)
    agent = LocalPilotAgent(config, tmp_path)
    agent.memory = memory
    context, retrieved = agent._learning_context(
        "How should uncertainty and reinforcing context growth affect research loops?"
    )
    payload = __import__("json").loads(context.split("\n", 1)[1])
    heuristic = next(
        item for item in payload["facts"] if item.get("epistemic_type") == "heuristic"
    )

    assert retrieved
    assert heuristic["objective_fact"] is False
    assert heuristic["fact_type"] == "library_heuristic"
    assert "Never state an item marked objective_fact=false as fact" in context
    assert any(
        "Library heuristic" in item
        for item in memory.reusable_lessons(
            limit=8,
            query="uncertainty reinforcing context growth research loops",
        )
    )
    discovery = memory.discovery_context()
    hypothesis = next(
        item
        for item in discovery["library_learnings"]
        if item["learning_type"] == "selfdev_hypothesis"
    )
    assert hypothesis["objective_fact"] is False


def test_education_bridge_stores_no_raw_dump_and_grants_no_extra_authority(tmp_path):
    memory = LearningMemory(tmp_path / "data" / "learning.sqlite3")
    library = FakeLibrary()
    reader, _, _ = _reader(
        tmp_path,
        memory=memory,
        library=library,
        extractor=_learning_candidates,
        verifier=_learning_verifications,
    )
    source_text = "\n\n".join(row[2] for row in library.sections["First Book.pdf"][:6])
    result = reader.run_once()
    stored = [item.summary for item in memory.knowledge_facts()] + [
        item.summary for item in memory.durable_learnings()
    ]

    assert result["status"] == "read"
    assert all(source_text not in summary for summary in stored)
    assert not memory.schema_columns() & {
        "prompt",
        "transcript",
        "messages",
        "reasoning",
        "raw_source",
        "source_text",
        "weights",
    }
    assert not hasattr(reader, "merge")
    assert not hasattr(reader, "train")
    assert all(
        item["objective_fact"] is False
        for item in result["durable_learning"]["persisted"]
        if item["learning_type"] not in {"source_claim", "source_concept"}
    )
