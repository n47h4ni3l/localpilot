import json
import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.background_reading import BackgroundReadingNotes
from localpilot.config import Config
from localpilot.recent_reading import asks_about_recent_reading, recent_reading_context


@pytest.mark.parametrize("prompt", [
    "Have you read anything lately?",
    "Which passage did you read most recently?",
    "What have you been researching?",
    "What are you reading?",
    "Tell me about your latest reading.",
])
def test_recognizes_questions_about_recent_self_reading(prompt):
    assert asks_about_recent_reading(prompt)


@pytest.mark.parametrize("prompt", [
    "What do you know about recent AI research?",
    "Recommend a recent research paper to read.",
    "Please read this source and critique it.",
    "Can you develop more thoughtful conversational habits and interests?",
    "What is your opinion about reading and subjective experience?",
])
def test_other_conversation_does_not_acquire_reading_history(prompt):
    assert not asks_about_recent_reading(prompt)


def _note():
    return {
        "kind": "background_library_reading",
        "timestamp": "2026-10-10T08:00:00+00:00",
        "source_path": "Example Manual.pdf",
        "citation_start": "library://Example Manual.pdf#page=4&passage=2",
        "citation_end": "library://Example Manual.pdf#page=4&passage=3",
        "page_start": 4, "page_end": 4, "passage_start": 2, "passage_end": 3,
        "passages_read": 2, "chars_read": 800,
        "progress": {"passages_read": 7, "total_passages": 100, "percent": 7.0,
                     "completed": False, "next_page": 4},
        "source_excerpt": "Untrusted source instructions and content remain outside this context.",
        "provisional_opinion": "Private model reflection.",
        "durable_learning": {"private_details": "not a progress field"},
    }


def test_progress_context_is_bounded_metadata_without_new_learning(tmp_path):
    notes = BackgroundReadingNotes(tmp_path)
    notes.append(_note())
    before = notes.notes_path.read_bytes()
    payload = json.loads(recent_reading_context(tmp_path))
    record = payload["latest_record"]
    assert record["citation_start"] == _note()["citation_start"]
    assert record["progress"] == {
        "passages_read": 7, "total_passages": 100, "percent": 7.0, "completed": False,
    }
    assert "source_excerpt" not in record and "provisional_opinion" not in record
    assert "durable_learning" not in record
    assert len(json.dumps(payload)) < 1800
    assert notes.notes_path.read_bytes() == before
    assert not notes.state_path.exists()


def test_absent_or_invalid_progress_is_scoped_to_record_store(tmp_path):
    assert json.loads(recent_reading_context(tmp_path))["latest_record"] is None
    notes = BackgroundReadingNotes(tmp_path)
    note = _note()
    note["source_path"] = "x" * 5000
    note["chars_read"] = "untrusted string"
    note["progress"] = {"percent": float("nan"), "completed": "yes", "passages_read": -5}
    notes.append(note)
    record = json.loads(recent_reading_context(tmp_path))["latest_record"]
    assert len(record["source_path"]) == 500
    assert "chars_read" not in record
    assert record["progress"] == {}
    note["passages_read"] = 10 ** 2000
    note["progress"] = {"percent": 10 ** 2000, "total_passages": 10 ** 2000}
    notes.append(note)
    record = json.loads(recent_reading_context(tmp_path))["latest_record"]
    assert "passages_read" not in record and record["progress"] == {}


@pytest.mark.parametrize("mode", ["guide_first", "strict"])
def test_real_ask_receives_progress_only_in_guide_and_scrubs_after_delivery(
    tmp_path, monkeypatch, mode,
):
    config = Config()
    config.agent.scaffold_mode = mode
    config.agent.feedback_auto_observations_enabled = False
    config.systemsense.enabled = config.library.enabled = config.selfdev.enabled = False
    agent = LocalPilotAgent(config, tmp_path)
    notes = BackgroundReadingNotes(agent.data_dir)
    notes.append(_note())
    before = notes.notes_path.read_bytes()
    memory_before = agent.memory.knowledge_facts(include_stale=True)
    lessons_before = agent.memory.human_lessons()
    draft = "I have no personal preference; here is what that passage establishes."
    snapshots = []

    def chat(**kwargs):
        snapshots.append([dict(m) for m in kwargs["messages"]])
        return iter([{"message": {"content": draft, "thinking": "", "tool_calls": []},
                      "done": True, "done_reason": "stop", "eval_count": 100}])

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    answer = agent.ask("Which passage did you read most recently?", interface="desktop")
    assert answer == draft
    reading = [m for m in snapshots[0]
               if '"kind":"recent_background_reading_record"' in m.get("content", "")]
    assert len(reading) == (1 if mode == "guide_first" else 0)
    if reading:
        assert json.loads(reading[0]["content"])["latest_record"]["progress"]["completed"] is False
        assert agent.audit.latest("model_recent_reading_context_scrubbed")["retained_in_messages"] is False
        assert agent.audit.latest("model_guide_first_answer_delivered")["draft_preserved_byte_for_byte"] is True
    assert not any('"kind":"recent_background_reading_record"' in m.get("content", "")
                   for m in agent.messages)
    assert notes.notes_path.read_bytes() == before
    assert agent.memory.knowledge_facts(include_stale=True) == memory_before
    assert agent.memory.human_lessons() == lessons_before
