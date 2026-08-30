from ollama import Client

from localpilot.learning import LearningMemory
from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.learning_readonly import LearningMemoryReader


def _memory(root):
    return LearningMemory(root / "localpilot-data" / "learning.sqlite3")


def _fact(memory, *, key, fact_type, source, digest, stage="python"):
    memory.upsert_knowledge_fact(
        stage=stage,
        fact_key=key,
        fact_type=fact_type,
        subject=key,
        summary=f"summary for {key}",
        source_uri=source,
        source_kind="official_docs",
        source_digest=digest,
        confidence=0.9,
    )


def test_learning_reader_separates_current_and_stale_by_type(tmp_path):
    memory = _memory(tmp_path)
    source = "https://docs.python.org/3/library/pathlib.html"
    _fact(memory, key="python.pathlib.old", fact_type="verified_lesson", source=source, digest="old")
    _fact(memory, key="python.pathlib.current", fact_type="symbol", source=source, digest="new")
    _fact(
        memory,
        key="self.owner",
        fact_type="owner",
        source="repo://ARCHITECTURE.md",
        digest="repo-digest",
        stage="self",
    )

    result = LearningMemoryReader(tmp_path).get_learning_memory_summary(stage="python")

    assert result["counts"] == {"total": 2, "current": 1, "stale": 1}
    assert result["by_stage"] == [
        {"stage": "python", "total": 2, "current": 1, "stale": 1}
    ]
    by_type = {row["fact_type"]: row for row in result["by_fact_type"]}
    assert by_type["verified_lesson"]["stale"] == 1
    assert by_type["verified_lesson"]["current"] == 0
    assert by_type["symbol"]["current"] == 1
    assert by_type["symbol"]["stale"] == 0
    assert len(result["stale_samples"]) == 1
    assert result["stale_samples"][0]["fact_key"] == "python.pathlib.old"
    assert result["staleness"]["invalidation_history_available"] is False


def test_learning_reader_can_filter_fact_type_and_empty_state(tmp_path):
    memory = _memory(tmp_path)
    _fact(
        memory,
        key="python.sqlite3.connect",
        fact_type="symbol",
        source="https://docs.python.org/3/library/sqlite3.html",
        digest="one",
    )

    reader = LearningMemoryReader(tmp_path)
    filtered = reader.get_learning_memory_summary(stage="python", fact_type="symbol")
    missing = reader.get_learning_memory_summary(stage="qwen")

    assert filtered["counts"] == {"total": 1, "current": 1, "stale": 0}
    assert filtered["by_fact_type"][0]["fact_type"] == "symbol"
    assert missing["counts"] == {"total": 0, "current": 0, "stale": 0}
    assert missing["by_stage"] == []
    assert missing["by_fact_type"] == []


def test_learning_reader_exposes_the_real_memory_scope_and_writer_boundaries(tmp_path):
    memory = _memory(tmp_path)
    memory.record_human_lesson(
        "Keep factual claims tied to their source.",
        topic="evidence",
        source="owner",
    )
    memory.upsert_durable_learning(
        learning_key="reading:question:one",
        learning_type="question",
        subject="Memory integrity",
        summary="How should conflicting sources be reconciled?",
        source_uri="library://notes/page/1",
        source_kind="local_library",
        source_digest="abc",
        provenance="verified background reading",
        confidence=0.8,
    )

    result = LearningMemoryReader(tmp_path).get_learning_memory_summary()

    assert result["available"] is True
    assert result["store"] == "LearningMemory"
    assert result["read_interface"] == "get_learning_memory_summary"
    assert result["human_lessons"] == {"total": 1, "active": 1, "inactive": 0}
    assert result["durable_learnings"]["current"] == 1
    assert result["durable_learnings"]["by_type"][0]["learning_type"] == "question"
    assert result["self_development_memory"]["cycles"] == 0
    assert result["scope"]["ordinary_chat_auto_persistence"] is False
    assert "model weights" in result["scope"]["does_not_store"]
    assert "verified staged study" in result["scope"]["write_paths"]


def test_learning_reader_bounds_stale_samples_and_sources(tmp_path):
    memory = _memory(tmp_path)
    for index in range(20):
        source = f"https://docs.python.org/3/library/module{index}.html"
        _fact(
            memory,
            key=f"python.module{index}.old",
            fact_type="verified_lesson",
            source=source,
            digest="old",
        )
        _fact(
            memory,
            key=f"python.module{index}.current",
            fact_type="verified_lesson",
            source=source,
            digest="new",
        )

    result = LearningMemoryReader(tmp_path).get_learning_memory_summary(
        stage="python",
        sample_limit=999,
        source_limit=999,
    )

    assert result["counts"] == {"total": 40, "current": 20, "stale": 20}
    assert len(result["stale_samples"]) == 12
    assert len(result["top_sources"]) == 12


def test_learning_memory_summary_is_registered_read_only(tmp_path):
    _memory(tmp_path)
    tools = registry(tmp_path)

    spec = tools["get_learning_memory_summary"]
    assert spec.risk == RiskLevel.READ_ONLY
    assert "real LearningMemory" in spec.description
    result = spec.fn(stage="python")
    assert result["counts"] == {"total": 0, "current": 0, "stale": 0}


def test_ollama_client_generates_schema_for_learning_memory_summary(tmp_path, monkeypatch):
    _memory(tmp_path)
    request_json = {}

    def fake_request(_client, _response_type, *_args, **kwargs):
        request_json.update(kwargs["json"])
        return {}

    monkeypatch.setattr(Client, "_request", fake_request)
    tool = registry(tmp_path)["get_learning_memory_summary"].fn

    Client().chat(model="test-model", messages=[], tools=[tool])

    schema = request_json["tools"][0]["function"]
    assert schema["name"] == "get_learning_memory_summary"
    assert set(schema["parameters"]["properties"]) == {
        "stage",
        "fact_type",
        "sample_limit",
        "source_limit",
    }
