from __future__ import annotations

import re
import time
from pathlib import Path

from localpilot.agent import _ollama_memory_embedder
from localpilot.learning import LearningMemory


def _record(
    memory: LearningMemory,
    *,
    key: str,
    subject: str,
    summary: str,
    stage: str = "self",
    source_uri: str | None = None,
):
    memory.upsert_knowledge_fact(
        stage=stage,
        fact_key=key,
        fact_type="architecture_contract",
        subject=subject,
        summary=summary,
        source_uri=source_uri or f"repo://{stage}/{key}.md",
        source_kind="committed_repository",
        source_digest=f"digest-{stage}-{key}",
        confidence=0.95,
    )


class ConceptEmbedder:
    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [0.0] * self.dimensions
        if (
            "human-reviewed merge" in lowered
            or "proposed upgrade becomes active" in lowered
        ):
            vector[0] = 1.0
            return vector
        if "commandrunner" in lowered or "command runner" in lowered:
            vector[1] = 1.0
            return vector
        match = re.search(r"\b(?:concept|scenario)(\d+)\b", lowered)
        if match and int(match.group(1)) < self.dimensions - 2:
            vector[int(match.group(1)) + 2] = 1.0
            return vector
        vector[-1] = 1.0
        return vector


def _hybrid(path: Path, embedder: ConceptEmbedder) -> LearningMemory:
    return LearningMemory(
        path,
        embedding_provider=embedder,
        embedding_model="test-embedding-v1",
        semantic_min_similarity=0.7,
        embedding_batch_size=16,
        embedding_migration_limit=256,
    )


def test_ollama_embedder_uses_the_batch_api_without_model_management(monkeypatch):
    calls = []

    def fake_embed(**kwargs):
        calls.append(kwargs)
        return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    monkeypatch.setattr("ollama.embed", fake_embed)

    provider = _ollama_memory_embedder("embeddinggemma", "5m")

    assert provider(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert calls == [
        {
            "model": "embeddinggemma",
            "input": ["first", "second"],
            "truncate": True,
            "keep_alive": "5m",
        }
    ]


def test_paraphrase_retrieval_improves_without_replacing_lexical_search(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    lexical = LearningMemory(path)
    _record(
        lexical,
        key="promotion-boundary",
        subject="human promotion boundary",
        summary=(
            "Only a human-reviewed merge may promote an isolated candidate into "
            "the stable runtime."
        ),
    )
    _record(
        lexical,
        key="weather",
        subject="weather observation",
        summary="Read current conditions from a live local source.",
    )
    query = "Who decides whether a proposed upgrade becomes active?"

    assert lexical.search_knowledge_facts(query) == []

    hybrid = _hybrid(path, ConceptEmbedder())
    results = hybrid.search_knowledge_facts(query)

    assert results[0].fact_key == "promotion-boundary"
    assert hybrid.last_retrieval_diagnostics.mode == "hybrid"
    assert hybrid.last_retrieval_diagnostics.indexed_facts == 2


def test_exact_authority_sensitive_retrieval_is_not_displaced(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    lexical = LearningMemory(path)
    _record(
        lexical,
        key="command-runner",
        subject="CommandRunner",
        summary="CommandRunner executes approved argument vectors with shell disabled.",
    )
    _record(
        lexical,
        key="related-runner",
        subject="Operator action dispatcher",
        summary="A command runner concept may route reversible operations.",
    )
    lexical_top = lexical.search_knowledge_facts("CommandRunner", limit=2)[0].fact_key

    hybrid = _hybrid(path, ConceptEmbedder())
    hybrid_top = hybrid.search_knowledge_facts("CommandRunner", limit=2)[0].fact_key

    assert lexical_top == "command-runner"
    assert hybrid_top == lexical_top


def test_semantic_retrieval_preserves_stage_staleness_and_provenance(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    memory = LearningMemory(path)
    _record(
        memory,
        key="promotion",
        subject="human promotion boundary",
        summary="Only a human-reviewed merge changes stable code.",
        stage="self",
        source_uri="repo://SECURITY.md",
    )
    _record(
        memory,
        key="promotion",
        subject="Python package promotion",
        summary="A human-reviewed merge publishes the package.",
        stage="python",
        source_uri="https://docs.python.org/example",
    )
    memory.invalidate_knowledge_source("repo://SECURITY.md", "new-digest")
    hybrid = _hybrid(path, ConceptEmbedder())

    results = hybrid.search_knowledge_facts(
        "Who decides whether a proposed upgrade becomes active?",
        stage="self",
        include_stale=True,
    )

    assert len(results) == 1
    assert results[0].stage == "self"
    assert results[0].stale is True
    assert results[0].source_uri == "repo://SECURITY.md"
    assert hybrid.search_knowledge_facts(
        "Who decides whether a proposed upgrade becomes active?",
        stage="self",
        include_stale=False,
    ) == []


def test_lazy_migration_reindexes_changed_fact_digests(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    memory = LearningMemory(path)
    _record(
        memory,
        key="promotion-boundary",
        subject="human promotion boundary",
        summary="Only a human-reviewed merge changes stable code.",
    )
    embedder = ConceptEmbedder()
    hybrid = _hybrid(path, embedder)

    assert hybrid.knowledge_embedding_count() == 0
    hybrid.search_knowledge_facts("Who decides whether a proposed upgrade becomes active?")
    assert hybrid.knowledge_embedding_count() == 1
    first_document_batches = len(embedder.calls)

    _record(
        hybrid,
        key="promotion-boundary",
        subject="human promotion boundary",
        summary="A human-reviewed merge remains the only stable promotion path.",
    )
    hybrid.search_knowledge_facts("Who decides whether a proposed upgrade becomes active?")

    assert hybrid.knowledge_embedding_count() == 1
    assert len(embedder.calls) > first_document_batches
    assert hybrid.last_retrieval_diagnostics.indexed_facts == 1


def test_embedding_failure_falls_back_once_to_identical_lexical_results(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    lexical = LearningMemory(path)
    _record(
        lexical,
        key="command-runner",
        subject="CommandRunner",
        summary="CommandRunner executes approved argument vectors.",
    )
    expected = [
        fact.fact_key
        for fact in lexical.search_knowledge_facts("CommandRunner", limit=4)
    ]
    calls = []

    def unavailable(texts):
        calls.append(list(texts))
        raise ConnectionError("embedding model unavailable")

    hybrid = LearningMemory(
        path,
        embedding_provider=unavailable,
        embedding_model="missing-model",
    )
    actual = [
        fact.fact_key
        for fact in hybrid.search_knowledge_facts("CommandRunner", limit=4)
    ]
    second = [
        fact.fact_key
        for fact in hybrid.search_knowledge_facts("CommandRunner", limit=4)
    ]

    assert actual == expected == second
    assert hybrid.last_retrieval_diagnostics.mode == "lexical_fallback"
    assert hybrid.last_retrieval_diagnostics.error_type == "ConnectionError"
    assert len(calls) == 1


def test_cached_hybrid_benchmark_improves_paraphrase_recall_without_exact_regression(
    tmp_path: Path,
):
    path = tmp_path / "learning.sqlite3"
    lexical = LearningMemory(path)
    for index in range(20):
        _record(
            lexical,
            key=f"capability-{index}",
            subject=f"Capability{index}",
            summary=f"concept{index} governs a distinct retained architecture decision.",
        )
    embedder = ConceptEmbedder()
    hybrid = _hybrid(path, embedder)

    lexical_paraphrase_hits = sum(
        bool(results) and results[0].fact_key == f"capability-{index}"
        for index in range(20)
        for results in [
            lexical.search_knowledge_facts(
                f"Explain scenario{index} without using the stored terminology",
                limit=1,
            )
        ]
    )
    hybrid_paraphrase_hits = sum(
        bool(results) and results[0].fact_key == f"capability-{index}"
        for index in range(20)
        for results in [
            hybrid.search_knowledge_facts(
                f"Explain scenario{index} without using the stored terminology",
                limit=1,
            )
        ]
    )
    lexical_exact_hits = sum(
        lexical.search_knowledge_facts(f"Capability{index}", limit=1)[0].fact_key
        == f"capability-{index}"
        for index in range(20)
    )

    started = time.perf_counter()
    hybrid_exact_hits = sum(
        hybrid.search_knowledge_facts(f"Capability{index}", limit=1)[0].fact_key
        == f"capability-{index}"
        for index in range(20)
    )
    cached_latency_ms = (time.perf_counter() - started) * 1000

    assert lexical_paraphrase_hits == 0
    assert hybrid_paraphrase_hits == 20
    assert hybrid_exact_hits == lexical_exact_hits == 20
    assert cached_latency_ms < 1000
