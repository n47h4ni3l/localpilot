"""Stateless turn-budget/memory-tuning constants and the embedding-provider
factory for LocalPilotAgent.

Extracted verbatim (no logic changes) from agent.py as part of the low-risk
mechanical decomposition. _ollama_memory_embedder is re-imported as a bare
name into agent.py, where __init__ still calls it exactly as before to
build the LearningMemory embedding_provider. ask() and
_continue_high_reasoning_answer() were not touched."""

from collections.abc import Callable

_REPEATED_UNHELPFUL_TOOL_LIMIT = 2
_LIBRARY_SEARCHES_PER_TURN = 2


def _ollama_memory_embedder(
    model: str,
    keep_alive: float | str,
) -> Callable[[list[str]], list[list[float]]]:
    """Bind the official batch embedding API without pulling any model."""

    def embed_texts(texts: list[str]) -> list[list[float]]:
        from ollama import embed

        response = embed(
            model=model,
            input=texts,
            truncate=True,
            keep_alive=keep_alive,
        )
        values = (
            response.get("embeddings")
            if isinstance(response, dict)
            else getattr(response, "embeddings", None)
        )
        return [list(vector) for vector in (values or [])]

    return embed_texts


_LEARNING_MEMORY_FACT_LIMIT = 6
_LEARNING_MEMORY_CHAR_BUDGET = 6000
_LEARNING_MEMORY_SOFT_TOOL_ROUNDS = 4
_LEARNING_MEMORY_HARD_TOOL_ROUNDS = 4
_PUBLIC_WEB_FETCHES_PER_TURN = 6
