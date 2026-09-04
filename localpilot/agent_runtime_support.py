"""Stateless runtime/generation-support constants and helpers for LocalPilotAgent.

Extracted verbatim (no logic changes) from agent.py as part of the low-risk
mechanical decomposition. Constants and _int_or_none are re-imported into
agent.py as bare names where still needed there; the four staticmethods are
left behind on LocalPilotAgent as staticmethod(...) shims. ask() and
_continue_high_reasoning_answer() were not touched."""

from typing import Any

_STREAM_RUNTIME_FIELDS = (
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)

_OPERATOR_NUM_PREDICT = 2048
_FINAL_ANSWER_NUM_PREDICT = 4096
_GENERATION_LIMIT_CONTINUATION_CEILING = 8192
_GENERATION_LIMIT_CONTINUATION_MINIMUM = 256
_TOOL_CALL_PROTOCOL_RETRY_LIMIT = 2


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _classify_runtime(
    *,
    done_reason: str,
    eval_count: int | None,
    num_predict: int | None,
    context_used_percent: float | None,
) -> str:
    if done_reason.lower() == "length":
        return "generation_limit"
    if num_predict is not None and num_predict > 0 and eval_count is not None and eval_count >= num_predict:
        return "generation_limit"
    if context_used_percent is not None and context_used_percent >= 90.0:
        return "context_pressure"
    if done_reason:
        return f"done:{done_reason.lower()}"
    return "unknown"


def _is_tool_call_protocol_error(exc: Exception) -> bool:
    """Recognize only Ollama response errors that explicitly identify tool-call parsing."""
    try:
        from ollama import ResponseError
    except ImportError:
        return False
    if not isinstance(exc, ResponseError):
        return False
    detail = str(getattr(exc, "error", "") or exc).lower().replace("_", " ")
    mentions_tool_call = any(
        marker in detail for marker in ("tool call", "tool-call", "tool calls")
    )
    mentions_protocol_failure = any(
        marker in detail
        for marker in (
            "parse",
            "parsing",
            "protocol",
            "invalid json",
            "invalid character",
            "malformed",
            "decode",
        )
    )
    return mentions_tool_call and mentions_protocol_failure


def _visible_decline(content: str) -> str:
    stripped = content.strip()
    if stripped.upper().startswith("DECLINE:"):
        reason = stripped.split(":", 1)[1].strip() or "no reason was provided"
        return f"[LocalPilot chose not to answer: {reason}]"
    return content


def _generation_limit_continuation_budget(runtime: dict[str, Any]) -> int:
    """Bound one continuation within the measured live-context headroom."""
    context_tokens = _int_or_none(runtime.get("context_tokens"))
    prompt_tokens = _int_or_none(runtime.get("prompt_eval_count"))
    generated_tokens = _int_or_none(runtime.get("eval_count"))
    if context_tokens is None or context_tokens <= 0 or prompt_tokens is None:
        return 0

    # The next prompt contains the previous prompt, its reasoning-only completion, and a
    # short continuation instruction. Reserve five percent of the window (at least 1K)
    # for serialization/token-count variance and that instruction before allocating output.
    safety_margin = max(1024, context_tokens // 20)
    estimated_next_prompt = prompt_tokens + max(0, generated_tokens or 0)
    available = context_tokens - estimated_next_prompt - safety_margin
    if available < _GENERATION_LIMIT_CONTINUATION_MINIMUM:
        return 0
    return min(_GENERATION_LIMIT_CONTINUATION_CEILING, available)

