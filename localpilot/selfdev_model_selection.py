"""Ollama model selection and chat helpers for the self-development pipeline.

Extracted verbatim (no logic changes) from selfdev.py: choosing which
developer model to use under a memory ceiling (select_developer_model,
select_resource_aware_developer_model, DeveloperModelSelection), reading
Ollama's own state (available/installed/running_ollama_models,
ollama_keep_alive_seconds), and driving a chat call with optional
thinking and preemptible streaming (developer_chat, StreamedChatResponse,
plus the three small response-parsing statics _content/_calls/_call_parts
carried over from SelfDeveloper as shims).

Note preserved as-is, not fixed here: installed_ollama_models is typed to
return dict[str, int | None] but returns a bare set() on its except path.
Flagging it rather than changing it -- this pass moves code, it doesn't
change behaviour, even to fix something that looks like a real bug."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class DeveloperModelSelection:
    model: str | None
    size_bytes: int | None
    projected_memory_percent: float | None
    reason: str


@dataclass(slots=True)
class StreamedChatResponse:
    message: dict[str, Any]


def select_developer_model(preferred: str, everyday: str, available: Iterable[str]) -> str:
    installed = {str(name).strip() for name in available}
    return preferred if preferred in installed else everyday


def available_ollama_models() -> set[str]:
    """Read model names through the Ollama SDK without invoking a shell."""
    return set(installed_ollama_models())


def installed_ollama_models() -> dict[str, int | None]:
    """Return installed Ollama model names and their on-disk byte sizes."""
    try:
        from ollama import list as list_models

        response = list_models()
    except Exception:
        return set()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    installed: dict[str, int | None] = {}
    for model in models or []:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
            size = model.get("size")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            size = getattr(model, "size", None)
        if name:
            try:
                size_bytes = int(size) if size is not None else None
            except (TypeError, ValueError):
                size_bytes = None
            installed[str(name)] = size_bytes
    return installed


def running_ollama_models() -> set[str]:
    """Return models currently resident in Ollama without starting one."""
    try:
        from ollama import ps

        response = ps()
    except Exception:
        return set()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    running: set[str] = set()
    for model in models or []:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            running.add(str(name))
    return running


def ollama_keep_alive_seconds(value: float | str) -> float:
    """Parse the bounded Ollama duration forms used by LocalPilot."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float("inf") if float(value) < 0 else float(value)
    text = str(value).strip().lower()
    if text in {"-1", "-1s"}:
        return float("inf")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", text)
    if not match:
        return 0.0
    amount = float(match.group(1))
    multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, None: 1.0}
    return amount * multiplier[match.group(2)]


def select_resource_aware_developer_model(
    preferred: str,
    everyday: str,
    fallbacks: Iterable[str],
    installed: dict[str, int | None],
    *,
    total_memory_bytes: int,
    available_memory_bytes: int,
    max_memory_percent: float,
    overhead_bytes: int = 0,
    resident_models: Iterable[str] = (),
) -> DeveloperModelSelection:
    """Select the first configured model that preserves the memory ceiling."""
    candidates: list[str] = []
    for name in (preferred, everyday, *fallbacks):
        normalized = str(name).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    total = max(1, int(total_memory_bytes))
    available = max(0, int(available_memory_bytes))
    used = max(0, total - available)
    ceiling = total * max(0.0, min(float(max_memory_percent), 100.0)) / 100.0
    rejected: list[str] = []
    resident = {str(name).strip() for name in resident_models}
    if everyday in resident and everyday in candidates:
        candidates = [everyday, *(name for name in candidates if name != everyday)]

    for name in candidates:
        if name not in installed:
            rejected.append(f"{name} is not installed")
            continue
        size = installed[name]
        if size is None:
            rejected.append(f"{name} has no usable size metadata")
            continue
        if name in resident:
            projected = used + max(0, int(overhead_bytes))
            projected_percent = projected * 100.0 / total
            if projected <= ceiling:
                skipped = f"Skipped {'; '.join(rejected)}. " if rejected else ""
                return DeveloperModelSelection(
                    name,
                    size,
                    projected_percent,
                    f"{skipped}Reused resident {name}; projected incremental memory "
                    f"{projected_percent:.1f}% within the {max_memory_percent:.1f}% "
                    "background ceiling. This preserves foreground model residency.",
                )
            rejected.append(
                f"resident {name} plus context overhead would project memory to "
                f"{projected_percent:.1f}% > {max_memory_percent:.1f}%"
            )
            return DeveloperModelSelection(
                None,
                None,
                projected_percent,
                "Preserved the resident foreground model instead of evicting it for a fallback: "
                + rejected[-1]
                + ".",
            )
        projected = used + max(0, int(size)) + max(0, int(overhead_bytes))
        projected_percent = projected * 100.0 / total
        if projected <= ceiling:
            skipped = f"Skipped {'; '.join(rejected)}. " if rejected else ""
            return DeveloperModelSelection(
                name,
                size,
                projected_percent,
                f"{skipped}Selected {name}; projected memory {projected_percent:.1f}% "
                f"within the {max_memory_percent:.1f}% background ceiling.",
            )
        rejected.append(
            f"{name} would project memory to {projected_percent:.1f}% "
            f"> {max_memory_percent:.1f}%"
        )

    detail = "; ".join(rejected) or "no configured model candidates were provided"
    return DeveloperModelSelection(
        None,
        None,
        None,
        f"No installed developer model fits the background memory budget: {detail}.",
    )


def developer_chat(
    chat: Callable[..., Any],
    *,
    request_think: bool | str,
    context_tokens: int | None = None,
    keep_alive: float | str | None = None,
    stream_guard: Callable[[], None] | None = None,
    preempt_before_first_chunk: bool = False,
    guard_poll_seconds: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Use thinking when supported and permit prompt cancellation while streaming."""

    def add_chunk(
        chunk: Any,
        content: list[str],
        thinking: list[str],
        tool_calls: list[Any],
    ) -> None:
        message = getattr(chunk, "message", chunk)
        if isinstance(message, dict) and isinstance(message.get("message"), dict):
            message = message["message"]
        if isinstance(message, dict):
            content.append(str(message.get("content") or ""))
            thinking.append(str(message.get("thinking") or ""))
            tool_calls.extend(list(message.get("tool_calls") or []))
        else:
            content.append(str(getattr(message, "content", "") or ""))
            thinking.append(str(getattr(message, "thinking", "") or ""))
            tool_calls.extend(list(getattr(message, "tool_calls", None) or []))

    def merged_response(
        content: list[str],
        thinking: list[str],
        tool_calls: list[Any],
    ) -> StreamedChatResponse:
        message = {"role": "assistant", "content": "".join(content), "tool_calls": tool_calls}
        if any(thinking):
            message["thinking"] = "".join(thinking)
        return StreamedChatResponse(message)

    async def invoke_preemptible(call_kwargs: dict[str, Any]) -> Any:
        from ollama import AsyncClient

        client = AsyncClient()
        response_stream = None
        pending_chunk = None
        content: list[str] = []
        thinking: list[str] = []
        tool_calls: list[Any] = []
        try:
            if stream_guard is not None:
                stream_guard()
            response_stream = await client.chat(**call_kwargs)
            pending_chunk = asyncio.create_task(anext(response_stream))
            while True:
                done, _ = await asyncio.wait(
                    {pending_chunk},
                    timeout=max(0.01, float(guard_poll_seconds)),
                )
                if not done:
                    if stream_guard is not None:
                        stream_guard()
                    continue
                try:
                    chunk = pending_chunk.result()
                except StopAsyncIteration:
                    break
                if stream_guard is not None:
                    stream_guard()
                add_chunk(chunk, content, thinking, tool_calls)
                pending_chunk = asyncio.create_task(anext(response_stream))
        except BaseException:
            if pending_chunk is not None and not pending_chunk.done():
                pending_chunk.cancel()
                try:
                    await pending_chunk
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            if response_stream is not None:
                await response_stream.aclose()
            raise
        finally:
            await client.close()
        return merged_response(content, thinking, tool_calls)

    def invoke(*, think: bool | str | None) -> Any:
        call_kwargs = dict(kwargs)
        if context_tokens is not None:
            options = dict(call_kwargs.get("options") or {})
            options["num_ctx"] = int(context_tokens)
            call_kwargs["options"] = options
        if think is not None:
            call_kwargs["think"] = think
        if keep_alive is not None:
            call_kwargs["keep_alive"] = keep_alive
        if stream_guard is None:
            return chat(**call_kwargs)

        call_kwargs["stream"] = True
        if preempt_before_first_chunk:
            return asyncio.run(invoke_preemptible(call_kwargs))
        response_stream = chat(**call_kwargs)
        content: list[str] = []
        thinking: list[str] = []
        tool_calls: list[Any] = []
        try:
            for chunk in response_stream:
                stream_guard()
                add_chunk(chunk, content, thinking, tool_calls)
        finally:
            close = getattr(response_stream, "close", None)
            if callable(close):
                close()
        return merged_response(content, thinking, tool_calls)

    if request_think:
        model = str(kwargs.get("model") or "").lower()
        think: bool | str = request_think
        if "gpt-oss" not in model:
            think = True
        try:
            return invoke(think=think)
        except Exception as exc:
            message = str(exc).lower()
            if "does not support thinking" not in message:
                raise
    return invoke(think=None)


def _content(response: Any) -> str:
    message = getattr(response, "message", response)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _calls(response: Any) -> list[Any]:
    message = getattr(response, "message", response)
    if isinstance(message, dict):
        return list(message.get("tool_calls") or [])
    return list(getattr(message, "tool_calls", None) or [])


def _call_parts(call: Any) -> tuple[str, dict[str, Any]]:
    function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(function, dict):
        return str(function.get("name") or ""), dict(function.get("arguments") or {})
    return str(getattr(function, "name", "")), dict(getattr(function, "arguments", None) or {})

