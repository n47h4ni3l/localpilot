"""Ollama model selection and chat helpers for the self-development pipeline.

This module owns developer-model selection plus the inference-time streaming
safety boundary. Background admission remains conservative before a model is
started; once an admitted inference is running, expected model residency may
cross that background threshold until a separate emergency memory boundary is
reached. Foreground-user preemption remains immediate.

Note preserved as-is, not fixed here: installed_ollama_models is typed to
return dict[str, int | None] but returns a bare set() on its except path.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import psutil

from localpilot.selfdev_results import CyclePaused


_INFERENCE_EMERGENCY_MEMORY_PERCENT = 94.0
_INFERENCE_MIN_AVAILABLE_GIB = 2.0
_GIB = 1024**3
_GROUNDING_PLANNER_MARKER = "pre-implementation repository-grounding planner"
_GROUNDING_FINALIZATION_AFTER_TOOL_TURNS = 3


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
    """Select a model after background admission, bounded by inference safety.

    ``max_memory_percent`` is the conservative background-admission threshold.
    It applies to memory already in use before LocalPilot loads a developer
    model. Expected model residency is instead checked against the separate
    inference emergency ceiling and minimum-free-memory reserve used by the
    streaming guard.
    """
    candidates: list[str] = []
    for name in (preferred, everyday, *fallbacks):
        normalized = str(name).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    total = max(1, int(total_memory_bytes))
    available = max(0, min(int(available_memory_bytes), total))
    used = max(0, total - available)
    admission_percent = max(0.0, min(float(max_memory_percent), 100.0))
    admission_ceiling = total * admission_percent / 100.0
    current_percent = used * 100.0 / total

    if used > admission_ceiling:
        return DeveloperModelSelection(
            None,
            None,
            current_percent,
            f"Current memory {current_percent:.1f}% already exceeds the "
            f"{admission_percent:.1f}% background admission ceiling; "
            "developer-model loading was not attempted.",
        )

    inference_ceiling = total * _INFERENCE_EMERGENCY_MEMORY_PERCENT / 100.0
    minimum_available = int(_INFERENCE_MIN_AVAILABLE_GIB * _GIB)
    rejected: list[str] = []
    resident = {str(name).strip() for name in resident_models}
    if everyday in resident and everyday in candidates:
        candidates = [everyday, *(name for name in candidates if name != everyday)]

    def projection_status(projected: int) -> tuple[bool, float, float]:
        projected_percent = projected * 100.0 / total
        projected_available = max(0, total - projected)
        projected_available_gib = projected_available / _GIB
        safe = (
            projected < inference_ceiling
            and projected_available >= minimum_available
        )
        return safe, projected_percent, projected_available_gib

    def rejection_detail(name: str, projected_percent: float, available_gib: float) -> str:
        reasons: list[str] = []
        if projected_percent >= _INFERENCE_EMERGENCY_MEMORY_PERCENT:
            reasons.append(
                f"{projected_percent:.1f}% >= "
                f"{_INFERENCE_EMERGENCY_MEMORY_PERCENT:.1f}% inference emergency ceiling"
            )
        if available_gib < _INFERENCE_MIN_AVAILABLE_GIB:
            reasons.append(
                f"{available_gib:.2f} GiB available < "
                f"{_INFERENCE_MIN_AVAILABLE_GIB:.1f} GiB reserve"
            )
        return f"{name} would project memory to " + " and ".join(reasons)

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
            safe, projected_percent, projected_available_gib = projection_status(projected)
            if safe:
                skipped = f"Skipped {'; '.join(rejected)}. " if rejected else ""
                return DeveloperModelSelection(
                    name,
                    size,
                    projected_percent,
                    f"{skipped}Reused resident {name}; current memory passed the "
                    f"{admission_percent:.1f}% background admission ceiling and projected "
                    f"incremental memory is {projected_percent:.1f}% with "
                    f"{projected_available_gib:.2f} GiB available, inside the "
                    f"{_INFERENCE_EMERGENCY_MEMORY_PERCENT:.1f}% / "
                    f"{_INFERENCE_MIN_AVAILABLE_GIB:.1f} GiB inference safety boundary. "
                    "This preserves foreground model residency.",
                )
            rejected.append(
                rejection_detail(
                    f"resident {name} plus context overhead",
                    projected_percent,
                    projected_available_gib,
                )
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
        safe, projected_percent, projected_available_gib = projection_status(projected)
        if safe:
            skipped = f"Skipped {'; '.join(rejected)}. " if rejected else ""
            return DeveloperModelSelection(
                name,
                size,
                projected_percent,
                f"{skipped}Selected {name}; current memory passed the "
                f"{admission_percent:.1f}% background admission ceiling and projected "
                f"loaded-model memory is {projected_percent:.1f}% with "
                f"{projected_available_gib:.2f} GiB available, inside the "
                f"{_INFERENCE_EMERGENCY_MEMORY_PERCENT:.1f}% / "
                f"{_INFERENCE_MIN_AVAILABLE_GIB:.1f} GiB inference safety boundary.",
            )
        rejected.append(
            rejection_detail(name, projected_percent, projected_available_gib)
        )

    detail = "; ".join(rejected) or "no configured model candidates were provided"
    return DeveloperModelSelection(
        None,
        None,
        None,
        f"No installed developer model fits the inference memory safety budget after "
        f"passing the {admission_percent:.1f}% background admission gate: {detail}.",
    )


def _run_inference_guard(stream_guard: Callable[[], None] | None) -> bool:
    """Run the active-inference guard and report suppressed admission pressure.

    ``True`` means the ordinary background memory ceiling was exceeded only
    because an already-admitted model is resident, while the separate
    inference emergency boundary still has safe headroom.
    """
    if stream_guard is None:
        return False
    try:
        stream_guard()
    except CyclePaused as exc:
        parts = [part.strip().casefold() for part in str(exc).split(";") if part.strip()]
        if not parts or not all(part.startswith("memory ") for part in parts):
            raise
        try:
            vm = psutil.virtual_memory()
            memory_percent = float(vm.percent)
            available_gib = float(vm.available) / _GIB
        except Exception:
            raise
        if (
            memory_percent < _INFERENCE_EMERGENCY_MEMORY_PERCENT
            and available_gib >= _INFERENCE_MIN_AVAILABLE_GIB
        ):
            return True
        raise CyclePaused(
            f"memory emergency during inference: {memory_percent:.1f}% used, "
            f"{available_gib:.2f} GiB available; hard stop at "
            f"{_INFERENCE_EMERGENCY_MEMORY_PERCENT:.1f}% or below "
            f"{_INFERENCE_MIN_AVAILABLE_GIB:.1f} GiB available"
        ) from exc
    return False


def _unload_ollama_model(model_name: str) -> None:
    """Best-effort unload for a model LocalPilot itself just made resident."""
    if not model_name:
        return
    try:
        from ollama import generate

        generate(model=model_name, keep_alive=0)
    except Exception:
        pass


def _prepare_grounding_finalization(call_kwargs: dict[str, Any]) -> None:
    """Use the last grounding turn for structured synthesis instead of another tool call."""
    tools = call_kwargs.get("tools")
    messages = call_kwargs.get("messages")
    if not tools or not isinstance(messages, list):
        return

    system_text = ""
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "system":
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            system_text = str(content or "").casefold()
            break
    if _GROUNDING_PLANNER_MARKER not in system_text:
        return

    tool_turns = 0
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "assistant":
            continue
        calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        if calls:
            tool_turns += 1
    if tool_turns < _GROUNDING_FINALIZATION_AFTER_TOOL_TURNS:
        return

    final_messages = list(messages)
    final_messages.append(
        {
            "role": "user",
            "content": (
                "Repository inspection budget is exhausted. Do not call any more tools. "
                "Using only the repository evidence already present in this conversation, "
                "return the required single strict JSON object now. Use empty lists for "
                "unsupported claim classes. Do not invent evidence, prose, markdown, or "
                "hidden reasoning."
            ),
        }
    )
    call_kwargs["messages"] = final_messages
    call_kwargs.pop("tools", None)
    call_kwargs["format"] = "json"
    options = dict(call_kwargs.get("options") or {})
    options["temperature"] = 0.0
    call_kwargs["options"] = options


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

    model_name = str(kwargs.get("model") or "")
    manage_residency = (
        stream_guard is not None
        and bool(model_name)
        and keep_alive is not None
        and ollama_keep_alive_seconds(keep_alive) > 0
    )
    model_was_resident = (
        model_name in running_ollama_models()
        if manage_residency
        else False
    )

    def release_new_model_if_needed(*, pressure_seen: bool, failed: bool) -> None:
        if (
            manage_residency
            and not model_was_resident
            and (pressure_seen or failed)
        ):
            _unload_ollama_model(model_name)

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
        memory_pressure_seen = False
        model_started = False
        failed = False
        try:
            memory_pressure_seen |= _run_inference_guard(stream_guard)
            response_stream = await client.chat(**call_kwargs)
            model_started = True
            pending_chunk = asyncio.create_task(anext(response_stream))
            while True:
                done, _ = await asyncio.wait(
                    {pending_chunk},
                    timeout=max(0.01, float(guard_poll_seconds)),
                )
                if not done:
                    memory_pressure_seen |= _run_inference_guard(stream_guard)
                    continue
                try:
                    chunk = pending_chunk.result()
                except StopAsyncIteration:
                    break
                memory_pressure_seen |= _run_inference_guard(stream_guard)
                add_chunk(chunk, content, thinking, tool_calls)
                pending_chunk = asyncio.create_task(anext(response_stream))
        except BaseException:
            failed = model_started
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
            release_new_model_if_needed(
                pressure_seen=memory_pressure_seen,
                failed=failed,
            )
        return merged_response(content, thinking, tool_calls)

    def invoke(*, think: bool | str | None) -> Any:
        call_kwargs = dict(kwargs)
        _prepare_grounding_finalization(call_kwargs)
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
        memory_pressure_seen = False
        failed = False
        try:
            for chunk in response_stream:
                memory_pressure_seen |= _run_inference_guard(stream_guard)
                add_chunk(chunk, content, thinking, tool_calls)
        except BaseException:
            failed = True
            raise
        finally:
            close = getattr(response_stream, "close", None)
            if callable(close):
                close()
            release_new_model_if_needed(
                pressure_seen=memory_pressure_seen,
                failed=failed,
            )
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
