from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config, load_config


@dataclass(frozen=True, slots=True)
class CognitionProbeResult:
    ok: bool
    stage: str
    detail: str
    tool_called: bool
    runtime: dict[str, Any]


def _call_name(call: Any) -> str:
    function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(getattr(function, "name", "") or "")


def _json_object(text: str) -> dict[str, Any]:
    candidate = str(text).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("final answer did not contain a JSON object")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("final answer JSON was not an object")
    return value


def run_cognition_probe(
    config: Config,
    root: str | Path,
    *,
    chat: Callable[..., Any] | None = None,
    facts: dict[str, Any] | None = None,
) -> CognitionProbeResult:
    """Exercise tool acquisition -> same-model reasoning -> final answer with unknowable facts.

    The values are generated after the model weights were trained and are supplied only by the
    synthetic read-only tool. Passing this probe therefore measures the runtime/tool/context path,
    not memorization of a repository-specific expected answer.
    """
    if config.model.provider.lower() != "ollama":
        raise RuntimeError("Cognition probe currently supports Ollama only.")
    if chat is None:
        try:
            from ollama import chat as ollama_chat
        except ImportError as exc:
            raise RuntimeError("Ollama Python package is not installed.") from exc
        chat = ollama_chat

    supplied = dict(facts or {})
    left = int(supplied.get("left", 100 + secrets.randbelow(900)))
    right = int(supplied.get("right", 100 + secrets.randbelow(900)))
    nonce = str(supplied.get("nonce") or secrets.token_hex(8))
    expected_sum = left + right

    def get_probe_facts() -> str:
        """Return this run's unpredictable read-only cognition-probe facts."""
        return json.dumps({"left": left, "right": right, "nonce": nonce}, sort_keys=True)

    agent = LocalPilotAgent(config, root)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "This is a bounded LocalPilot cognition-runtime diagnostic. Use the supplied read-only "
                "tool to acquire the facts. Do not guess them and do not expose hidden reasoning. After "
                "reading the facts, compute left + right and return only compact JSON with keys sum and nonce."
            ),
        },
        {
            "role": "user",
            "content": (
                "Retrieve the unpredictable probe facts, reason over the values you actually receive, "
                "then return their sum and the exact nonce."
            ),
        },
    ]

    first = agent._stream_chat_message(
        chat,
        think=config.model.think,
        tools=[get_probe_facts],
        messages=messages,
        options={"temperature": 0.0, "num_predict": 4096},
        phase="cognition_probe_acquire",
        turn_no=0,
    )
    messages.append(first)
    calls = list(first.get("tool_calls") or [])
    if not calls:
        runtime = dict(agent._last_stream_runtime)
        result = CognitionProbeResult(
            False,
            "tool_acquisition",
            "Model did not call the unpredictable read-only probe tool.",
            False,
            runtime,
        )
        agent.audit.write("cognition_probe", ok=False, stage=result.stage, **runtime)
        return result

    correct_call = next((call for call in calls if _call_name(call) == "get_probe_facts"), None)
    if correct_call is None:
        runtime = dict(agent._last_stream_runtime)
        result = CognitionProbeResult(
            False,
            "tool_selection",
            "Model called a tool, but not the supplied probe-facts tool.",
            True,
            runtime,
        )
        agent.audit.write("cognition_probe", ok=False, stage=result.stage, **runtime)
        return result

    messages.append({"role": "tool", "tool_name": "get_probe_facts", "content": get_probe_facts()})
    second = agent._stream_chat_message(
        chat,
        think=config.model.think,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 4096},
        phase="cognition_probe_synthesize",
        turn_no=1,
    )
    runtime = dict(agent._last_stream_runtime)
    content = str(second.get("content") or "").strip()
    if not content:
        stage = (
            "generation_limit"
            if runtime.get("runtime_classification") == "generation_limit"
            else "reasoning_without_answer"
        )
        result = CognitionProbeResult(
            False,
            stage,
            "The model received the tool result but produced no final answer.",
            True,
            runtime,
        )
        agent.audit.write("cognition_probe", ok=False, stage=result.stage, **runtime)
        return result

    try:
        value = _json_object(content)
        observed_sum = int(value.get("sum"))
        observed_nonce = str(value.get("nonce") or "")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        result = CognitionProbeResult(
            False,
            "answer_format",
            f"The final answer was not valid probe JSON: {exc}",
            True,
            runtime,
        )
        agent.audit.write("cognition_probe", ok=False, stage=result.stage, **runtime)
        return result

    ok = observed_sum == expected_sum and observed_nonce == nonce
    result = CognitionProbeResult(
        ok,
        "passed" if ok else "reasoning_result",
        (
            "Tool acquisition, same-model reasoning, and final answer all matched the unpredictable facts."
            if ok
            else "The model answered, but its computed result or copied nonce did not match the tool evidence."
        ),
        True,
        runtime,
    )
    agent.audit.write("cognition_probe", ok=ok, stage=result.stage, **runtime)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m localpilot.cognition_probe")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_cognition_probe(load_config(args.config), root)
    state = "PASS" if result.ok else "FAIL"
    print(f"LocalPilot cognition probe: {state}")
    print(f"Stage: {result.stage}")
    print(result.detail)
    runtime = result.runtime
    print(
        "Runtime: "
        f"done_reason={runtime.get('done_reason')!r}, "
        f"classification={runtime.get('runtime_classification')}, "
        f"prompt_tokens={runtime.get('prompt_eval_count')}, "
        f"generated_tokens={runtime.get('eval_count')}, "
        f"context_used={runtime.get('context_used_percent')}%, "
        f"num_predict={runtime.get('num_predict')}, "
        f"reasoning_chars={runtime.get('reasoning_chars')}, "
        f"content_chars={runtime.get('content_chars')}"
    )
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
