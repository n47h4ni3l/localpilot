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
    observation_count: int
    tool_rounds: int
    hard_budget: int
    runtime: dict[str, Any]


def _call_name(call: Any) -> str:
    function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(getattr(function, "name", "") or "")


def _call_arguments(call: Any) -> dict[str, Any]:
    function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(function, dict):
        return dict(function.get("arguments") or {})
    return dict(getattr(function, "arguments", None) or {})


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
    """Exercise multi-step planning -> retrieval -> reconciliation -> answer with unknowable facts.

    The values are generated after the model weights were trained and are supplied only by the
    synthetic read-only tools. The model must discover a manifest, retrieve three independently
    addressed fragments, validate a cross-fragment check, and answer inside the configured hard
    research budget. Passing therefore measures the runtime/tool/context path, not memorization of
    a repository-specific expected answer.
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
    fragment_ids = list(supplied.get("fragment_ids") or [secrets.token_hex(5) for _ in range(3)])
    if len(fragment_ids) != 3 or len(set(map(str, fragment_ids))) != 3:
        raise ValueError("cognition probe requires exactly three distinct fragment_ids")
    fragment_ids = [str(item) for item in fragment_ids]
    first_cut = max(1, len(nonce) // 3)
    second_cut = max(first_cut + 1, 2 * len(nonce) // 3)
    nonce_parts = (nonce[:first_cut], nonce[first_cut:second_cut], nonce[second_cut:])
    fragments = {
        fragment_ids[0]: {"position": 1, "left": left, "nonce_piece": nonce_parts[0]},
        fragment_ids[1]: {"position": 2, "right": right, "nonce_piece": nonce_parts[1]},
        fragment_ids[2]: {
            "position": 3,
            "xor_check": left ^ right,
            "nonce_piece": nonce_parts[2],
        },
    }

    def get_probe_manifest() -> str:
        """Return unpredictable fragment IDs and the reconciliation procedure for this probe run."""
        return json.dumps(
            {
                "fragment_ids": fragment_ids,
                "required_observations": 3,
                "procedure": (
                    "Read every distinct fragment ID. Order nonce_piece by position, compute left + right, "
                    "and verify xor_check equals left XOR right before answering."
                ),
            },
            sort_keys=True,
        )

    def get_probe_fragment(fragment_id: str) -> str:
        """Return one unpredictable fragment selected from the current probe manifest."""
        value = fragments.get(str(fragment_id))
        if value is None:
            return json.dumps({"error": "unknown fragment_id"}, sort_keys=True)
        return json.dumps(value, sort_keys=True)

    agent = LocalPilotAgent(config, root)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "This is a bounded LocalPilot cognition-runtime diagnostic. Use the supplied read-only "
                "tools to plan and acquire every required distinct observation. Do not guess facts or expose "
                "hidden reasoning. Reconcile the cross-fragment check, then return only compact JSON with "
                "keys sum and nonce."
            ),
        },
        {
            "role": "user",
            "content": (
                "Discover the unpredictable probe manifest, retrieve all of its fragments, reconcile them, "
                "then return the sum and exact reconstructed nonce without exceeding the hard research budget."
            ),
        },
    ]
    hard_budget = max(1, int(config.agent.research_hard_tool_rounds))
    observation_count = 0
    tool_rounds = 0
    manifest_read = False
    fragments_read: set[str] = set()
    seen_calls: set[tuple[str, str]] = set()
    content = ""
    runtime: dict[str, Any] = {}

    for turn_no in range(hard_budget + 4):
        allow_tools = tool_rounds < hard_budget
        response = agent._stream_chat_message(
            chat,
            think=config.model.think,
            tools=[get_probe_manifest, get_probe_fragment] if allow_tools else None,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 4096},
            phase="cognition_probe_research" if allow_tools else "cognition_probe_synthesize",
            turn_no=turn_no,
        )
        runtime = dict(agent._last_stream_runtime)
        messages.append(response)
        calls = list(response.get("tool_calls") or [])
        if not calls:
            content = str(response.get("content") or "").strip()
            break
        if not allow_tools:
            result = CognitionProbeResult(
                False,
                "hard_budget",
                "The model requested another observation after the hard research budget.",
                True,
                observation_count,
                tool_rounds,
                hard_budget,
                runtime,
            )
            agent.audit.write(
                "cognition_probe",
                ok=False,
                stage=result.stage,
                observation_count=observation_count,
                tool_rounds=tool_rounds,
                hard_budget=hard_budget,
                **runtime,
            )
            return result

        unique_this_round = False
        for call in calls:
            name = _call_name(call)
            arguments = _call_arguments(call)
            key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if key in seen_calls:
                tool_result = "Exact duplicate probe observation; reuse the earlier raw result."
            elif name == "get_probe_manifest":
                seen_calls.add(key)
                unique_this_round = True
                observation_count += 1
                manifest_read = True
                tool_result = get_probe_manifest()
            elif name == "get_probe_fragment":
                seen_calls.add(key)
                unique_this_round = True
                observation_count += 1
                fragment_id = str(arguments.get("fragment_id") or "")
                if fragment_id in fragments:
                    fragments_read.add(fragment_id)
                tool_result = get_probe_fragment(fragment_id)
            else:
                tool_result = f"Unknown cognition-probe tool: {name}"
            messages.append({"role": "tool", "tool_name": name, "content": tool_result})
        if unique_this_round:
            tool_rounds += 1
    else:
        content = ""

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
            observation_count > 0,
            observation_count,
            tool_rounds,
            hard_budget,
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
            observation_count > 0,
            observation_count,
            tool_rounds,
            hard_budget,
            runtime,
        )
        agent.audit.write("cognition_probe", ok=False, stage=result.stage, **runtime)
        return result

    complete_research = manifest_read and fragments_read == set(fragment_ids) and observation_count >= 4
    ok = complete_research and observed_sum == expected_sum and observed_nonce == nonce and tool_rounds <= hard_budget
    result = CognitionProbeResult(
        ok,
        "passed" if ok else ("incomplete_research" if not complete_research else "reasoning_result"),
        (
            "Multi-step planning, four distinct observations, reconciliation, and the final answer all "
            "matched the unpredictable facts within the hard research budget."
            if ok
            else "The model answered, but research was incomplete or its reconciled result did not match all raw observations."
        ),
        observation_count > 0,
        observation_count,
        tool_rounds,
        hard_budget,
        runtime,
    )
    agent.audit.write(
        "cognition_probe",
        ok=ok,
        stage=result.stage,
        observation_count=observation_count,
        tool_rounds=tool_rounds,
        hard_budget=hard_budget,
        **runtime,
    )
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
    print(
        f"Research: observations={result.observation_count}, "
        f"tool_rounds={result.tool_rounds}/{result.hard_budget}"
    )
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
