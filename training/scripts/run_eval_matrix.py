#!/usr/bin/env python3
"""Three-arm, three-repeat evaluation without changing Eval v1's historical results.

The comparison is between *configurations*: unassisted frozen base, frozen base
with the same current LocalPilot scaffold, and accepted Nestra with that scaffold.
Only the final two arms isolate the change in model weights.
"""
from __future__ import annotations

import argparse
import copy
import gc
from contextvars import ContextVar
import hashlib
import json
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from training.scripts import run_eval_v1 as original

ROOT = original.ROOT
EXTRA_ROOT = ROOT / "training" / "evals_paired_v2"
ARMS = ("base_direct", "base_localpilot", "nestra_localpilot")
REPEATS = 3
CELL_EVIDENCE: ContextVar[dict[str, Any] | None] = ContextVar("cell_evidence", default=None)
BOUNDED_GENERATION_POLICY = {
    "name": "bounded-visible-final-v1", "per_turn_tokens": 3072,
    "direct_completion_tokens": 3072, "context_margin_tokens": 512,
    "direct_completion_think": False, "max_direct_completions": 1,
}


class EvaluatorIntegrityError(RuntimeError):
    """A captured model call did not preserve the selected weight identity."""


def _check_call_identity(model: str, expected: str | None, record: dict, phase: str) -> None:
    if expected is None:
        return
    try:
        actual = original.ollama_model_identity(model).get('digest')
    except Exception as exc:
        raise EvaluatorIntegrityError('Model identity unavailable during evaluation call') from exc
    record['model_digest_' + phase] = actual
    if actual != expected:
        raise EvaluatorIntegrityError('Model digest changed during evaluation call')


def evidence_json(value: Any) -> Any:
    """Freeze SDK objects as JSON evidence without retaining live agent state."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): evidence_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unserializable_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def require_evaluator_checkout(state: dict[str, Any], expected_revision: str | None) -> None:
    if expected_revision is None:
        original.require_baseline_checkout(state, allow_non_main=False, allow_dirty=False, skip_upstream_check=False)
        return
    if len(expected_revision) != 40 or any(c not in "0123456789abcdef" for c in expected_revision):
        raise RuntimeError("Expected evaluator revision must be a full commit SHA")
    if not state["clean"] or state["head"] != expected_revision:
        raise RuntimeError("Evaluator requires a clean checkout at the exact approved revision")
ROTATIONS = (
    ARMS,
    (ARMS[2], ARMS[0], ARMS[1]),
    (ARMS[1], ARMS[2], ARMS[0]),
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_tasks() -> list[dict[str, Any]]:
    tasks = original.load_eval_tasks() + original.load_eval_tasks(EXTRA_ROOT)
    ids = [item["id"] for item in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate IDs across historical and new evaluation sets")
    for task in tasks:
        if task.get("split") != "held_out_eval" or not task.get("expected_behavior"):
            raise RuntimeError("Every comparison item requires a held-out rubric")
        original._task_prompt(task)  # reject tool transcripts and malformed prompts
    return sorted(tasks, key=lambda item: item["id"])


def plan(tasks: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    # Interleave conditions and rotate order so warmup/time effects do not always
    # favour the same model. Every task/repeat/arm is independently reconstructed.
    return [(task["id"], repetition, arm)
            for task in sorted(tasks, key=lambda t: digest(t["id"]))
            for repetition in range(REPEATS)
            for arm in ROTATIONS[repetition]]


def _response_text(result: Any) -> str:
    message = result.get("message") if isinstance(result, dict) else getattr(result, "message", None)
    if message is None:
        raise RuntimeError("Ollama returned no message")
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str):
        raise RuntimeError("Ollama returned no text content")
    return content


def _direct(task: dict[str, Any], config: Any, *, generation_policy: dict | None = None,
            expected_model_digest: str | None = None) -> str:
    from ollama import chat
    prompt, systems = original._task_prompt(task)
    messages = [
        {"role": "system", "content": "Answer the user's question. No tools, live repository, machine state, web or personal memory are available. Do not claim to have inspected them."},
        *systems,
        {"role": "user", "content": prompt},
    ]
    evidence = CELL_EVIDENCE.get()
    options = {"temperature": config.model.temperature, "num_ctx": config.model.context_tokens}
    if generation_policy:
        options['num_predict'] = generation_policy['per_turn_tokens']
    attempts = []
    if evidence is not None:
        evidence['request_messages'] = evidence_json(messages)
        evidence['direct_attempts'] = attempts
    def generate(context, think, settings, phase):
        attempt = {'phase':phase, 'messages':evidence_json(context), 'think':think, 'options':dict(settings)}
        attempts.append(attempt)
        _check_call_identity(config.model.name, expected_model_digest, attempt, 'before')
        result = chat(model=config.model.name, messages=context, think=think, options=settings)
        attempt['raw_response'] = evidence_json(result)
        if evidence is not None and phase == 'initial':
            evidence['raw_response'] = attempt['raw_response']
        _check_call_identity(config.model.name, expected_model_digest, attempt, 'after')
        return result
    response = generate(messages, config.model.think, options, 'initial')
    content = _response_text(response)
    raw = evidence_json(response)
    if generation_policy and (not content.strip() or raw.get('done_reason') == 'length'):
        # Preserve the original blank/length failure. Delivery recovery is a
        # separate, bounded attempt, not a replacement of its raw evidence.
        if evidence is not None:
            evidence['initial_delivery_failure'] = {'empty':not content.strip(), 'done_reason':raw.get('done_reason')}
        headroom = config.model.context_tokens - int(raw.get('prompt_eval_count') or 0) - int(raw.get('eval_count') or 0) - generation_policy['context_margin_tokens']
        budget = min(generation_policy['direct_completion_tokens'], headroom)
        if budget >= 256:
            recovery = [*messages, {'role':'assistant', 'content':content,
                                   'thinking':str(raw.get('message', {}).get('thinking') or '')},
                        {'role':'user','content':
                         "The preceding attempt produced a blank or generation-limited final answer. "
                         "Give one concise complete final answer to my original request from the supplied premises. "
                         "No tools or new observations are available. Preserve uncertainty, do not invent inspection "
                         "or execution, and do not restart your reasoning. Original request:\n" + prompt}]
            completed = generate(recovery, generation_policy['direct_completion_think'],
                                 {**options,'num_predict':budget}, 'bounded_completion')
            content = _response_text(completed)
        elif evidence is not None:
            evidence['completion_skipped'] = 'insufficient_context_headroom'
    return content


def _scaffold(task: dict[str, Any], config: Any, snapshot: Path, namespace: str, *,
              generation_policy: dict | None = None, expected_model_digest: str | None = None) -> str:
    cfg = copy.deepcopy(config)
    # Each of the 450+ turns starts with genuinely empty memory and chat context.
    cfg.agent.data_dir = str(snapshot / "localpilot-data" / "paired-eval" / namespace)
    agent = original.LocalPilotAgent(cfg, snapshot)
    original.restrict_eval_tools(agent)
    agent.messages.append({"role": "system", "content": original.EVAL_ISOLATION_MESSAGE})
    prompt, systems = original._task_prompt(task)
    agent.messages.extend(systems)
    evidence = CELL_EVIDENCE.get()
    stream = agent._stream_chat_message
    if evidence is not None:
        evidence["model_turns"] = []
        def capture_stream(*args: Any, **kwargs: Any) -> Any:
            from ollama._utils import convert_function_to_tool
            if generation_policy:
                requested = dict(kwargs.get('options') or {})
                requested['num_predict'] = min(int(requested.get('num_predict') or generation_policy['per_turn_tokens']), generation_policy['per_turn_tokens'])
                kwargs['options'] = requested
            settings = {k: copy.deepcopy(v) for k, v in kwargs.items() if k not in {"messages", "tools", "chat"}}
            if kwargs.get("tools") is not None:
                settings["tools"] = [
                    convert_function_to_tool(tool).model_dump(mode="json") if callable(tool)
                    else tool.model_dump(mode="json") if hasattr(tool, "model_dump")
                    else copy.deepcopy(tool)
                    for tool in kwargs["tools"]
                ]
            turn = {"messages": evidence_json(kwargs.get("messages") if kwargs.get("messages") is not None else agent.messages),
                    "settings": settings}
            evidence["model_turns"].append(turn)
            try:
                _check_call_identity(cfg.model.name, expected_model_digest, turn, 'before')
                result = stream(*args, **kwargs)
                turn["response"] = evidence_json(result)
                _check_call_identity(cfg.model.name, expected_model_digest, turn, 'after')
                return result
            except Exception as exc:
                turn["error"] = {"type": type(exc).__name__, "message": str(exc)}
                if hasattr(exc, "partial_message"):
                    turn["partial_message"] = evidence_json(exc.partial_message)
                raise
        agent._stream_chat_message = capture_stream
    try:
        return agent.ask(prompt, interface="direct")
    finally:
        # Avoid a cycle from the capturing closure back to its bound agent.
        agent._stream_chat_message = stream
        if evidence is not None:
            evidence["messages"] = evidence_json(agent.messages)
            audit_path = agent.data_dir / "audit.jsonl"
            evidence["audit_jsonl"] = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""


def _run_cell(task: dict[str, Any], repetition: int, arm: str, base: Any, candidate: Any, snapshot: Path, *,
              generation_policy: dict | None = None, expected_model_digest: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    evidence: dict[str, Any] = {}
    token = CELL_EVIDENCE.set(evidence)
    try:
        if arm == "base_direct":
            response = (_direct(task, base) if generation_policy is None and expected_model_digest is None else
                        _direct(task, base, generation_policy=generation_policy, expected_model_digest=expected_model_digest))
        else:
            config = base if arm == "base_localpilot" else candidate
            namespace = f"{task['id']}-{repetition}-{arm}"
            response = (_scaffold(task, config, snapshot, namespace) if generation_policy is None and expected_model_digest is None else
                        _scaffold(task, config, snapshot, namespace,
                                  generation_policy=generation_policy, expected_model_digest=expected_model_digest))
        if any(turn.get('error', {}).get('type') == 'EvaluatorIntegrityError' for turn in evidence.get('model_turns', [])):
            raise EvaluatorIntegrityError('A scaffold model call failed its frozen identity check')
        if not isinstance(response, str):
            raise RuntimeError("Model response is not text")
        error = None
    except Exception as exc:
        # Runtime exceptions are NOT automatically recorded as model hard failures.
        response, error = "", {"type": type(exc).__name__, "message": str(exc)[:1200]}
    finally:
        CELL_EVIDENCE.reset(token)
    return {
        "task_id": task["id"], "task_type": task["task_type"],
        "repeat": repetition + 1, "arm": arm,
        "response": response, "error": error,
        "evidence": evidence,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def _initial_report(tasks: list[dict[str, Any]], state: dict[str, Any], base: Any, candidate: Any, *, generation_policy: dict | None = None) -> dict[str, Any]:
    base_identity = original.ollama_model_identity(base.model.name)
    candidate_identity = original.ollama_model_identity(candidate.model.name)
    if not base_identity.get("digest") or not candidate_identity.get("digest"):
        raise RuntimeError("Both Ollama model digests must be available")
    if base_identity["digest"] == candidate_identity["digest"]:
        raise RuntimeError("Base and candidate resolve to the same model digest")
    if (base.model.think, base.model.temperature, base.model.context_tokens) != (
            candidate.model.think, candidate.model.temperature, candidate.model.context_tokens):
        raise RuntimeError("Foundation comparison requires identical reasoning and sampling configuration")
    return {
        "schema_version": 1, "suite": "Nestra Paired Evaluation v2",
        "run_id": uuid.uuid4().hex, "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None, "repository": state,
        "generation_policy": copy.deepcopy(generation_policy),
        "task_ids": [task["id"] for task in tasks],
        "task_digest": digest(tasks), "planned_cells": len(tasks) * REPEATS * len(ARMS),
        "repeats": REPEATS, "arms": list(ARMS),
        "models": {"base": base_identity, "candidate": candidate_identity},
        "inference": {"think": base.model.think,
                      "temperature": base.model.temperature,
                      "context_tokens": base.model.context_tokens},
        "isolation": {
            "fresh_memory_per_cell": True, "library_disabled": True,
            "training_tree_removed": True, "web_disabled": True,
            "scaffold_tools": sorted(original.ALLOWED_EVAL_TOOLS),
            "direct_tools": [], "source_snapshot": "git archive HEAD",
        },
        "cells": [],
    }


def validate_resume(report: dict[str, Any], tasks: list[dict[str, Any]], state: dict[str, Any], base: Any, candidate: Any, *, generation_policy: dict | None = None) -> None:
    if 'integrity_error' in report or any(c.get('error', {}).get('type') == 'EvaluatorIntegrityError' for c in report.get('cells', []) if c.get('error')):
        raise RuntimeError('Integrity-failed report cannot resume; preserve it and start a fresh report')
    if report.get('generation_policy') != generation_policy:
        raise RuntimeError('Generation policy changed; cannot resume')
    if report.get("suite") != "Nestra Paired Evaluation v2" or report.get("repeats") != REPEATS:
        raise RuntimeError("Not a supported triplicate comparison report")
    if report.get("task_digest") != digest(tasks) or report.get("repository", {}).get("head") != state["head"]:
        raise RuntimeError("Cannot resume after task or repository changes")
    if (report.get("models", {}).get("base", {}).get("digest") != original.ollama_model_identity(base.model.name).get("digest")
            or report.get("models", {}).get("candidate", {}).get("digest") != original.ollama_model_identity(candidate.model.name).get("digest")):
        raise RuntimeError("Cannot resume with different model digests")
    if report.get("inference") != {
        "think": base.model.think, "temperature": base.model.temperature, "context_tokens": base.model.context_tokens
    }:
        raise RuntimeError("Cannot resume with altered inference settings")
    expected = set(plan(tasks))
    found = [(cell.get("task_id"), cell.get("repeat", 0) - 1, cell.get("arm")) for cell in report["cells"]]
    if len(found) != len(set(found)) or any(key not in expected for key in found):
        raise RuntimeError("Report contains duplicate or unexpected cells")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="gpt-oss:20b")
    parser.add_argument("--candidate-model", default="nestra:20b-p1")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="Pilot only: choose N tasks before starting a report")
    parser.add_argument("--max-new-cells", type=int, default=None, help="Bound current session; use --resume for continuation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-revision", help="Full approved evaluator SHA for a clean unmerged PR checkout")
    parser.add_argument("--plan-only", action="store_true", help="No Ollama calls, local state mutation or evaluation")
    args = parser.parse_args()
    if args.repeats != REPEATS:
        parser.error("This suite requires exactly three repeats per arm")
    tasks = load_tasks()
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [t for t in tasks if t["id"] in wanted]
        if wanted - {t["id"] for t in tasks}:
            parser.error("Unknown selected task ID")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    if not tasks:
        parser.error("No evaluation tasks selected")
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    if args.plan_only:
        print(json.dumps({"tasks": len(tasks), "repeats": REPEATS, "arms": list(ARMS),
                          "model_calls": len(plan(tasks)), "categories": sorted({t["task_type"] for t in tasks})}, indent=2))
        return 0
    state = original.repository_state(ROOT)
    require_evaluator_checkout(state, args.expected_revision)
    base = original.isolated_config(ROOT, model=args.base_model)
    candidate = original.isolated_config(ROOT, model=args.candidate_model)
    output = args.output or original.default_output_path("nestra_paired_v2")
    if args.resume:
        if not output.is_file():
            raise RuntimeError("--resume requires an existing --output report")
        report = json.loads(output.read_text(encoding="utf-8"))
        validate_resume(report, tasks, state, base, candidate, generation_policy=BOUNDED_GENERATION_POLICY)
    else:
        if output.exists():
            raise RuntimeError("Refusing to overwrite an existing benchmark report")
        report = _initial_report(tasks, state, base, candidate, generation_policy=BOUNDED_GENERATION_POLICY)
        original.write_report(output, report)
    completed = {(c["task_id"], c["repeat"] - 1, c["arm"]) for c in report["cells"]}
    remaining = [key for key in plan(tasks) if key not in completed]
    if args.max_new_cells is not None:
        remaining = remaining[:args.max_new_cells]
    task_by_id = {t["id"]: t for t in tasks}
    errors = 0
    with tempfile.TemporaryDirectory(prefix="nestra-paired-v2-") as tmp:
        snapshot = Path(tmp) / "repo"
        original.build_isolated_snapshot(ROOT, snapshot)
        for position, (task_id, repetition, arm) in enumerate(remaining, start=1):
            print(f"[{position}/{len(remaining)}] {task_id} repeat {repetition+1} {arm}", flush=True)
            expected_digest = report['models']['base' if arm in {'base_direct','base_localpilot'} else 'candidate']['digest']
            result = _run_cell(task_by_id[task_id], repetition, arm, base, candidate, snapshot,
                               generation_policy=BOUNDED_GENERATION_POLICY, expected_model_digest=expected_digest)
            # SQLite context managers commit but may leave unreachable handles
            # until collection; release them before Windows snapshot deletion.
            gc.collect()
            report["cells"].append(result)
            if result.get('error') and result['error']['type'] == 'EvaluatorIntegrityError':
                report['integrity_error'] = result['error']['message']
                report['completed_at'] = None
                original.write_report(output, report)
                raise EvaluatorIntegrityError(report['integrity_error'])
            original.write_report(output, report)
            if result["error"]:
                errors += 1
            if errors >= 3:
                print("Three execution errors; stopping to investigate rather than obscuring a broken environment.", flush=True)
                break
    if len(report["cells"]) == report["planned_cells"]:
        report["completed_at"] = datetime.now(UTC).isoformat()
    original.write_report(output, report)
    print(f"Saved {len(report['cells'])}/{report['planned_cells']} comparison cells to {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
