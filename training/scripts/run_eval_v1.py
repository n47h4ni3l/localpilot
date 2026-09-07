from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config, load_config

EVAL_ROOT = ROOT / "training" / "evals"
REPORT_ROOT = ROOT / "training" / "reports"
ALLOWED_EVAL_TOOLS = frozenset(
    {
        "list_repository_tree",
        "read_repository_file",
        "search_repository",
        "inspect_project_dependencies",
    }
)
EVAL_ISOLATION_MESSAGE = (
    "HELD-OUT EVALUATION MODE: benchmark/training files are intentionally absent from the "
    "repository snapshot and external web, GitHub, machine-state, library, memory-summary, "
    "and reversible-action tools are intentionally unavailable. Do not try to locate the "
    "benchmark. Use only the task prompt and the allowed repository evidence. The absence of "
    "the training tree in this temporary snapshot is an evaluation boundary, not evidence "
    "about the real LocalPilot repository."
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {str(stderr).strip()}")
    return completed.stdout


def repository_state(root: Path) -> dict[str, Any]:
    branch = str(_git(root, "branch", "--show-current")).strip()
    head = str(_git(root, "rev-parse", "HEAD")).strip()
    status = str(_git(root, "status", "--porcelain")).strip()
    try:
        upstream = str(_git(root, "rev-parse", "--verify", "origin/main")).strip()
    except RuntimeError:
        upstream = ""
    return {
        "branch": branch,
        "head": head,
        "origin_main": upstream or None,
        "clean": not bool(status),
        "status": status,
    }


def require_baseline_checkout(
    state: dict[str, Any],
    *,
    allow_non_main: bool,
    allow_dirty: bool,
    skip_upstream_check: bool,
) -> None:
    if not allow_non_main and state["branch"] != "main":
        raise RuntimeError(
            f"Eval v1 baseline must run from main; current branch is {state['branch']!r}."
        )
    if not allow_dirty and not state["clean"]:
        raise RuntimeError("Eval v1 baseline requires a clean working tree.")
    if not skip_upstream_check:
        upstream = state.get("origin_main")
        if not upstream:
            raise RuntimeError(
                "origin/main could not be resolved. Fetch/pull first or use --skip-upstream-check."
            )
        if state["head"] != upstream:
            raise RuntimeError(
                "HEAD does not match origin/main. Run git pull --ff-only origin main before scoring."
            )


def load_eval_tasks(eval_root: Path = EVAL_ROOT) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("*.jsonl")):
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if record.get("split") != "held_out_eval":
                continue
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RuntimeError(f"Held-out task {record.get('id')} has no messages.")
            if any(message.get("role") == "assistant" for message in messages if isinstance(message, dict)):
                raise RuntimeError(f"Held-out task {record.get('id')} contains an assistant answer.")
            users = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if len(users) != 1:
                raise RuntimeError(
                    f"Eval v1 runner currently requires exactly one user message per task: {record.get('id')}"
                )
            tasks.append(record)
    ids = [str(task.get("id") or "") for task in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Held-out evaluation task IDs are not unique.")
    return sorted(tasks, key=lambda item: str(item["id"]))


def _safe_extract_git_archive(root: Path, destination: Path) -> None:
    archive = _git(root, "archive", "--format=tar", "HEAD", binary=True)
    assert isinstance(archive, bytes)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        destination_resolved = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve(strict=False)
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in git archive: {member.name}") from exc
        handle.extractall(destination)


def build_isolated_snapshot(root: Path, destination: Path) -> None:
    """Export tracked HEAD and remove the complete training tree before model access."""
    _safe_extract_git_archive(root, destination)
    training_root = destination / "training"
    if training_root.exists():
        shutil.rmtree(training_root)


def isolated_config(root: Path, *, model: str | None = None) -> Config:
    cfg = copy.deepcopy(load_config(root / "localpilot.toml"))
    if model:
        cfg.model.name = model
    cfg.model.memory_embeddings_enabled = False
    cfg.library.enabled = False
    cfg.systemsense.enabled = False
    cfg.selfdev.enabled = False
    cfg.safety.auto_allow_read_only = True
    cfg.safety.auto_allow_reversible = False
    cfg.safety.require_confirmation_for_destructive = True
    return cfg


def restrict_eval_tools(agent: LocalPilotAgent) -> None:
    agent.tools = {
        name: spec for name, spec in agent.tools.items() if name in ALLOWED_EVAL_TOOLS
    }
    missing = ALLOWED_EVAL_TOOLS.difference(agent.tools)
    if missing:
        raise RuntimeError(f"Eval tool surface is incomplete: {sorted(missing)}")


def ollama_model_identity(model_name: str) -> dict[str, Any]:
    try:
        from ollama import list as list_models
    except ImportError as exc:
        raise RuntimeError("Ollama Python package is not installed.") from exc

    response = list_models()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    def field(model: Any, name: str) -> Any:
        if isinstance(model, dict):
            return model.get(name)
        return getattr(model, name, None)

    for item in models or []:
        name = field(item, "model") or field(item, "name")
        if str(name or "") != model_name:
            continue
        size = field(item, "size")
        return {
            "name": model_name,
            "digest": field(item, "digest"),
            "size_bytes": int(size) if isinstance(size, (int, float)) else size,
            "modified_at": str(field(item, "modified_at") or "") or None,
        }
    raise RuntimeError(f"Required Ollama model is not installed: {model_name}")


def _task_prompt(task: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    system_messages: list[dict[str, str]] = []
    user_prompt = ""
    for message in task["messages"]:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system":
            system_messages.append({"role": "system", "content": content})
        elif role == "user":
            user_prompt = content
        elif role == "tool":
            raise RuntimeError(
                f"Held-out task {task['id']} contains a tool transcript; Eval v1 runner does not replay tools."
            )
    return user_prompt, system_messages


def run_task(
    snapshot_root: Path,
    base_config: Config,
    task: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.agent.data_dir = f"localpilot-data/eval/{task['id']}"
    agent = LocalPilotAgent(cfg, snapshot_root)
    restrict_eval_tools(agent)
    agent.messages.append({"role": "system", "content": EVAL_ISOLATION_MESSAGE})
    prompt, system_messages = _task_prompt(task)
    agent.messages.extend(system_messages)
    started = time.perf_counter()
    try:
        response = agent.ask(prompt, interface="direct")
        error = None
    except Exception as exc:  # preserve a failed task as benchmark evidence
        response = ""
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    elapsed = round(time.perf_counter() - started, 3)
    return {
        "task_id": task["id"],
        "task_type": task["task_type"],
        "difficulty": task.get("difficulty"),
        "response": response,
        "error": error,
        "duration_seconds": elapsed,
    }


def default_output_path(label: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    return REPORT_ROOT / f"{safe_label}_{stamp}.json"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LocalPilot Eval v1 against an isolated tracked-source snapshot."
    )
    parser.add_argument("--label", default="current_localpilot_baseline")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-non-main", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-upstream-check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    state = repository_state(ROOT)
    require_baseline_checkout(
        state,
        allow_non_main=args.allow_non_main,
        allow_dirty=args.allow_dirty,
        skip_upstream_check=args.skip_upstream_check,
    )
    tasks = load_eval_tasks()
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted.difference(task["id"] for task in tasks)
        if missing:
            raise RuntimeError(f"Unknown Eval v1 task IDs: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise RuntimeError("--limit must be positive.")
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("No Eval v1 tasks selected.")

    base_config = isolated_config(ROOT, model=args.model)
    identity = ollama_model_identity(base_config.model.name)
    output = args.output or default_output_path(args.label)
    report: dict[str, Any] = {
        "schema_version": 1,
        "eval_name": "LocalPilot Eval v1",
        "label": args.label,
        "started_at": _utc_now(),
        "completed_at": None,
        "repository": state,
        "model": {
            **identity,
            "provider": base_config.model.provider,
            "think": base_config.model.think,
            "temperature": base_config.model.temperature,
            "context_tokens": base_config.model.context_tokens,
        },
        "isolation": {
            "source_snapshot": "git archive HEAD",
            "training_tree_removed": True,
            "durable_memory_empty_per_task": True,
            "memory_embeddings_disabled": True,
            "library_disabled": True,
            "systemsense_disabled": True,
            "external_web_disabled": True,
            "github_tools_disabled": True,
            "reversible_actions_disabled": True,
            "allowed_tools": sorted(ALLOWED_EVAL_TOOLS),
        },
        "task_count": len(tasks),
        "tasks": [],
    }
    write_report(output, report)

    with tempfile.TemporaryDirectory(prefix="localpilot-eval-v1-") as temp_dir:
        snapshot_root = Path(temp_dir) / "repo"
        build_isolated_snapshot(ROOT, snapshot_root)
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task['id']} ({task['task_type']})", flush=True)
            result = run_task(snapshot_root, base_config, task)
            report["tasks"].append(result)
            write_report(output, report)
            if args.fail_fast and result["error"] is not None:
                break

    report["completed_at"] = _utc_now()
    report["completed_task_count"] = len(report["tasks"])
    report["error_count"] = sum(item["error"] is not None for item in report["tasks"])
    write_report(output, report)
    print(f"Wrote Eval v1 responses: {output}")
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
