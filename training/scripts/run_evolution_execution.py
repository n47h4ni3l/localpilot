from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localpilot.config import load_config
from localpilot.selfdev import CandidateTools, SelfDeveloper
from localpilot.selfdev_response_parsing import apply_change_plan, parse_change_plan

BENCHMARK_ROOT = ROOT / "training" / "evolution_execution"
CASES_PATH = BENCHMARK_ROOT / "cases.json"
REPORT_ROOT = ROOT / "training" / "reports"
SCORE_SCRIPT = ROOT / "training" / "scripts" / "score_evolution_execution.py"
ACCEPTANCE_DIR_NAME = ".benchmark_acceptance"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain").strip()
    try:
        origin_main = _git(root, "rev-parse", "--verify", "origin/main").strip()
    except RuntimeError:
        origin_main = ""
    return {
        "branch": _git(root, "branch", "--show-current").strip(),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "origin_main": origin_main or None,
        "clean": not bool(status),
        "status": status,
    }


def require_benchmark_checkout(state: dict[str, Any], *, post_cc: bool = False) -> None:
    if not post_cc and state.get("branch") != "main":
        raise RuntimeError("Evolution-execution baseline must be launched from main after merge.")
    if not state.get("clean"):
        raise RuntimeError("Evolution-execution baseline requires a clean working tree.")
    if not state.get("origin_main"):
        raise RuntimeError("origin/main is unavailable; fetch and pull before running the baseline.")
    if state.get("head") != state.get("origin_main"):
        if not post_cc:
            raise RuntimeError("HEAD does not match origin/main; pull latest main before running the baseline.")
        merge_base = _git(ROOT, "merge-base", "HEAD", "origin/main").strip()
        if merge_base != state.get("origin_main"):
            raise RuntimeError("Post-CC benchmark branch must be based on the latest origin/main.")


def load_case_document(path: Path = CASES_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("benchmark_name") != "LocalPilot Evolution Execution v1":
        raise RuntimeError("Unexpected evolution-execution benchmark definition.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Evolution-execution benchmark has no cases.")
    ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id in ids:
            raise RuntimeError(f"Invalid or duplicate benchmark case id: {case_id!r}")
        ids.add(case_id)
        initial = case.get("initial_files")
        allowed = case.get("allowed_paths")
        if not isinstance(initial, dict) or not isinstance(allowed, list):
            raise RuntimeError(f"Benchmark case {case_id} has invalid fixture fields.")
        for relative in [*initial, *allowed]:
            validate_relative_path(str(relative))
        hidden = BENCHMARK_ROOT / str(case.get("hidden_fixture") or "")
        if not hidden.is_dir():
            raise RuntimeError(f"Benchmark case {case_id} has no hidden acceptance fixture.")
    return document


def validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe benchmark fixture path: {value!r}")
    return path


def create_fixture_repository(case: dict[str, Any], workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    for relative, content in case["initial_files"].items():
        target = workspace / validate_relative_path(str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    commands = (
        ["git", "init", "--quiet"],
        ["git", "config", "user.name", "LocalPilot Benchmark"],
        ["git", "config", "user.email", "benchmark@invalid.local"],
        ["git", "add", "--all"],
        ["git", "commit", "--quiet", "-m", "fixture baseline"],
    )
    for command in commands:
        completed = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Fixture setup failed: {completed.stderr.strip()}")


class RecordingCandidateTools(CandidateTools):
    """Current candidate tools plus benchmark-only scope and write-order evidence."""

    def __init__(
        self,
        workspace: Path,
        *,
        allowed_paths: Iterable[str],
        readable_paths: Iterable[str],
        max_files: int,
    ) -> None:
        self.allowed_benchmark_paths = {Path(item).as_posix() for item in allowed_paths}
        self.write_events: list[dict[str, Any]] = []
        self.current_stage = "implementation"
        super().__init__(
            workspace,
            max_files=max_files,
            readable_paths=readable_paths,
            soft_file_budget=max_files,
        )

    def validate_project_write(self, relative_path: str, content: str, **kwargs: Any) -> Path:
        normalized = Path(relative_path).as_posix()
        if normalized not in self.allowed_benchmark_paths:
            raise PermissionError(
                f"Path is outside this benchmark task's allowed change set: {normalized}"
            )
        return super().validate_project_write(relative_path, content, **kwargs)

    def create_project_directory(self, relative_path: str) -> str:
        normalized = Path(relative_path).as_posix().rstrip("/")
        permitted = any(
            item == normalized or item.startswith(normalized + "/")
            for item in self.allowed_benchmark_paths
        )
        if not permitted:
            exc = PermissionError(
                f"Directory is outside this benchmark task's allowed change set: {normalized}"
            )
            self._record_failed_write(relative_path, exc)
            raise exc
        return super().create_project_directory(relative_path)

    def write_project_file(self, relative_path: str, content: str) -> str:
        result = super().write_project_file(relative_path, content)
        self.write_events.append(
            {
                "sequence": len(self.write_events) + 1,
                "stage": self.current_stage,
                "path": Path(relative_path).as_posix(),
            }
        )
        return result


def implementation_messages(case: dict[str, Any], *, repair_feedback: str | None = None) -> list[dict[str, Any]]:
    allowed = json.dumps(case["allowed_paths"], ensure_ascii=False)
    if repair_feedback is None:
        system = (
            "You are LocalPilot's current pre-Claude-Code implementation-stage developer. Work only in the "
            "supplied disposable fixture repository through the candidate file tools. Inspect the relevant "
            "files, implement the complete contract, add or update tests, inspect the diff, and run the "
            "non-executing static checks. You cannot execute candidate code; deterministic evaluator checks "
            "run outside your tool surface. Never access the real LocalPilot checkout, use shell commands, "
            "or write outside the allowed paths. When the contract requires a regression test before a source "
            "fix, make the test-file write first. Finish with compact JSON containing summary and reusable_lesson.\n"
            f"Allowed paths: {allowed}\nTask contract: {case['instruction']}"
        )
        user = "Implement this fixture task now and make concrete candidate changes."
    else:
        system = (
            "You are LocalPilot's current pre-Claude-Code repair-stage developer. Repair the same disposable "
            "fixture candidate through the supplied tools. Use the exact evaluator failure output below, inspect "
            "the current files and diff, make the smallest complete correction, and rerun non-executing static "
            "checks. You still cannot execute candidate code. Never access the real LocalPilot checkout, use "
            "shell commands, or write outside the allowed paths. Finish with compact JSON containing summary "
            "and reusable_lesson.\n"
            f"Allowed paths: {allowed}\nTask contract: {case['instruction']}\n"
            f"Evaluator failure output:\n{repair_feedback[:12000]}"
        )
        user = "Diagnose the evaluator failure and repair this same candidate now."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _candidate_functions(tools: RecordingCandidateTools) -> list[Callable[..., Any]]:
    return [
        tools.list_project_files,
        tools.read_project_file,
        tools.create_project_directory,
        tools.write_project_file,
        tools.complexity_report,
        tools.run_candidate_static_checks,
        tools.show_candidate_diff,
    ]


def run_model_stage(
    developer: SelfDeveloper,
    chat: Callable[..., Any],
    model: str,
    case: dict[str, Any],
    tools: RecordingCandidateTools,
    *,
    stage: str,
    rounds: int,
    repair_feedback: str | None = None,
) -> str:
    tools.current_stage = stage
    writes_before = tools.write_count
    final_text = developer._tool_stage(
        chat=chat,
        model=model,
        messages=implementation_messages(case, repair_feedback=repair_feedback),
        functions=_candidate_functions(tools),
        rounds=rounds,
        force=True,
        branch=f"benchmark/{case['id']}",
        stage=f"evolution_execution_{stage}",
    )
    if tools.write_count != writes_before:
        return final_text

    response = developer._developer_chat(
        chat,
        force=True,
        branch=f"benchmark/{case['id']}",
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return one strict JSON object with summary, reusable_lesson, and a non-empty changes list. "
                    "Every change needs path, complete replacement content, and reason. Use only the allowed paths. "
                    "No markdown or hidden reasoning. The caller will validate and apply the plan through the same "
                    "confined candidate write tool.\n"
                    f"Allowed paths: {json.dumps(case['allowed_paths'])}\n"
                    f"Task contract: {case['instruction']}\n"
                    f"Prior stage output: {final_text[:4000]}"
                ),
            },
            {"role": "user", "content": "Produce the concrete candidate change plan now."},
        ],
        options={"temperature": 0.0},
    )
    plan = parse_change_plan(developer._content(response), tools.max_files)
    apply_change_plan(plan, tools)
    return json.dumps({"summary": plan.summary, "reusable_lesson": plan.reusable_lesson})


def _run_command(workspace: Path, argv: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-B", *argv],
            cwd=workspace,
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        timed_out = False
        output = (completed.stdout + "\n" + completed.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        timed_out = True
        output = "\n".join(
            part.decode("utf-8", errors="replace") if isinstance(part, bytes) else str(part or "")
            for part in (exc.stdout, exc.stderr)
        ).strip()
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "output": output.replace(str(workspace), "<fixture>")[-6000:],
    }


def run_checks(case: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    acceptance_dir = workspace / ACCEPTANCE_DIR_NAME
    if acceptance_dir.exists():
        shutil.rmtree(acceptance_dir)
    try:
        for check in case["checks"]:
            kind = str(check["kind"])
            if kind == "project_tests":
                argv = ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
            elif kind == "hidden_acceptance":
                source = BENCHMARK_ROOT / str(case["hidden_fixture"])
                shutil.copytree(source, acceptance_dir)
                argv = [
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    ACCEPTANCE_DIR_NAME,
                    "-p",
                    "test_*.py",
                ]
            else:
                raise RuntimeError(f"Unknown deterministic check kind: {kind!r}")
            outcome = _run_command(workspace, argv)
            results.append({"id": str(check["id"]), "kind": kind, **outcome})
            if kind == "hidden_acceptance" and acceptance_dir.exists():
                shutil.rmtree(acceptance_dir)
    finally:
        if acceptance_dir.exists():
            shutil.rmtree(acceptance_dir)
    return results


def checks_passed(checks: list[dict[str, Any]]) -> bool:
    return bool(checks) and all(
        int(check.get("returncode", -1)) == 0 and not check.get("timed_out")
        for check in checks
    )


def failure_feedback(checks: list[dict[str, Any]]) -> str:
    failed = [
        f"--- {item['id']} (exit {item['returncode']}) ---\n{item.get('output', '')}"
        for item in checks
        if int(item.get("returncode", -1)) != 0 or item.get("timed_out")
    ]
    return "\n\n".join(failed)[:12000]


def changed_paths(workspace: Path) -> list[str]:
    rows = _git(workspace, "status", "--porcelain", "--untracked-files=all").splitlines()
    paths: list[str] = []
    for row in rows:
        if len(row) < 4:
            continue
        value = row[3:].split(" -> ")[-1].strip().strip('"')
        normalized = Path(value).as_posix()
        if not is_evaluator_python_cache_artifact(normalized):
            paths.append(normalized)
    return sorted(set(paths))


def is_evaluator_python_cache_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return "__pycache__" in parts and normalized.lower().endswith((".pyc", ".pyo"))


def out_of_scope_attempts(tools: RecordingCandidateTools) -> list[str]:
    marker = "outside this benchmark task's allowed change set"
    return [item for item in tools.failed_write_attempts if marker in item]


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
        return model.get(name) if isinstance(model, dict) else getattr(model, name, None)

    for item in models or []:
        name = field(item, "model") or field(item, "name")
        if str(name or "") == model_name:
            return {
                "name": model_name,
                "digest": field(item, "digest"),
                "size_bytes": field(item, "size"),
                "provider": "ollama",
            }
    raise RuntimeError(f"Selected benchmark model is not installed: {model_name}")


def run_case(
    case: dict[str, Any],
    workspace: Path,
    developer: SelfDeveloper,
    chat: Callable[..., Any],
    model: str,
    *,
    implementation_rounds: int,
    repair_rounds: int,
) -> dict[str, Any]:
    create_fixture_repository(case, workspace)
    baseline_checks = run_checks(case, workspace)
    readable = set(case["initial_files"]).union(case["allowed_paths"])
    tools = RecordingCandidateTools(
        workspace,
        allowed_paths=case["allowed_paths"],
        readable_paths=readable,
        max_files=len(case["allowed_paths"]),
    )
    repair_used = False
    runtime_error: dict[str, str] | None = None
    try:
        run_model_stage(
            developer,
            chat,
            model,
            case,
            tools,
            stage="implementation",
            rounds=implementation_rounds,
        )
        initial_checks = run_checks(case, workspace)
        if not checks_passed(initial_checks):
            repair_used = True
            run_model_stage(
                developer,
                chat,
                model,
                case,
                tools,
                stage="repair",
                rounds=repair_rounds,
                repair_feedback=failure_feedback(initial_checks),
            )
        final_checks = run_checks(case, workspace)
    except Exception as exc:
        runtime_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        initial_checks = []
        final_checks = run_checks(case, workspace)
    return {
        "task_id": str(case["id"]),
        "baseline_checks": baseline_checks,
        "initial_checks": initial_checks,
        "final_checks": final_checks,
        "repair_used": repair_used,
        "changed_paths": changed_paths(workspace),
        "write_events": tools.write_events,
        "out_of_scope_attempts": out_of_scope_attempts(tools),
        "rejected_write_attempt_count": len(tools.failed_write_attempts),
        "runtime_error": runtime_error,
    }


def run_post_cc_case(
    case: dict[str, Any],
    workspace: Path,
    developer: SelfDeveloper,
    chat: Callable[..., Any],
    model: str,
) -> dict[str, Any]:
    create_fixture_repository(case, workspace)
    baseline_checks = run_checks(case, workspace)
    readable = set(case["initial_files"]).union(case["allowed_paths"])
    tools = RecordingCandidateTools(
        workspace,
        allowed_paths=case["allowed_paths"],
        readable_paths=readable,
        max_files=len(case["allowed_paths"]),
    )
    initial_mtimes = {
        path: (workspace / path).stat().st_mtime_ns
        for path in case["allowed_paths"]
        if (workspace / path).is_file()
    }
    task = {
        "id": f"post-cc-{case['id']}",
        "title": str(case["id"]),
        "acceptance": [str(case["instruction"])],
        "hypothesis": "The Claude Code implementation backend will complete this held-out fixture contract.",
        "evaluation": {
            "metric": "Evolution Execution v1 deterministic criteria",
            "baseline": "pre-CC execution mean 3.0/4 with two hard failures",
            "success_criterion": "task passes all four deterministic criteria",
            "measurement_method": "hidden fixture checks owned by the evaluator",
        },
    }
    existing = [path for path in case["allowed_paths"] if (workspace / path).is_file()]
    new_paths = [path for path in case["allowed_paths"] if not (workspace / path).exists()]
    runtime_error: dict[str, str] | None = None
    try:
        developer._run_claude_code_implementation(
            chat=chat,
            developer_model=model,
            task=task,
            branch=f"benchmark/{case['id']}",
            workspace=workspace,
            tools=tools,
            cycle_id=0,
            research="The disposable fixture files are the complete repository evidence for this task.",
            grounding_plan={"referenced_paths": existing, "new_runtime_paths": new_paths},
            grounding_evidence=existing,
            evolution_context=str(case["instruction"]),
            lessons=[],
            force=True,
        )
        initial_checks = run_checks(case, workspace)
    except Exception as exc:
        runtime_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        initial_checks = []
    final_checks = run_checks(case, workspace)
    ordered_paths = sorted(
        changed_paths(workspace),
        key=lambda path: (
            (workspace / path).stat().st_mtime_ns
            if (workspace / path).is_file()
            else initial_mtimes.get(path, 0)
        ),
    )
    evidence = developer.audit.latest("selfdev_implementation_backend") or {}
    return {
        "task_id": str(case["id"]),
        "baseline_checks": baseline_checks,
        "initial_checks": initial_checks,
        "final_checks": final_checks,
        "repair_used": int(evidence.get("review_repair_pass_count") or 0) > 0,
        "changed_paths": changed_paths(workspace),
        "write_events": [
            {"sequence": index, "stage": "claude_code", "path": path}
            for index, path in enumerate(ordered_paths, start=1)
        ],
        "out_of_scope_attempts": [],
        "rejected_write_attempt_count": len(tools.failed_write_attempts),
        "runtime_error": runtime_error,
        "implementation_evidence": {
            key: evidence.get(key)
            for key in (
                "backend", "model", "diff_digest", "tests", "review_repair_pass_count",
                "review_status", "final_status", "exit_code", "usage", "duration_seconds",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current LocalPilot self-development implementation path against "
            "disposable held-out fixture repositories."
        )
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--model", help="Override current resource-aware developer-model selection.")
    parser.add_argument("--implementation-rounds", type=int, default=8)
    parser.add_argument("--repair-rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument(
        "--post-cc",
        action="store_true",
        help="Exercise the Claude Code backend on the same four disposable tasks.",
    )
    parser.add_argument("--claude-path", help="Absolute Claude Code executable path for post-CC mode.")
    args = parser.parse_args()

    state_before = repository_state()
    require_benchmark_checkout(state_before, post_cc=args.post_cc)
    document = load_case_document()
    selected_ids = set(args.task_id)
    cases = [
        case for case in document["cases"]
        if not selected_ids or str(case["id"]) in selected_ids
    ]
    unknown = selected_ids.difference(str(case["id"]) for case in cases)
    if unknown:
        raise RuntimeError(f"Unknown benchmark task IDs: {sorted(unknown)}")

    config = copy.deepcopy(load_config(ROOT / "localpilot.toml"))
    config.github.auto_push_candidates = False
    config.selfdev.auto_promote = False
    if args.post_cc:
        config.selfdev.implementation_backend = "claude_code"
        if args.claude_path:
            config.selfdev.implementation_executable = args.claude_path
    started_at = _utc_now()
    label = f"evolution_execution_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    task_results: list[dict[str, Any]] = []

    try:
        from ollama import chat
    except ImportError as exc:
        raise RuntimeError("Ollama Python package is required for this benchmark.") from exc

    with tempfile.TemporaryDirectory(
        prefix="localpilot-evoexec-", ignore_cleanup_errors=True
    ) as temp_dir:
        temp_root = Path(temp_dir)
        controller_root = temp_root / "controller"
        controller_root.mkdir()
        config.agent.data_dir = "benchmark-data"
        developer = SelfDeveloper(config, controller_root)
        if args.model:
            selected_model = args.model
            selection_reason = "explicit command-line override"
        else:
            selection = developer._select_developer_model()
            if not selection.model:
                raise RuntimeError(f"No current-path developer model is available: {selection.reason}")
            selected_model = selection.model
            selection_reason = selection.reason
        model_identity = ollama_model_identity(selected_model)

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] Running {case['id']} in a disposable fixture")
            task_results.append(
                run_post_cc_case(
                    case,
                    temp_root / f"fixture-{index}",
                    developer,
                    chat,
                    selected_model,
                )
                if args.post_cc
                else run_case(
                    case,
                    temp_root / f"fixture-{index}",
                    developer,
                    chat,
                    selected_model,
                    implementation_rounds=max(1, args.implementation_rounds),
                    repair_rounds=max(1, args.repair_rounds),
                )
            )
        del developer
        gc.collect()

    state_after = repository_state()
    if state_after != state_before:
        raise RuntimeError(
            "Real repository state changed during the disposable benchmark; refusing to finalize the report."
        )

    report = {
        "schema_version": 1,
        "benchmark_name": "LocalPilot Evolution Execution v1",
        "label": label,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "repository": state_before,
        "model": {**model_identity, "selection_reason": selection_reason},
        "implementation_backend": "claude_code" if args.post_cc else "local_tools",
        "isolation": {
            "disposable_fixture_repository_per_task": True,
            "real_repository_writes": False,
            "real_repository_state_verified_unchanged": True,
            "hidden_acceptance_removed_during_model_stages": True,
            "hidden_acceptance_paths_excluded_from_model_tools": True,
            "target_patches_present": False,
            "candidate_execution_available_to_model": bool(args.post_cc),
            "deterministic_execution_owned_by_evaluator": True,
            "localpilot_independent_review": bool(args.post_cc),
            "repair_pass_limit": 1,
        },
        "task_count": len(task_results),
        "tasks": task_results,
    }
    output = args.output or REPORT_ROOT / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote evolution-execution run report: {output}")

    if not args.no_score:
        scored = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT), str(output)],
            cwd=ROOT,
            check=False,
        )
        if scored.returncode != 0:
            raise RuntimeError("Evolution-execution scorer rejected the completed report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
