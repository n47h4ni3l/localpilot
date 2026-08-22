from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from localpilot.audit import AuditLog
from localpilot.config import Config
from localpilot.github_integration import GitHubIntegration
from localpilot.learning import LearningMemory
from localpilot.resource import ResourceGovernor

_IGNORE_NAMES = {".git", ".github", ".venv", "__pycache__", ".pytest_cache", "localpilot-data"}
_ALLOWED_SUFFIXES = {".py", ".toml", ".md", ".txt", ".json", ".yml", ".yaml", ".ps1", ".gitignore"}


@dataclass(slots=True)
class EvolutionResult:
    status: str
    branch: str | None
    workspace: Path | None
    summary: str
    tests_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class PlannedChange:
    path: str
    content: str
    reason: str


@dataclass(frozen=True, slots=True)
class ChangePlan:
    summary: str
    reusable_lesson: str
    changes: tuple[PlannedChange, ...]


@dataclass(frozen=True, slots=True)
class StaticRepairResult:
    check_result: str
    passed: bool
    final_text: str
    attempts_used: int


class CyclePaused(RuntimeError):
    pass


def classify_candidate_result(files_written: int, checks_passed: bool | None) -> str:
    if files_written == 0:
        return "no_changes"
    return "candidate_ready" if checks_passed else "candidate_needs_work"


def choose_next_task(
    tasks: Iterable[dict[str, Any]],
    completed_task_ids: set[str],
    pending_task_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Advance sequentially only after the current task passes CI and merges."""
    pending = pending_task_ids or set()
    for task in tasks:
        if task.get("status", "todo") != "todo":
            continue
        task_id = str(task.get("id"))
        if task_id in completed_task_ids:
            continue
        if task_id in pending:
            return None
        return task
    return None


def select_developer_model(preferred: str, everyday: str, available: Iterable[str]) -> str:
    installed = {str(name).strip() for name in available}
    return preferred if preferred in installed else everyday


def available_ollama_models() -> set[str]:
    """Read model names through the Ollama SDK without invoking a shell."""
    try:
        from ollama import list as list_models

        response = list_models()
    except Exception:
        return set()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    names: set[str] = set()
    for model in models or []:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.add(str(name))
    return names


def developer_chat(
    chat: Callable[..., Any],
    *,
    request_think: bool,
    **kwargs: Any,
) -> Any:
    """Use Ollama thinking when supported; retry without it when unsupported."""
    if request_think:
        try:
            return chat(think=True, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "does not support thinking" not in message:
                raise
    return chat(**kwargs)


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object.")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Change plan must be a JSON object.")
    return value


def parse_change_plan(text: str, max_files: int = 8) -> ChangePlan:
    value = _json_object(text)
    changes = value.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Change plan must contain at least one change.")
    if len(changes) > max_files:
        raise ValueError("Change plan exceeds the candidate file limit.")
    parsed: list[PlannedChange] = []
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError("Each planned change must be an object.")
        path, content = item.get("path"), item.get("content")
        if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
            raise ValueError("Each planned change requires string path and content fields.")
        parsed.append(PlannedChange(path.strip(), content, str(item.get("reason") or "")))
    return ChangePlan(
        str(value.get("summary") or "Structured fallback plan applied."),
        str(value.get("reusable_lesson") or "Use a structured write plan when direct tool editing stalls."),
        tuple(parsed),
    )


def apply_change_plan(plan: ChangePlan, tools: "CandidateTools") -> list[str]:
    """The sole fallback application path: every change passes CandidateTools."""
    return [tools.write_project_file(change.path, change.content) for change in plan.changes]


def build_read_context(
    tools: "CandidateTools",
    *,
    max_files: int = 8,
    max_chars_per_file: int = 16000,
) -> str:
    """Return bounded source context only for files already inspected through CandidateTools."""
    rows: list[str] = []
    paths = sorted(
        tools.files_read,
        key=lambda item: item.relative_to(tools.workspace).as_posix(),
    )

    for file_path in paths[:max_files]:
        try:
            relative = file_path.relative_to(tools.workspace).as_posix()
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )[:max_chars_per_file]
        except Exception:
            continue

        rows.append(f"--- {relative} ---\n{content}")

    return "\n\n".join(rows) or "(No candidate files were successfully inspected.)"


def build_static_repair_context(
    tools: "CandidateTools",
    check_result: str,
    *,
    max_files: int = 8,
    max_chars_per_file: int = 5000,
) -> str:
    """Build bounded failure, diff, and changed-file feedback for a repair."""
    rows: list[str] = []
    paths = sorted(
        tools.files_written,
        key=lambda item: item.relative_to(tools.workspace).as_posix(),
    )
    for file_path in paths[:max_files]:
        relative = file_path.relative_to(tools.workspace).as_posix()
        content = tools.read_project_file(relative, max_chars=max_chars_per_file)
        rows.append(f"--- {relative} ---\n{content}")

    changed_files = "\n\n".join(rows) or "(No changed file content is available.)"
    candidate_diff = tools.show_candidate_diff()[-16000:]
    return (
        f"Static-check failure:\n{check_result[:8000]}\n\n"
        f"Candidate diff:\n{candidate_diff}\n\n"
        f"Changed candidate files:\n{changed_files}"
    )


class CandidateTools:
    """File tools confined to one candidate workspace."""

    def __init__(
        self,
        workspace: Path,
        max_files: int = 8,
        protected_paths: Iterable[str] | None = None,
        existing_changed_paths: Iterable[str] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.max_files = max_files
        self.protected_paths = {
            Path(item).as_posix()
            for item in (protected_paths or ())
        }
        existing = {
            self._resolve(item)
            for item in (existing_changed_paths or ())
        }
        if len(existing) > self.max_files:
            raise RuntimeError(
                "Recovered candidate exceeds the file-write limit."
            )
        for path in existing:
            relative = path.relative_to(self.workspace).as_posix()
            if relative in self.protected_paths:
                raise PermissionError(
                    "Reviewer-controlled regression contract has local changes: "
                    f"{relative}"
                )
            suffix = path.suffix.lower() if path.name != ".gitignore" else ".gitignore"
            if suffix not in _ALLOWED_SUFFIXES:
                raise ValueError(
                    "Recovered candidate contains a disallowed file type: "
                    f"{suffix or '(none)'}"
                )
            if not path.is_file():
                raise ValueError(
                    "Recovered candidate contains a deletion or missing file: "
                    f"{relative}"
                )
            if path.stat().st_size > 1_000_000:
                raise ValueError(
                    f"Recovered candidate file exceeds 1 MB: {relative}"
                )
        self.files_written = existing
        self.files_read: set[Path] = set()
        self.write_count = 0

    def _resolve(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute():
            raise ValueError("Absolute paths are not allowed in candidate tools.")
        path = (self.workspace / raw).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("Path escapes candidate workspace.")
        if any(part in _IGNORE_NAMES for part in path.relative_to(self.workspace).parts):
            raise ValueError("Protected candidate path.")
        return path

    def list_project_files(self, max_results: int = 160) -> str:
        max_results = max(1, min(int(max_results), 300))
        rows: list[str] = []
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace)
            if any(part in _IGNORE_NAMES for part in rel.parts):
                continue
            rows.append(rel.as_posix())
            if len(rows) >= max_results:
                break
        return "\n".join(sorted(rows))

    def read_project_file(self, relative_path: str, max_chars: int = 24000) -> str:
        path = self._resolve(relative_path)
        max_chars = max(1000, min(int(max_chars), 100000))
        if not path.is_file():
            return f"File not found: {relative_path}"
        self.files_read.add(path)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def write_project_file(self, relative_path: str, content: str) -> str:
        path = self._resolve(relative_path)
        relative = path.relative_to(self.workspace).as_posix()

        if relative in self.protected_paths:
            raise PermissionError(
                "Reviewer-controlled regression contract is read-only during "
                f"autonomous repair: {relative}"
            )

        suffix = path.suffix.lower() if path.name != ".gitignore" else ".gitignore"
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"File type is not allowed for autonomous editing: {suffix or '(none)'}")
        if len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("Candidate file exceeds 1 MB safety limit.")
        if path not in self.files_written and len(self.files_written) >= self.max_files:
            raise RuntimeError("Candidate file-write limit reached for this cycle.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files_written.add(path)
        self.write_count += 1
        return f"Wrote {path.relative_to(self.workspace).as_posix()} ({len(content)} chars)."

    def run_candidate_static_checks(self) -> str:
        import tomllib

        errors: list[str] = []
        python_count = 0
        toml_count = 0
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace)
            if any(part in _IGNORE_NAMES for part in rel.parts):
                continue
            try:
                if path.suffix.lower() == ".py":
                    compile(path.read_text(encoding="utf-8", errors="strict"), str(rel), "exec")
                    python_count += 1
                elif path.suffix.lower() == ".toml":
                    with path.open("rb") as handle:
                        tomllib.load(handle)
                    toml_count += 1
            except Exception as exc:
                errors.append(f"{rel.as_posix()}: {type(exc).__name__}: {exc}")
        if errors:
            return "static_checks=failed\n" + "\n".join(errors[:50])
        return f"static_checks=passed\npython_files={python_count}\ntoml_files={toml_count}"

    def show_candidate_diff(self) -> str:
        completed = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            return completed.stderr.strip() or "git diff unavailable"
        return completed.stdout[-30000:] or "(no diff)"


class SelfDeveloper:
    """Researches and edits isolated candidates; it never promotes them."""

    def __init__(
        self,
        config: Config,
        project_root: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        if config.selfdev.auto_promote:
            raise ValueError("Automatic promotion is forbidden.")
        self.root = Path(project_root).resolve()
        self.data_dir = (self.root / config.agent.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.data_dir / "audit.jsonl")
        self.memory = LearningMemory(self.data_dir / config.selfdev.learning_database)
        self.governor = ResourceGovernor(config.resource)
        self.github = GitHubIntegration(self.root, config.github)
        self.progress = progress or (lambda _message: None)

    def _emit(self, message: str) -> None:
        self.progress(message)
        self.audit.write("selfdev_progress", message=message)

    def _reconcile_candidates(self) -> None:
        for candidate in self.memory.pending_candidates():
            lifecycle = self.github.candidate_lifecycle(candidate.branch)
            self.memory.update_candidate_review(
                candidate.cycle_id,
                validation_state=lifecycle.validation_state,
                merged=lifecycle.merged,
                pull_request_url=lifecycle.pull_request_url,
            )

    def _backlog_tasks(self) -> list[dict[str, Any]]:
        path = self.root / "selfdev-backlog.json"
        if not path.exists():
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("tasks", []))

    def _load_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        for task in self._backlog_tasks():
            if str(task.get("id")) == str(task_id):
                return task
        return None

    def _load_next_task(self) -> dict[str, Any] | None:
        return choose_next_task(
            self._backlog_tasks(),
            self.memory.completed_task_ids(),
            self.memory.pending_task_ids(),
        )

    def _candidate_workspace(self, branch: str) -> tuple[Path, bool]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = self.data_dir / "candidates" / f"{stamp}-{branch.split('/')[-1]}"
        if self.github.is_git_repo() and self.github.clean_worktree():
            result = self.github.create_candidate_worktree(branch, destination)
            if result.ok:
                return destination, True
            raise RuntimeError(result.stderr or result.stdout or "Could not create Git worktree.")

        def ignore(_directory: str, names: list[str]) -> set[str]:
            return {name for name in names if name in _IGNORE_NAMES or name.endswith(".pyc")}

        shutil.copytree(self.root, destination, ignore=ignore)
        return destination, False

    def _check_resources(self, force: bool, branch: str) -> None:
        current = self.governor.sample(interval=0.05)
        if not force and not current.background_allowed:
            self.governor.apply_process_priority(idle=False)
            self.audit.write("selfdev_paused", branch=branch, reason=current.reason)
            raise CyclePaused(current.reason)

    @staticmethod
    def _content(response: Any) -> str:
        message = getattr(response, "message", response)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    @staticmethod
    def _calls(response: Any) -> list[Any]:
        message = getattr(response, "message", response)
        if isinstance(message, dict):
            return list(message.get("tool_calls") or [])
        return list(getattr(message, "tool_calls", None) or [])

    @staticmethod
    def _call_parts(call: Any) -> tuple[str, dict[str, Any]]:
        function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
        if isinstance(function, dict):
            return str(function.get("name") or ""), dict(function.get("arguments") or {})
        return str(getattr(function, "name", "")), dict(getattr(function, "arguments", None) or {})

    def _tool_stage(
        self,
        *,
        chat: Callable[..., Any],
        model: str,
        messages: list[dict[str, Any]],
        functions: list[Callable[..., Any]],
        rounds: int,
        force: bool,
        branch: str,
        stage: str,
    ) -> str:
        by_name = {fn.__name__: fn for fn in functions}
        for round_no in range(rounds):
            self._check_resources(force, branch)
            self._emit(f"{stage} round {round_no + 1}/{rounds}")
            response = developer_chat(
                chat,
                request_think=self.config.model.think,
                model=model,
                messages=messages,
                tools=functions,
                options={"temperature": self.config.model.temperature},
            )
            message = getattr(response, "message", response)
            messages.append(message)
            calls = self._calls(response)
            if not calls:
                return self._content(response)
            for call in calls:
                name, args = self._call_parts(call)
                fn = by_name.get(name)
                self.audit.write(
                    "selfdev_tool",
                    branch=branch,
                    stage=stage,
                    tool=name,
                    args={key: ("<content>" if key == "content" else value) for key, value in args.items()},
                )
                if fn is None:
                    result = f"Unknown candidate tool: {name}"
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = f"Tool error: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
        return f"{stage} stopped at its tool-call limit."

    @staticmethod
    def _outcome(text: str, default_lesson: str) -> tuple[str, str]:
        try:
            value = _json_object(text)
        except (ValueError, json.JSONDecodeError):
            return (text.strip() or "Candidate cycle completed.")[:4000], default_lesson
        return (
            str(value.get("summary") or "Candidate cycle completed.")[:4000],
            str(value.get("reusable_lesson") or default_lesson)[:2000],
        )

    def _repair_static_failures(
        self,
        *,
        chat: Callable[..., Any],
        model: str,
        tools: CandidateTools,
        task: dict[str, Any],
        branch: str,
        cycle_id: int,
        check_result: str,
        attempts_used: int,
        protected_paths: Iterable[str] = (),
        force: bool,
    ) -> StaticRepairResult:
        """Feed static failures back into the same candidate, within limits."""
        limit = max(0, int(self.config.selfdev.max_local_repair_attempts))
        rounds = max(1, int(self.config.selfdev.local_repair_tool_rounds))
        final_text = ""
        protected_note = json.dumps(sorted(protected_paths), ensure_ascii=False)
        acceptance = json.dumps(task.get("acceptance", []), ensure_ascii=False)

        while (
            not check_result.startswith("static_checks=passed")
            and attempts_used < limit
        ):
            attempt = self.memory.record_local_repair_attempt(
                cycle_id,
                check_result=check_result,
            )
            attempts_used = max(attempts_used + 1, attempt)
            self._check_resources(force, branch)
            self._emit(
                f"Repairing local static failure for {branch} "
                f"(attempt {attempts_used}/{limit})"
            )
            context = build_static_repair_context(tools, check_result)
            writes_before = tools.write_count
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are LocalPilot's pre-push static-repair developer. "
                        "Repair the existing isolated candidate in place using only "
                        "the supplied candidate tools. The non-executing static "
                        "checks failed; use the exact failure, diff, and changed-file "
                        "context below. Inspect relevant files before editing, make "
                        "the smallest concrete correction, and rerun static checks. "
                        "Never edit stable, execute candidate code locally, weaken "
                        "safety, bypass the resource governor, promote, or use shell "
                        "command strings. Reviewer-controlled tests are immutable. "
                        "Finish with JSON containing only summary and reusable_lesson; "
                        "do not expose hidden reasoning.\n"
                        f"Reviewer-protected paths: {protected_note}\n"
                        f"Task: {task['title']}\nAcceptance: {acceptance}\n"
                        f"{context}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Repair this same candidate's static-check failure now.",
                },
            ]
            final_text = self._tool_stage(
                chat=chat,
                model=model,
                messages=messages,
                functions=[
                    tools.list_project_files,
                    tools.read_project_file,
                    tools.write_project_file,
                    tools.run_candidate_static_checks,
                    tools.show_candidate_diff,
                ],
                rounds=rounds,
                force=force,
                branch=branch,
                stage="local_static_repair",
            )

            if tools.write_count == writes_before:
                self._check_resources(force, branch)
                try:
                    response = developer_chat(
                        chat,
                        request_think=self.config.model.think,
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return one strict JSON object with summary, "
                                    "reusable_lesson, and a non-empty changes list. "
                                    "Each change requires path, complete replacement "
                                    "content, and reason. No markdown or hidden reasoning. "
                                    "The caller applies every file only through the "
                                    "confined candidate write tool. Do not include a "
                                    "reviewer-protected path.\n"
                                    f"Reviewer-protected paths: {protected_note}\n"
                                    f"Task: {task['title']}\n{context}\n"
                                    f"Prior repair response: {final_text[:4000]}"
                                ),
                            },
                            {
                                "role": "user",
                                "content": "Produce the concrete static-repair plan now.",
                            },
                        ],
                        options={"temperature": 0.0},
                    )
                    plan = parse_change_plan(self._content(response), tools.max_files)
                    apply_change_plan(plan, tools)
                    final_text = json.dumps(
                        {
                            "summary": plan.summary,
                            "reusable_lesson": plan.reusable_lesson,
                        }
                    )
                except Exception as exc:
                    final_text = (
                        f"{final_text}\nStructured static repair failed: "
                        f"{type(exc).__name__}: {exc}"
                    ).strip()

            check_result = tools.run_candidate_static_checks()
            passed = check_result.startswith("static_checks=passed")
            self.audit.write(
                "selfdev_local_static_repair",
                branch=branch,
                task_id=task["id"],
                attempt=attempts_used,
                checks_passed=passed,
                files_written=len(tools.files_written),
            )

        return StaticRepairResult(
            check_result,
            check_result.startswith("static_checks=passed"),
            final_text,
            attempts_used,
        )

    def _repair_local_candidate(
        self,
        *,
        force: bool,
    ) -> EvolutionResult | None:
        candidates = self.memory.local_candidates()
        if not candidates:
            return None

        candidate = candidates[0]
        branch = candidate.branch
        task = self._load_task_by_id(candidate.task_id)
        if task is None:
            summary = (
                "Cannot recover local candidate: backlog task "
                f"{candidate.task_id!r} no longer exists."
            )
            self.memory.finish_cycle(
                candidate.cycle_id,
                status="failed",
                summary=summary,
                reusable_lesson="Keep candidate tasks available until their local workspace is resolved.",
                checks_passed=False,
                pushed=False,
            )
            return EvolutionResult(
                "failed",
                branch,
                None,
                summary,
                False,
            )

        workspace = self.github.worktree_for_branch(branch)
        stored_workspace = Path(candidate.workspace) if candidate.workspace else None
        if workspace is None and stored_workspace is not None and stored_workspace.exists():
            summary = (
                "Stored candidate workspace exists but is no longer registered as "
                f"the worktree for {branch}: {stored_workspace}"
            )
            return EvolutionResult("failed", branch, stored_workspace, summary, False)

        if workspace is None:
            workspace = stored_workspace or (
                self.data_dir / "recoveries" / branch.split("/")[-1]
            )
            restored = self.github.checkout_existing_branch_worktree(branch, workspace)
            if not restored.ok:
                summary = (
                    "Could not restore unpushed candidate worktree: "
                    f"{restored.stderr or restored.stdout}"
                )
                return EvolutionResult("failed", branch, workspace, summary, False)
            self.memory.update_candidate_workspace(candidate.cycle_id, workspace)

        changed_paths = self.github.candidate_changed_paths(workspace)
        has_commit = self.github.branch_has_candidate_commit(workspace)
        if not changed_paths and not has_commit:
            summary = "The recoverable candidate has no local changes or candidate commit."
            self.memory.finish_cycle(
                candidate.cycle_id,
                status="failed",
                summary=summary,
                reusable_lesson="Only retain a local candidate when it has a recoverable diff.",
                checks_passed=None,
                pushed=False,
            )
            return EvolutionResult("failed", branch, workspace, summary)

        try:
            protected_paths = self.github.reviewer_modified_test_paths(
                workspace,
                refresh=False,
            )
        except RuntimeError as exc:
            summary = f"Local repair stopped because reviewer protection could not be established: {exc}"
            return EvolutionResult("failed", branch, workspace, summary, False)

        try:
            tools = CandidateTools(
                workspace,
                self.config.selfdev.max_files_per_cycle,
                protected_paths=protected_paths,
                existing_changed_paths=changed_paths,
            )
        except (PermissionError, RuntimeError, ValueError) as exc:
            summary = f"Recovered candidate failed safety validation: {exc}"
            self.memory.finish_cycle(
                candidate.cycle_id,
                status="candidate_needs_work",
                summary=summary,
                reusable_lesson="Revalidate every stale candidate against current file protections.",
                checks_passed=False,
                pushed=False,
            )
            return EvolutionResult(
                "candidate_needs_work",
                branch,
                workspace,
                summary,
                False,
            )
        check_result = tools.run_candidate_static_checks()
        repair_text = ""

        if not check_result.startswith("static_checks=passed"):
            if candidate.local_repair_attempts >= max(
                0,
                int(self.config.selfdev.max_local_repair_attempts),
            ):
                summary = (
                    f"Local static repair attempt limit reached for {branch}.\n\n"
                    f"{check_result}"
                )
                self.memory.finish_cycle(
                    candidate.cycle_id,
                    status="candidate_needs_work",
                    summary=summary,
                    reusable_lesson="Bound autonomous repair attempts and preserve the candidate for human review.",
                    checks_passed=False,
                    pushed=False,
                )
                return EvolutionResult(
                    "candidate_needs_work",
                    branch,
                    workspace,
                    summary,
                    False,
                )

            try:
                from ollama import chat
            except ImportError:
                return EvolutionResult(
                    "failed",
                    branch,
                    workspace,
                    "Ollama Python package is not installed.",
                    False,
                )
            model = select_developer_model(
                self.config.selfdev.developer_model,
                self.config.model.name,
                available_ollama_models(),
            )
            try:
                repaired = self._repair_static_failures(
                    chat=chat,
                    model=model,
                    tools=tools,
                    task=task,
                    branch=branch,
                    cycle_id=candidate.cycle_id,
                    check_result=check_result,
                    attempts_used=candidate.local_repair_attempts,
                    protected_paths=protected_paths,
                    force=force,
                )
            except CyclePaused as exc:
                summary = f"Local candidate repair paused because the PC became busy: {exc}"
                self.memory.finish_cycle(
                    candidate.cycle_id,
                    status="paused",
                    summary=summary,
                    reusable_lesson="Resume the same local candidate after the resource gate clears.",
                    checks_passed=False,
                    pushed=False,
                )
                return EvolutionResult("paused", branch, workspace, summary, False)
            check_result = repaired.check_result
            repair_text = repaired.final_text

        checks_passed = check_result.startswith("static_checks=passed")
        if not checks_passed:
            summary, lesson = self._outcome(
                repair_text,
                "Use bounded static-check feedback to repair the same local candidate.",
            )
            summary += f"\n\n{check_result}"
            self.memory.finish_cycle(
                candidate.cycle_id,
                status="candidate_needs_work",
                summary=summary,
                reusable_lesson=lesson,
                checks_passed=False,
                pushed=False,
            )
            return EvolutionResult(
                "candidate_needs_work",
                branch,
                workspace,
                summary,
                False,
            )

        changed_paths = self.github.candidate_changed_paths(workspace)
        if changed_paths:
            commit = self.github.commit_paths(
                workspace,
                f"candidate: {task['title']}",
                changed_paths,
            )
            if not commit.ok:
                summary = f"Candidate commit failed: {commit.stderr or commit.stdout}"
                self.memory.finish_cycle(
                    candidate.cycle_id,
                    status="candidate_needs_work",
                    summary=summary,
                    reusable_lesson="Keep a green local candidate recoverable when delivery fails.",
                    checks_passed=True,
                    pushed=False,
                )
                return EvolutionResult("candidate_needs_work", branch, workspace, summary, True)

        pushed = False
        delivery = "Candidate is locally green and committed; automatic push is disabled."
        status = "candidate_ready"
        if self.config.github.auto_push_candidates:
            push = self.github.push_branch(workspace, branch)
            pushed = push.ok
            if pushed:
                status = "candidate_pending_validation"
                delivery = "Candidate pushed; it remains pending until CI passes and its PR is merged."
            else:
                status = "candidate_needs_work"
                delivery = f"Candidate push failed: {push.stderr or push.stdout}"

        summary, lesson = self._outcome(
            repair_text,
            "Repair and recheck the same isolated candidate before committing or pushing.",
        )
        summary = f"{summary}\n\n{check_result}\n\n{delivery}"
        self.memory.finish_cycle(
            candidate.cycle_id,
            status=status,
            summary=summary,
            reusable_lesson=lesson,
            checks_passed=True,
            pushed=pushed,
        )
        self.audit.write(
            "selfdev_local_recovery",
            branch=branch,
            task_id=task["id"],
            checks_passed=True,
            pushed=pushed,
            status=status,
        )
        return EvolutionResult(status, branch, workspace, summary, True)

    def _repair_failed_candidate(
        self,
        *,
        force: bool,
    ) -> EvolutionResult | None:
        failed = self.memory.failed_candidates()
        if not failed:
            return None

        candidate = failed[0]
        task = self._load_task_by_id(candidate.task_id)

        if task is None:
            summary = (
                "Cannot repair failed candidate: backlog task "
                f"{candidate.task_id!r} no longer exists."
            )
            return EvolutionResult(
                "failed",
                candidate.branch,
                None,
                summary,
                False,
            )

        branch = candidate.branch
        workspace = self.github.worktree_for_branch(branch)

        if workspace is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            workspace = (
                self.data_dir
                / "repairs"
                / f"{stamp}-{branch.split('/')[-1]}"
            )

            restored = self.github.checkout_existing_branch_worktree(
                branch,
                workspace,
            )

            if not restored.ok:
                summary = (
                    "Could not restore failed candidate worktree: "
                    f"{restored.stderr or restored.stdout}"
                )
                return EvolutionResult(
                    "failed",
                    branch,
                    workspace,
                    summary,
                    False,
                )

        available = available_ollama_models()
        developer_model = select_developer_model(
            self.config.selfdev.developer_model,
            self.config.model.name,
            available,
        )

        try:
            protected_paths = self.github.reviewer_modified_test_paths(workspace)
        except RuntimeError as exc:
            summary = (
                "CI repair stopped because reviewer regression protection "
                f"could not be established safely: {exc}"
            )
            return EvolutionResult(
                "failed",
                branch,
                workspace,
                summary,
                False,
            )

        existing_changed_paths = self.github.candidate_changed_paths(workspace)
        try:
            tools = CandidateTools(
                workspace,
                self.config.selfdev.max_files_per_cycle,
                protected_paths=protected_paths,
                existing_changed_paths=existing_changed_paths,
            )
        except (PermissionError, RuntimeError, ValueError) as exc:
            summary = f"CI repair candidate failed safety validation: {exc}"
            return EvolutionResult("failed", branch, workspace, summary, False)

        failure_log = self.github.failed_workflow_log(branch)
        acceptance = json.dumps(
            task.get("acceptance", []),
            ensure_ascii=False,
        )
        lessons = self.memory.reusable_lessons(
            self.config.selfdev.lesson_limit
        )
        protected_note = json.dumps(
            sorted(protected_paths),
            ensure_ascii=False,
        )

        try:
            from ollama import chat
        except ImportError:
            return EvolutionResult(
                "failed",
                branch,
                workspace,
                "Ollama Python package is not installed.",
                False,
            )

        self._emit(
            f"CI failed for {branch}; starting autonomous repair"
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are LocalPilot's CI-repair developer. A candidate "
                    "you previously wrote failed GitHub Actions. Diagnose the "
                    "concrete failure from the supplied CI log, inspect the "
                    "candidate files, and repair only the isolated candidate "
                    "using the supplied tools. Do not touch stable, do not "
                    "weaken safety, do not execute candidate code locally, "
                    "and do not use shell command strings. Reviewer-controlled "
                    "regression tests listed below are immutable contracts: you "
                    "may inspect them, but you must not edit them, weaken their "
                    "assertions, or work around them. Repair the implementation "
                    "so it satisfies those tests. Other tests may only be changed "
                    "when they are not reviewer-protected and the failure proves "
                    "the test itself is invalid. Finish with JSON containing only "
                    "summary and reusable_lesson; do not expose hidden reasoning.\n"
                    f"Reviewer-protected paths: {protected_note}\n"
                    f"Task: {task['title']}\n"
                    f"Acceptance: {acceptance}\n"
                    f"CI failed-step log:\n{failure_log[:16000]}\n"
                    "Reusable lessons:\n"
                    f"{json.dumps(lessons, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair the failed candidate now. Inspect the relevant "
                    "files before editing."
                ),
            },
        ]

        try:
            final_text = self._tool_stage(
                chat=chat,
                model=developer_model,
                messages=messages,
                functions=[
                    tools.list_project_files,
                    tools.read_project_file,
                    tools.write_project_file,
                    tools.run_candidate_static_checks,
                    tools.show_candidate_diff,
                ],
                rounds=self.config.selfdev.max_tool_rounds,
                force=force,
                branch=branch,
                stage="ci_repair",
            )
        except CyclePaused as exc:
            return EvolutionResult(
                "paused",
                branch,
                workspace,
                f"CI repair paused because the PC became busy: {exc}",
            )

        fallback_error: str | None = None

        if not tools.files_written:
            self._check_resources(force, branch)
            self._emit(
                "Direct CI repair editing stalled; requesting structured fallback change plan"
            )

            try:
                inspected_context = build_read_context(tools)

                response = developer_chat(
                    chat,
                    request_think=self.config.model.think,
                    model=developer_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You correctly analysed a failed candidate but did not make a file edit. "
                                "Now return one strict JSON object with summary, reusable_lesson, and changes. "
                                "changes must be a non-empty list of objects containing path, complete content, "
                                "and reason. Each content value must contain the COMPLETE replacement file, not "
                                "a diff or excerpt. No markdown and no hidden reasoning. The caller will apply "
                                "every proposed file only through CandidateTools.write_project_file, so candidate "
                                "path confinement and file limits remain enforced. Reviewer-controlled "
                                "regression tests are read-only and must not appear in the change plan. "
                                "Repair only the concrete CI failure and do not weaken safety.\n"
                                f"Reviewer-protected paths: {protected_note}\n"
                                f"Task: {task['title']}\n"
                                f"Acceptance: {acceptance}\n"
                                f"CI failed-step log:\n{failure_log[:16000]}\n"
                                f"Your prior diagnosis:\n{final_text[:8000]}\n"
                                f"Candidate files you inspected:\n{inspected_context}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Produce the concrete structured repair change plan now."
                            ),
                        },
                    ],
                    options={"temperature": 0.0},
                )

                fallback_plan = parse_change_plan(
                    self._content(response),
                    tools.max_files,
                )
                apply_change_plan(fallback_plan, tools)

                final_text = json.dumps(
                    {
                        "summary": fallback_plan.summary,
                        "reusable_lesson": fallback_plan.reusable_lesson,
                    }
                )

                self.audit.write(
                    "selfdev_ci_repair_fallback",
                    branch=branch,
                    task_id=task["id"],
                    files_written=len(tools.files_written),
                )

            except Exception as exc:
                fallback_error = (
                    f"{type(exc).__name__}: {exc}"
                )

        if not tools.files_written:
            summary, lesson = self._outcome(
                final_text,
                (
                    "A failed CI run must produce a concrete candidate "
                    "repair before retrying validation."
                ),
            )

            summary += "\n\nNo candidate files changed during CI repair."

            if fallback_error:
                summary += (
                    "\nStructured repair fallback also failed: "
                    + fallback_error
                )

            self.memory.finish_cycle(
                candidate.cycle_id,
                status="failed",
                summary=summary,
                reusable_lesson=lesson,
                checks_passed=False,
                pushed=True,
                validation_state="failed",
            )

            return EvolutionResult(
                "failed",
                branch,
                workspace,
                summary,
                False,
            )

        check_result = tools.run_candidate_static_checks()
        checks_passed = check_result.startswith(
            "static_checks=passed"
        )

        if not checks_passed:
            repaired = self._repair_static_failures(
                chat=chat,
                model=developer_model,
                tools=tools,
                task=task,
                branch=branch,
                cycle_id=candidate.cycle_id,
                check_result=check_result,
                attempts_used=candidate.local_repair_attempts,
                protected_paths=protected_paths,
                force=force,
            )
            check_result = repaired.check_result
            checks_passed = repaired.passed
            if repaired.final_text:
                final_text = repaired.final_text

        if not checks_passed:
            summary, lesson = self._outcome(
                final_text,
                (
                    "Repair candidates must pass static checks before "
                    "being pushed."
                ),
            )

            summary += f"\n\n{check_result}"

            self.memory.finish_cycle(
                candidate.cycle_id,
                status="candidate_needs_work",
                summary=summary,
                reusable_lesson=lesson,
                checks_passed=False,
                pushed=True,
                validation_state="failed",
            )

            return EvolutionResult(
                "candidate_needs_work",
                branch,
                workspace,
                summary,
                False,
            )

        relative_paths = [
            path.relative_to(workspace).as_posix()
            for path in tools.files_written
        ]

        commit = self.github.commit_paths(
            workspace,
            f"repair: {task['title']}",
            relative_paths,
        )

        if not commit.ok:
            summary = (
                "Candidate repair commit failed: "
                f"{commit.stderr or commit.stdout}"
            )

            self.memory.finish_cycle(
                candidate.cycle_id,
                status="failed",
                summary=summary,
                reusable_lesson=(
                    "A repair must create a real diff before committing."
                ),
                checks_passed=True,
                pushed=True,
                validation_state="failed",
            )

            return EvolutionResult(
                "failed",
                branch,
                workspace,
                summary,
                True,
            )

        push = self.github.push_branch(
            workspace,
            branch,
        )

        summary, lesson = self._outcome(
            final_text,
            (
                "Use GitHub CI failures as concrete feedback and repair "
                "the same isolated candidate."
            ),
        )

        summary += f"\n\n{check_result}"

        if push.ok:
            summary += (
                "\n\nRepair pushed; waiting for GitHub CI to validate "
                "the new candidate commit."
            )
            status = "candidate_pending_validation"
        else:
            summary += (
                "\n\nRepair push failed: "
                f"{push.stderr or push.stdout}"
            )
            status = "candidate_needs_work"

        # The original candidate branch already exists remotely, so keep
        # the cycle attached to it even if this particular push fails.
        self.memory.finish_cycle(
            candidate.cycle_id,
            status=status,
            summary=summary,
            reusable_lesson=lesson,
            checks_passed=True,
            pushed=True,
            validation_state=None if push.ok else "failed",
        )

        self.audit.write(
            "selfdev_ci_repair",
            branch=branch,
            task_id=task["id"],
            files_written=len(tools.files_written),
            pushed=push.ok,
            summary=summary[:2000],
        )

        self._emit(
            f"CI repair finished: {status} — "
            f"wrote {len(tools.files_written)} file(s)"
        )

        return EvolutionResult(
            status,
            branch,
            workspace,
            summary,
            True,
        )

    def run_once(self, *, force: bool = False) -> EvolutionResult:
        if not self.config.selfdev.enabled:
            return EvolutionResult("disabled", None, None, "Self-development is disabled in config.")

        sync = self.github.sync_trusted_main()
        self.audit.write(
            "selfdev_main_sync",
            ok=sync.ok,
            updated=sync.updated,
            summary=sync.summary[:2000],
        )
        if not sync.ok:
            return EvolutionResult("sync_blocked", None, None, sync.summary)
        if sync.updated:
            return EvolutionResult(
                "updated",
                None,
                None,
                sync.summary + " Evolve stopped so the next invocation loads the updated code.",
            )

        state = self.governor.sample()
        if not force and not state.background_allowed:
            self.governor.apply_process_priority(idle=False)
            return EvolutionResult("deferred", None, None, f"PC is in use or busy: {state.reason}")
        self.governor.apply_process_priority(idle=True)

        self._reconcile_candidates()

        local_repair = self._repair_local_candidate(force=force)
        if local_repair is not None:
            return local_repair

        repair = self._repair_failed_candidate(force=force)
        if repair is not None:
            return repair

        task = self._load_next_task()
        if not task:
            return EvolutionResult("idle", None, None, "No eligible todo item remains. Merged, validated tasks are skipped.")

        available = available_ollama_models()
        developer_model = select_developer_model(
            self.config.selfdev.developer_model,
            self.config.model.name,
            available,
        )
        slug = "".join(char if char.isalnum() else "-" for char in task["id"].lower()).strip("-")[:40]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch = f"localpilot/candidate-{slug}-{stamp}"
        self._emit(f"Creating candidate for: {task['title']}")
        workspace, is_worktree = self._candidate_workspace(branch)
        tools = CandidateTools(workspace, self.config.selfdev.max_files_per_cycle)
        cycle_id = self.memory.start_cycle(
            task_id=str(task["id"]),
            branch=branch,
            everyday_model=self.config.model.name,
            developer_model=developer_model,
            workspace=workspace,
            is_worktree=is_worktree,
        )
        self.audit.write(
            "selfdev_start",
            task_id=task["id"],
            branch=branch,
            workspace=str(workspace),
            developer_model=developer_model,
        )

        try:
            from ollama import chat
        except ImportError as exc:
            self.memory.finish_cycle(
                cycle_id,
                status="failed",
                summary="Ollama Python package is not installed.",
                reusable_lesson="Verify local model dependencies before starting a development cycle.",
                checks_passed=None,
                pushed=False,
            )
            raise RuntimeError("Ollama Python package is not installed. Run scripts/bootstrap.ps1.") from exc

        lessons = self.memory.reusable_lessons(self.config.selfdev.lesson_limit)
        acceptance = json.dumps(task.get("acceptance", []), ensure_ascii=False)
        research_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are LocalPilot's research-stage developer. Inspect the isolated candidate only. "
                    "Gather concrete repository evidence for the single task below. You cannot write in this stage. "
                    "Return a concise implementation brief with relevant files, constraints, tests, and risks. "
                    "Do not provide or request hidden chain-of-thought.\n"
                    f"Task: {task['title']}\nAcceptance: {acceptance}"
                ),
            },
            {"role": "user", "content": "Research the candidate and return the evidence-based brief."},
        ]
        try:
            research = self._tool_stage(
                chat=chat,
                model=developer_model,
                messages=research_messages,
                functions=[tools.list_project_files, tools.read_project_file],
                rounds=self.config.selfdev.research_tool_rounds,
                force=force,
                branch=branch,
                stage="research",
            )

            implementation_messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are LocalPilot's implementation-stage developer. Modify only the isolated candidate "
                        "through the supplied file tools. Implement one focused task, add/update tests, inspect the "
                        "diff, and run static checks. Never edit stable, execute candidate code locally, promote, "
                        "weaken confinement, bypass the resource governor, or use shell command strings. "
                        "Finish with JSON containing only summary and reusable_lesson; do not expose hidden reasoning.\n"
                        f"Task: {task['title']}\nAcceptance: {acceptance}\n"
                        f"Research brief:\n{research[:12000]}\n"
                        f"Reusable lessons from earlier cycles:\n{json.dumps(lessons, ensure_ascii=False)}"
                    ),
                },
                {"role": "user", "content": "Implement the task now and make concrete candidate changes."},
            ]
            final_text = self._tool_stage(
                chat=chat,
                model=developer_model,
                messages=implementation_messages,
                functions=[
                    tools.list_project_files,
                    tools.read_project_file,
                    tools.write_project_file,
                    tools.run_candidate_static_checks,
                    tools.show_candidate_diff,
                ],
                rounds=self.config.selfdev.max_tool_rounds,
                force=force,
                branch=branch,
                stage="implementation",
            )

            fallback_plan: ChangePlan | None = None
            if not tools.files_written:
                self._check_resources(force, branch)
                self._emit("Direct editing stalled; requesting structured fallback change plan")
                response = developer_chat(
                    chat,
                    request_think=self.config.model.think,
                    model=developer_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return one strict JSON object with summary, reusable_lesson, and changes. "
                                "changes must be a non-empty list of objects containing path, complete content, and reason. "
                                "Only propose files needed for the task. No markdown and no hidden reasoning. "
                                "The caller will validate every path and apply every change through write_project_file.\n"
                                f"Task: {task['title']}\nAcceptance: {acceptance}\nResearch:\n{research[:12000]}"
                            ),
                        },
                        {"role": "user", "content": "Produce the candidate change plan now."},
                    ],
                    options={"temperature": 0.0},
                )
                fallback_plan = parse_change_plan(self._content(response), tools.max_files)
                apply_change_plan(fallback_plan, tools)
                final_text = json.dumps(
                    {"summary": fallback_plan.summary, "reusable_lesson": fallback_plan.reusable_lesson}
                )

            checks_passed: bool | None = None
            check_result = "static checks disabled"
            if self.config.selfdev.run_static_checks:
                self._emit("Running final non-executing static checks")
                check_result = tools.run_candidate_static_checks()
                checks_passed = check_result.startswith("static_checks=passed")
                if tools.files_written and not checks_passed:
                    repaired = self._repair_static_failures(
                        chat=chat,
                        model=developer_model,
                        tools=tools,
                        task=task,
                        branch=branch,
                        cycle_id=cycle_id,
                        check_result=check_result,
                        attempts_used=0,
                        force=force,
                    )
                    check_result = repaired.check_result
                    checks_passed = repaired.passed
                    if repaired.final_text:
                        final_text = repaired.final_text

            status = classify_candidate_result(len(tools.files_written), checks_passed)
            pushed = False
            delivery = ""
            if tools.files_written and is_worktree and checks_passed:
                relative_paths = [path.relative_to(workspace).as_posix() for path in tools.files_written]
                commit = self.github.commit_paths(workspace, f"candidate: {task['title']}", relative_paths)
                if commit.ok and self.config.github.auto_push_candidates:
                    push = self.github.push_branch(workspace, branch)
                    pushed = push.ok
                    if pushed:
                        status = "candidate_pending_validation"
                        delivery = "Candidate pushed; it remains pending until CI passes and its PR is merged."
                    else:
                        status = "candidate_needs_work"
                        delivery = f"Candidate push failed: {push.stderr or push.stdout}"
                elif not commit.ok:
                    status = "candidate_needs_work"
                    delivery = f"Candidate commit failed: {commit.stderr or commit.stdout}"

            summary, lesson = self._outcome(
                final_text,
                "Use repository evidence and candidate-only tools before validating a self-development change.",
            )
            summary = f"{summary}\n\n{check_result}"
            if delivery:
                summary += f"\n\n{delivery}"
            self.memory.finish_cycle(
                cycle_id,
                status=status,
                summary=summary,
                reusable_lesson=lesson,
                checks_passed=checks_passed,
                pushed=pushed,
            )
            self.audit.write(
                "selfdev_end",
                branch=branch,
                task_id=task["id"],
                checks_passed=checks_passed,
                files_read=len(tools.files_read),
                files_written=len(tools.files_written),
                status=status,
                summary=summary[:2000],
            )
            self._emit(f"Finished: {status} — wrote {len(tools.files_written)} candidate file(s)")
            return EvolutionResult(status, branch, workspace, summary, checks_passed)
        except CyclePaused as exc:
            summary = f"User returned or PC became busy: {exc}"
            self.memory.finish_cycle(
                cycle_id,
                status="paused",
                summary=summary,
                reusable_lesson="Re-check the resource governor between every model/tool round and resume later.",
                checks_passed=None,
                pushed=False,
            )
            return EvolutionResult("paused", branch, workspace, summary)
        except Exception as exc:
            summary = f"Cycle failed: {type(exc).__name__}: {exc}"
            recoverable = False
            if is_worktree:
                try:
                    recoverable = bool(
                        self.github.candidate_changed_paths(workspace)
                        or self.github.branch_has_candidate_commit(workspace)
                    )
                except Exception:
                    recoverable = False
            status = "candidate_needs_work" if recoverable else "failed"
            if recoverable:
                summary += " The existing local candidate was retained for the next evolve invocation."
            self.memory.finish_cycle(
                cycle_id,
                status=status,
                summary=summary,
                reusable_lesson=f"Handle {type(exc).__name__} before retrying this task.",
                checks_passed=None,
                pushed=False,
            )
            self.audit.write("selfdev_end", branch=branch, task_id=task["id"], status=status, summary=summary)
            return EvolutionResult(status, branch, workspace, summary)

