from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from localpilot.audit import AuditLog
from localpilot.config import Config
from localpilot.github_integration import GitHubIntegration
from localpilot.resource import ResourceGovernor

_IGNORE_NAMES = {".git", ".github", ".venv", "__pycache__", ".pytest_cache", "localpilot-data"}
_ALLOWED_SUFFIXES = {
    ".py", ".toml", ".md", ".txt", ".json", ".yml", ".yaml", ".ps1", ".gitignore"
}


@dataclass(slots=True)
class EvolutionResult:
    status: str
    branch: str | None
    workspace: Path | None
    summary: str
    tests_passed: bool | None = None


def classify_candidate_result(files_written: int, checks_passed: bool | None) -> str:
    """Return a truthful result state for a self-development cycle."""
    if files_written == 0:
        return "no_changes"
    return "candidate_ready" if checks_passed else "candidate_needs_work"


class CandidateTools:
    """File and test tools confined to one candidate workspace."""

    def __init__(self, workspace: Path, max_files: int = 8) -> None:
        self.workspace = workspace.resolve()
        self.max_files = max_files
        self.files_written: set[Path] = set()
        self.files_read: set[Path] = set()

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
        """List candidate project files. max_results may be 1 to 300."""
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
        """Read a UTF-8 text file inside the candidate workspace."""
        path = self._resolve(relative_path)
        max_chars = max(1000, min(int(max_chars), 100000))
        if not path.is_file():
            return f"File not found: {relative_path}"
        self.files_read.add(path)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def write_project_file(self, relative_path: str, content: str) -> str:
        """Create or fully replace one text file inside the candidate workspace."""
        path = self._resolve(relative_path)
        suffix = path.suffix.lower() if path.name != ".gitignore" else ".gitignore"
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"File type is not allowed for autonomous editing: {suffix or '(none)'})")
        if len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("Candidate file exceeds 1 MB safety limit.")
        if path not in self.files_written and len(self.files_written) >= self.max_files:
            raise RuntimeError("Candidate file-write limit reached for this cycle.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files_written.add(path)
        return f"Wrote {path.relative_to(self.workspace).as_posix()} ({len(content)} chars)."

    def run_candidate_static_checks(self) -> str:
        """Compile Python files and parse TOML without importing candidate code."""
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
                    source = path.read_text(encoding="utf-8", errors="strict")
                    compile(source, str(rel), "exec")
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
        """Show the current candidate diff when the workspace is a Git worktree."""
        completed = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return completed.stderr.strip() or "git diff unavailable"
        return completed.stdout[-30000:] or "(no diff)"


class SelfDeveloper:
    """Creates isolated candidate builds; it never overwrites stable directly."""

    def __init__(
        self,
        config: Config,
        project_root: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.root = Path(project_root).resolve()
        self.data_dir = (self.root / config.agent.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.data_dir / "audit.jsonl")
        self.governor = ResourceGovernor(config.resource)
        self.github = GitHubIntegration(self.root, config.github)
        self.progress = progress or (lambda _message: None)

    def _emit(self, message: str) -> None:
        self.progress(message)
        self.audit.write("selfdev_progress", message=message)

    def _load_next_task(self) -> dict[str, Any] | None:
        path = self.root / "selfdev-backlog.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("tasks", []):
            if item.get("status", "todo") == "todo":
                return item
        return None

    def _candidate_workspace(self, branch: str) -> tuple[Path, bool]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = self.data_dir / "candidates" / f"{stamp}-{branch.split('/')[-1]}"
        if self.github.is_git_repo() and self.github.clean_worktree():
            result = self.github.create_candidate_worktree(branch, destination)
            if result.ok:
                return destination, True
            raise RuntimeError(result.stderr or result.stdout or "Could not create Git worktree.")

        def ignore(directory: str, names: list[str]) -> set[str]:
            return {name for name in names if name in _IGNORE_NAMES or name.endswith(".pyc")}

        shutil.copytree(self.root, destination, ignore=ignore)
        return destination, False

    def run_once(self, *, force: bool = False) -> EvolutionResult:
        if not self.config.selfdev.enabled:
            return EvolutionResult("disabled", None, None, "Self-development is disabled in config.")
        state = self.governor.sample()
        if not force and not state.background_allowed:
            self.governor.apply_process_priority(idle=False)
            return EvolutionResult("deferred", None, None, f"PC is in use or busy: {state.reason}")
        self.governor.apply_process_priority(idle=True)

        task = self._load_next_task()
        if not task:
            return EvolutionResult("idle", None, None, "No todo item exists in selfdev-backlog.json.")

        slug = "".join(c if c.isalnum() else "-" for c in task["id"].lower()).strip("-")[:40]
        branch_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch = f"localpilot/candidate-{slug}-{branch_stamp}"
        self._emit(f"Creating candidate for: {task['title']}")
        workspace, is_worktree = self._candidate_workspace(branch)
        tools = CandidateTools(workspace, self.config.selfdev.max_files_per_cycle)
        self.audit.write("selfdev_start", task=task, branch=branch, workspace=str(workspace))

        try:
            from ollama import chat
        except ImportError as exc:
            raise RuntimeError("Ollama Python package is not installed. Run scripts/bootstrap.ps1.") from exc

        self._emit(f"Loading model {self.config.model.name} via Ollama")
        system = f"""You are the developer instance of LocalPilot. Improve ONE candidate build, never the stable installation.
Task: {task['title']}
Acceptance criteria: {task.get('acceptance', [])}
The candidate workspace is isolated. Use the provided file tools. Keep changes focused and maintain Windows compatibility.
You MUST implement at least one concrete code or test change with write_project_file unless the task is genuinely impossible. Do not spend the whole cycle only reading or analysing. After gathering enough context, edit the candidate, inspect the diff, and run static checks.
Full executable tests run in GitHub Actions after a candidate branch is pushed. Never weaken path confinement, audit logging, stable/developer/candidate separation, or the resource governor.
Do not add cloud pricing or payment machinery. GitHub remains the source-control/CI layer.
When finished, return a concise implementation summary and any remaining risks.
"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Implement the task now. Make a real candidate change rather than only analysing it."},
        ]
        functions = [
            tools.list_project_files,
            tools.read_project_file,
            tools.write_project_file,
            tools.run_candidate_static_checks,
            tools.show_candidate_diff,
        ]
        final_text = ""
        for round_no in range(self.config.selfdev.max_tool_rounds):
            current = self.governor.sample(interval=0.05)
            if not force and not current.background_allowed:
                self.governor.apply_process_priority(idle=False)
                self.audit.write("selfdev_paused", branch=branch, reason=current.reason)
                self._emit(f"Paused: {current.reason}")
                return EvolutionResult("paused", branch, workspace, f"User returned or PC became busy: {current.reason}")

            self._emit(
                f"Tool round {round_no + 1}/{self.config.selfdev.max_tool_rounds} — "
                f"read {len(tools.files_read)}, wrote {len(tools.files_written)} file(s)"
            )
            response = chat(
                model=self.config.model.name,
                messages=messages,
                tools=functions,
                think=self.config.model.think,
                options={"temperature": self.config.model.temperature},
            )
            messages.append(response.message)
            calls = response.message.tool_calls or []
            if not calls:
                final_text = response.message.content or "Candidate cycle completed."
                break
            for call in calls:
                name = call.function.name
                fn = {f.__name__: f for f in functions}.get(name)
                if fn is None:
                    result = f"Unknown candidate tool: {name}"
                else:
                    args = call.function.arguments or {}
                    self.audit.write(
                        "selfdev_tool",
                        branch=branch,
                        tool=name,
                        args={k: ("<content>" if k == "content" else v) for k, v in args.items()},
                    )
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = f"Tool error: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
        else:
            final_text = "Candidate stopped at the self-development tool-call limit."

        checks_passed: bool | None = None
        if self.config.selfdev.run_static_checks:
            self._emit("Running final static checks")
            check_result = tools.run_candidate_static_checks()
            checks_passed = check_result.startswith("static_checks=passed")
            final_text += f"\n\nFinal static check:\n{check_result}"

        if not tools.files_written:
            final_text += (
                "\n\nNo candidate files were changed. This cycle is classified as no_changes, "
                "not candidate_ready."
            )
            self._emit("No files changed; candidate will not be committed or pushed")
        elif is_worktree and checks_passed:
            self._emit(f"Committing {len(tools.files_written)} changed file(s)")
            commit = self.github.commit_all(workspace, f"candidate: {task['title']}")
            if commit.ok and self.config.github.auto_push_candidates:
                self._emit(f"Pushing candidate branch {branch}")
                push = self.github.push_branch(workspace, branch)
                final_text += f"\n\nGit push: {'ok — GitHub Actions will validate the branch' if push.ok else push.stderr}"
            elif not commit.ok:
                final_text += f"\n\nGit commit failed: {commit.stderr or commit.stdout}"

        status = classify_candidate_result(len(tools.files_written), checks_passed)
        self.audit.write(
            "selfdev_end",
            branch=branch,
            checks_passed=checks_passed,
            files_read=len(tools.files_read),
            files_written=len(tools.files_written),
            status=status,
            summary=final_text[:2000],
        )
        self._emit(f"Finished: {status} — read {len(tools.files_read)}, wrote {len(tools.files_written)} file(s)")
        return EvolutionResult(status, branch, workspace, final_text, checks_passed)
