from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import ctypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence
from urllib import request as urllib_request
from urllib.parse import urlparse

import psutil

from localpilot.process import hidden_process_creation_flags
from localpilot.selfdev_results import _ALLOWED_SUFFIXES, _IGNORE_NAMES


class ImplementationStatus(str, Enum):
    COMPLETED = "completed"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    TIMEOUT = "timeout"
    RESOURCE_PRESSURE = "resource_pressure"
    CLI_ERROR = "cli_error"
    MALFORMED_OUTPUT = "malformed_output"
    CANDIDATE_TEST_FAILURE = "candidate_test_failure"
    CONFINEMENT_VIOLATION = "confinement_violation"
    REVIEW_REJECTED = "review_rejected"


@dataclass(frozen=True, slots=True)
class ImplementationPreflight:
    healthy: bool
    backend: str
    model: str
    executable: str
    version: str = ""
    context_tokens: int = 0
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImplementationRequest:
    workspace: Path
    prompt: str
    allowed_paths: tuple[str, ...]
    protected_paths: tuple[str, ...] = ()
    test_commands: tuple[tuple[str, ...], ...] = ()
    review_feedback: str = ""
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class ImplementationResult:
    status: ImplementationStatus
    backend: str
    model: str
    summary: str
    changed_paths: tuple[str, ...] = ()
    diff_digest: str = ""
    tests: tuple[dict[str, Any], ...] = ()
    session_id: str = ""
    exit_code: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    repair_pass: int = 0
    sanitized_output: str = ""


class ImplementationBackend(Protocol):
    name: str

    def preflight(self) -> ImplementationPreflight: ...

    def run(self, request: ImplementationRequest, *, repair_pass: int = 0) -> ImplementationResult: ...


def _bounded(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _relative_path(value: str) -> str:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe candidate-relative path: {value!r}")
    return path.as_posix()


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
        creationflags=hidden_process_creation_flags(),
    )


def candidate_changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = _git(workspace, "status", "--porcelain", "--untracked-files=all")
    if completed.returncode != 0:
        raise RuntimeError(_bounded(completed.stderr or completed.stdout, 1000))
    paths: set[str] = set()
    for row in completed.stdout.splitlines():
        if len(row) < 4:
            continue
        raw = row[3:].split(" -> ")[-1].strip().strip('"')
        paths.add(_relative_path(raw))
    return tuple(sorted(paths))


def candidate_diff_digest(workspace: Path) -> str:
    tracked = _git(workspace, "diff", "--binary", "--", ".")
    staged = _git(workspace, "diff", "--cached", "--binary", "--", ".")
    untracked: list[bytes] = []
    for relative in candidate_changed_paths(workspace):
        path = (workspace / relative).resolve()
        if path.is_file() and not _git(workspace, "ls-files", "--error-unmatch", relative).returncode == 0:
            untracked.append(relative.encode("utf-8") + b"\0" + path.read_bytes())
    payload = tracked.stdout.encode("utf-8", errors="replace") + staged.stdout.encode(
        "utf-8", errors="replace"
    ) + b"".join(untracked)
    return hashlib.sha256(payload).hexdigest() if payload else ""


_SECRET_VALUE = re.compile(
    r"(?i)(anthropic_(?:auth_token|api_key)|github_token|gh_token|api[_-]?key|password)\s*[:=]\s*([^\s,;]+)"
)


def _contains_secret_like_assignment(content: str) -> bool:
    for match in _SECRET_VALUE.finditer(content):
        value = match.group(2).strip("'\"[]{}()")
        if not value or value.lower() in {
            "none", "null", "example", "placeholder", "changeme", "<redacted>", "test"
        }:
            continue
        if len(value) >= 20 or value.startswith(("sk-", "ghp_", "github_pat_")):
            return True
    return False


def sanitize_process_output(value: str, *, workspace: Path, secrets: Iterable[str] = ()) -> str:
    text = str(value or "").replace(str(workspace), "<candidate>")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return _SECRET_VALUE.sub(lambda match: f"{match.group(1)}=<redacted>", text)[-12000:]


def parse_claude_json_output(output: str) -> dict[str, Any]:
    candidates: list[Any] = []
    stripped = str(output or "").strip()
    if stripped:
        try:
            candidates.append(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    for line in reversed(stripped.splitlines()):
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        structured = candidate.get("structured_output")
        if isinstance(structured, dict):
            merged = dict(structured)
            for key in (
                "session_id", "usage", "total_cost_usd", "is_error", "subtype", "stop_reason"
            ):
                if key in candidate:
                    merged[f"_claude_{key}"] = candidate[key]
            return merged
        result = candidate.get("result")
        if isinstance(result, str):
            try:
                inner = json.loads(result)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                merged = dict(inner)
                for key in (
                    "session_id", "usage", "total_cost_usd", "is_error", "subtype", "stop_reason"
                ):
                    if key in candidate:
                        merged[f"_claude_{key}"] = candidate[key]
                return merged
            if candidate.get("subtype") in {"success", "error_max_turns"}:
                return {
                    "summary": _bounded(result, 2000) or "Claude Code returned a completion envelope.",
                    "tests": [],
                    "_claude_session_id": candidate.get("session_id", ""),
                    "_claude_usage": candidate.get("usage", {}),
                    "_claude_is_error": candidate.get("is_error"),
                    "_claude_subtype": candidate.get("subtype"),
                    "_claude_stop_reason": candidate.get("stop_reason"),
                }
        if candidate.get("subtype") in {"success", "error_max_turns"}:
            return {
                "summary": "Claude Code returned a completion envelope without an inner result object.",
                "tests": [],
                "_claude_session_id": candidate.get("session_id", ""),
                "_claude_usage": candidate.get("usage", {}),
                "_claude_is_error": candidate.get("is_error"),
                "_claude_subtype": candidate.get("subtype"),
                "_claude_stop_reason": candidate.get("stop_reason"),
            }
        if any(key in candidate for key in ("summary", "tests", "status")):
            return candidate
    raise ValueError("Claude Code output did not contain the required JSON result object")


def _claude_error_detail(stdout: str, stderr: str, *, workspace: Path) -> str:
    fields: list[str] = []
    for line in reversed(str(stdout or "").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("subtype", "error", "errors", "stop_reason"):
            if payload.get(key) not in (None, "", []):
                fields.append(f"{key}={_bounded(payload[key], 500)}")
        break
    stderr_tail = sanitize_process_output(stderr, workspace=workspace)
    if stderr_tail:
        fields.append(f"stderr={_bounded(stderr_tail, 1000)}")
    return "; ".join(fields)[:1600]


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=2)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            parent.terminate()
            parent.wait(timeout=2)
        except psutil.Error:
            try:
                parent.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        process.kill()


def _trim_windows_gpu_runner_working_sets() -> int:
    """Drop reclaimable host mappings after Ollama has placed the model in VRAM."""
    if os.name != "nt":
        return 0
    trimmed = 0
    for process in psutil.process_iter(("name",)):
        try:
            if str(process.info.get("name") or "").lower() != "llama-server.exe":
                continue
            handle = ctypes.windll.kernel32.OpenProcess(0x0100 | 0x0400, False, process.pid)
            if not handle:
                continue
            try:
                if ctypes.windll.psapi.EmptyWorkingSet(handle):
                    trimmed += 1
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (OSError, psutil.Error):
            continue
    return trimmed


class ClaudeCodeBackend:
    """A bounded Claude Code process confined to one Git candidate checkout."""

    name = "claude_code"

    _REQUIRED_FLAGS = (
        "--allowedTools",
        "--disallowedTools",
        "--model",
        "--output-format",
        "--restricted",
    )

    def __init__(
        self,
        *,
        executable: str = "claude",
        executable_argv: Sequence[str] | None = None,
        model: str = "gpt-oss:20b",
        context_tokens: int = 65536,
        max_turns: int = 40,
        timeout_seconds: float = 600.0,
        max_output_tokens: int = 2048,
        max_output_chars: int = 120_000,
        base_url: str = "http://localhost:11434",
        resource_guard: Callable[[], None] | None = None,
    ) -> None:
        self.executable = str(executable).strip() or "claude"
        self.executable_argv = tuple(executable_argv or (self.executable,))
        self.model = str(model).strip()
        self.context_tokens = int(context_tokens)
        self.max_turns = int(max_turns)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = int(max_output_tokens)
        self.max_output_chars = int(max_output_chars)
        self.base_url = str(base_url).rstrip("/")
        self.resource_guard = resource_guard

    def _run_probe(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.executable_argv, *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            shell=False,
            creationflags=hidden_process_creation_flags(),
        )

    def _ollama_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        probe_base_url = self.base_url.replace("//localhost", "//127.0.0.1", 1)
        req = urllib_request.Request(
            f"{probe_base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
        with opener.open(req, timeout=min(300.0, max(120.0, self.timeout_seconds))) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Ollama {path} returned a non-object response")
        return result

    def preflight(self) -> ImplementationPreflight:
        messages: list[str] = []
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            messages.append("implementation_base_url must be a loopback HTTP Ollama endpoint")
        if self.model != "gpt-oss:20b":
            messages.append("implementation_model must remain gpt-oss:20b for the one-model design")
        if self.context_tokens < 65536:
            messages.append(
                f"configured implementation context is {self.context_tokens}; Claude Code with Ollama requires at least 65536"
            )
        executable_found = bool(self.executable_argv) and (
            len(self.executable_argv) > 1
            or Path(self.executable_argv[0]).is_file()
            or shutil.which(self.executable_argv[0]) is not None
        )
        version = ""
        if not executable_found:
            messages.append(f"Claude Code executable is unavailable: {self.executable_argv[0]}")
        else:
            try:
                version_probe = self._run_probe("--version")
                help_probe = self._run_probe("--help")
                version = _bounded(version_probe.stdout or version_probe.stderr, 300)
                if version_probe.returncode != 0 or help_probe.returncode != 0:
                    messages.append("Claude Code version/help preflight returned a nonzero exit")
                missing = [flag for flag in self._REQUIRED_FLAGS if flag not in help_probe.stdout]
                if missing:
                    messages.append(
                        "installed CLI help did not advertise required flags: " + ", ".join(missing)
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                messages.append(f"Claude Code preflight failed: {type(exc).__name__}: {_bounded(exc, 500)}")

        try:
            model_probe = subprocess.run(
                ["ollama", "show", self.model],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
            if model_probe.returncode != 0:
                messages.append(f"Ollama model is unavailable: {self.model}")
            else:
                self._ollama_json(
                    "/api/generate",
                    {
                        "model": self.model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {"num_ctx": self.context_tokens, "num_predict": 1},
                    },
                )
                processes = self._ollama_json("/api/ps").get("models", [])
                active = next(
                    (
                        item for item in processes
                        if isinstance(item, dict)
                        and str(item.get("name") or item.get("model") or "").split(":", 1)[0]
                        == self.model.split(":", 1)[0]
                    ),
                    None,
                )
                allocated = int(active.get("context_length") or 0) if isinstance(active, dict) else 0
                if allocated < self.context_tokens:
                    messages.append(
                        f"Ollama allocated {allocated or 'unknown'} context tokens after loading {self.model}; "
                        f"at least {self.context_tokens} are required"
                    )
                elif int(active.get("size_vram") or 0) >= int(active.get("size") or 0) * 0.95:
                    _trim_windows_gpu_runner_working_sets()
        except (OSError, subprocess.SubprocessError) as exc:
            messages.append(f"Ollama preflight failed: {type(exc).__name__}: {_bounded(exc, 500)}")
        except (ValueError, json.JSONDecodeError) as exc:
            messages.append(f"Ollama preflight returned invalid data: {type(exc).__name__}: {_bounded(exc, 500)}")

        return ImplementationPreflight(
            healthy=not messages,
            backend=self.name,
            model=self.model,
            executable=self.executable_argv[0] if self.executable_argv else self.executable,
            version=version,
            context_tokens=self.context_tokens,
            messages=tuple(messages),
        )

    @staticmethod
    def _settings() -> str:
        deny = [
            "WebFetch", "WebSearch", "Agent", "Skill", "NotebookEdit", "AskUserQuestion",
            "Read(../**)", "Edit(../**)", "Write(../**)",
            "Read(./localpilot-data/**)", "Read(./training/evals/**)",
            "Read(./training/evolution_execution/acceptance/**)",
            "Edit(./localpilot-data/**)", "Write(./localpilot-data/**)",
            "Edit(./training/evals/**)", "Write(./training/evals/**)",
            "Bash(git commit *)", "Bash(git push *)", "Bash(git checkout *)",
            "Bash(git switch *)", "Bash(git branch *)", "Bash(git reset *)", "Bash(git clean *)",
            "Bash(curl *)", "Bash(wget *)", "Bash(npm *)", "Bash(pip *)", "Bash(uv *)",
            "Bash(rm *)", "Bash(rmdir *)", "Bash(del *)", "Bash(Remove-Item *)",
            "Bash(powershell *)", "Bash(pwsh *)", "Bash(cmd *)", "Bash(bash *)", "Bash(sh *)",
        ]
        return json.dumps({"permissions": {"deny": deny}}, separators=(",", ":"))

    def _command(
        self, *, session_id: str = "", allowed_paths: Sequence[str] = ()
    ) -> list[str]:
        allowed = [
            "Read", "Glob", "Grep",
            "Bash(git status *)", "Bash(git diff *)", "Bash(git log *)",
            "Bash(python -m compileall *)", "Bash(python -m pytest *)",
            "Bash(python -m unittest *)", "Bash(py -m pytest *)", "Bash(py -m unittest *)",
        ]
        for relative in sorted({_relative_path(item) for item in allowed_paths}):
            allowed.extend((f"Edit(./{relative})", f"Write(./{relative})"))
        result_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "maxLength": 2000},
                    "tests": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "maxLength": 300},
                                "passed": {"type": "boolean"},
                                "exit_code": {"type": "integer"},
                                "output_digest": {"type": "string", "maxLength": 128},
                            },
                            "required": ["command", "passed", "exit_code", "output_digest"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["summary", "tests"],
                "additionalProperties": False,
            },
            separators=(",", ":"),
        )
        command = [
            *self.executable_argv,
            "--bare", "--restricted", "--print", "--output-format", "json",
            "--json-schema", result_schema,
            "--model", self.model, "--max-turns", str(self.max_turns),
            "--tools", "Read,Glob,Grep,Edit,Write,Bash",
            "--allowedTools", *allowed,
            "--disallowedTools", *json.loads(self._settings())["permissions"]["deny"],
            "--settings", self._settings(), "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--no-session-persistence", "--no-chrome",
        ]
        return command

    def _environment(self) -> dict[str, str]:
        retained = (
            "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PROGRAMDATA",
            "TEMP", "TMP", "COMSPEC",
            "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
        )
        env = {name: os.environ[name] for name in retained if name in os.environ}
        env.update(
            {
                "ANTHROPIC_AUTH_TOKEN": "ollama",
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_BASE_URL": self.base_url,
                "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
                "CLAUDE_CODE_USE_POWERSHELL_TOOL": "0",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(self.context_tokens),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
                "MAX_THINKING_TOKENS": "0",
                "DISABLE_COMPACT": "1",
                "CLAUDE_CODE_MAX_RETRIES": "1",
                "CLAUDE_CODE_DISABLE_THINKING": "1",
                "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
                "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "localhost,127.0.0.1,::1",
            }
        )
        return env

    def _cancel_ollama_inference(self) -> None:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = self.base_url
        try:
            subprocess.run(
                ["ollama", "stop", self.model],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=False,
                env=env,
                creationflags=hidden_process_creation_flags(),
            )
        except (OSError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _test_environment() -> dict[str, str]:
        retained = (
            "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PROGRAMDATA",
            "TEMP", "TMP", "COMSPEC", "USERPROFILE", "LOCALAPPDATA", "APPDATA",
        )
        env = {name: os.environ[name] for name in retained if name in os.environ}
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "localhost,127.0.0.1,::1",
            }
        )
        return env

    @staticmethod
    def _validate_test_command(command: Sequence[str]) -> tuple[str, ...]:
        argv = tuple(str(item) for item in command)
        if len(argv) < 3 or Path(argv[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError("candidate test commands must use LocalPilot's Python executable")
        if argv[1:3] not in (("-m", "pytest"), ("-m", "unittest")):
            raise ValueError("candidate test command must invoke pytest or unittest as a module")
        if len(argv) > 12 or any(".." in Path(item).parts for item in argv[3:]):
            raise ValueError("candidate test command exceeds its bounded argument policy")
        return argv

    def _run_candidate_tests(
        self, workspace: Path, commands: Sequence[Sequence[str]]
    ) -> tuple[dict[str, Any], ...]:
        evidence: list[dict[str, Any]] = []
        for raw_command in commands[:3]:
            command = self._validate_test_command(raw_command)
            process = subprocess.Popen(
                command,
                cwd=str(workspace),
                env=self._test_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
            try:
                stdout, stderr = process.communicate(timeout=min(180.0, self.timeout_seconds))
                exit_code = int(process.returncode or 0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(process)
                stdout, stderr, exit_code = "", "candidate test timeout", 124
            bounded_output = sanitize_process_output(
                (stdout or "") + "\n" + (stderr or ""), workspace=workspace
            )
            evidence.append(
                {
                    "command": " ".join(("python", *command[1:])),
                    "passed": exit_code == 0,
                    "exit_code": exit_code,
                    "output_digest": hashlib.sha256(bounded_output.encode("utf-8")).hexdigest(),
                }
            )
        return tuple(evidence)

    def _validate_candidate(
        self,
        workspace: Path,
        *,
        allowed_paths: Iterable[str],
        protected_paths: Iterable[str],
    ) -> tuple[tuple[str, ...], str]:
        root = workspace.resolve()
        allowed = {_relative_path(item) for item in allowed_paths}
        protected = {_relative_path(item) for item in protected_paths}
        changed = candidate_changed_paths(root)
        if len(changed) > 500:
            raise PermissionError("candidate changed-path hard ceiling exceeded")
        for relative in changed:
            if relative not in allowed:
                raise PermissionError(f"changed path was outside the grounded implementation plan: {relative}")
            if relative in protected:
                raise PermissionError(f"reviewer-protected path changed: {relative}")
            path = (root / relative).resolve()
            if path != root and root not in path.parents:
                raise PermissionError(f"changed path escaped the candidate workspace: {relative}")
            if path.is_symlink() or not path.is_file():
                raise PermissionError(f"changed path is a deletion, link, or non-file: {relative}")
            rel_path = Path(relative)
            if any(part in _IGNORE_NAMES for part in rel_path.parts):
                raise PermissionError(f"protected candidate path changed: {relative}")
            suffix = rel_path.suffix.lower() if rel_path.name != ".gitignore" else ".gitignore"
            if suffix not in _ALLOWED_SUFFIXES:
                raise PermissionError(f"disallowed candidate file type changed: {relative}")
            if path.stat().st_size > 1_000_000:
                raise PermissionError(f"candidate file exceeds 1 MB: {relative}")
            content = path.read_text(encoding="utf-8", errors="strict")
            if _contains_secret_like_assignment(content):
                raise PermissionError(f"candidate content resembles a secret assignment: {relative}")
        return changed, candidate_diff_digest(root)

    def run(self, request: ImplementationRequest, *, repair_pass: int = 0) -> ImplementationResult:
        workspace = Path(request.workspace).resolve()
        started = time.monotonic()
        if not workspace.is_dir() or not (workspace / ".git").exists():
            return ImplementationResult(
                ImplementationStatus.CONFINEMENT_VIOLATION, self.name, self.model,
                "Candidate workspace is not an isolated Git checkout.", repair_pass=repair_pass,
            )
        preflight = self.preflight()
        if not preflight.healthy:
            return ImplementationResult(
                ImplementationStatus.BACKEND_UNAVAILABLE, self.name, self.model,
                "; ".join(preflight.messages), repair_pass=repair_pass,
            )

        def stopped_result(
            status: ImplementationStatus,
            summary: str,
            *,
            exit_code: int | None,
            duration: float,
            output: str = "",
        ) -> ImplementationResult:
            try:
                stopped_paths, stopped_digest = self._validate_candidate(
                    workspace,
                    allowed_paths=request.allowed_paths,
                    protected_paths=request.protected_paths,
                )
            except (OSError, UnicodeError, ValueError, RuntimeError, PermissionError) as exc:
                return ImplementationResult(
                    ImplementationStatus.CONFINEMENT_VIOLATION,
                    self.name,
                    self.model,
                    f"Post-run candidate validation failed: {type(exc).__name__}: {_bounded(exc, 1000)}",
                    exit_code=exit_code,
                    duration_seconds=duration,
                    repair_pass=repair_pass,
                    sanitized_output=output,
                )
            return ImplementationResult(
                status,
                self.name,
                self.model,
                summary,
                changed_paths=stopped_paths,
                diff_digest=stopped_digest,
                exit_code=exit_code,
                duration_seconds=duration,
                repair_pass=repair_pass,
                sanitized_output=output,
            )
        prompt = request.prompt
        if request.review_feedback:
            prompt += (
                "\n\nLocalPilot independent review rejected the prior result. Repair the same candidate and rerun "
                "the relevant repository tests. Address only this bounded feedback:\n" + request.review_feedback[:12000]
            )
        env = self._environment()
        try:
            process = subprocess.Popen(
                self._command(
                    session_id=request.session_id, allowed_paths=request.allowed_paths
                ), cwd=str(workspace), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=False, creationflags=hidden_process_creation_flags(),
            )
        except OSError as exc:
            return ImplementationResult(
                ImplementationStatus.BACKEND_UNAVAILABLE, self.name, self.model,
                f"Claude Code could not start: {type(exc).__name__}: {_bounded(exc, 500)}",
                repair_pass=repair_pass,
            )
        try:
            stdout = stderr = ""
            first = True
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= self.timeout_seconds:
                    _kill_process_tree(process)
                    self._cancel_ollama_inference()
                    return stopped_result(
                        ImplementationStatus.TIMEOUT,
                        f"Claude Code exceeded the {self.timeout_seconds:.0f}s timeout.",
                        exit_code=process.poll(), duration=elapsed,
                    )
                try:
                    stdout, stderr = process.communicate(
                        input=prompt if first else None,
                        timeout=min(1.0, self.timeout_seconds - elapsed),
                    )
                    break
                except subprocess.TimeoutExpired:
                    first = False
                    if self.resource_guard is not None:
                        try:
                            self.resource_guard()
                        except Exception as exc:
                            _kill_process_tree(process)
                            self._cancel_ollama_inference()
                            return stopped_result(
                                ImplementationStatus.RESOURCE_PRESSURE,
                                f"Claude Code stopped at the resource boundary: {type(exc).__name__}: {_bounded(exc, 500)}",
                                exit_code=process.poll(), duration=time.monotonic() - started,
                            )
            duration = time.monotonic() - started
        finally:
            if process.poll() is None:
                _kill_process_tree(process)
        combined = (stdout or "") + "\n" + (stderr or "")
        sanitized = sanitize_process_output(
            combined, workspace=workspace, secrets=(env.get("ANTHROPIC_AUTH_TOKEN", ""),)
        )
        if len(combined) > self.max_output_chars:
            return stopped_result(
                ImplementationStatus.RESOURCE_PRESSURE,
                f"Claude Code output exceeded the {self.max_output_chars}-character limit.",
                exit_code=process.returncode, duration=duration, output=sanitized,
            )
        try:
            changed, digest = self._validate_candidate(
                workspace, allowed_paths=request.allowed_paths, protected_paths=request.protected_paths
            )
        except (OSError, UnicodeError, ValueError, RuntimeError, PermissionError) as exc:
            return ImplementationResult(
                ImplementationStatus.CONFINEMENT_VIOLATION, self.name, self.model,
                f"Post-run candidate validation failed: {type(exc).__name__}: {_bounded(exc, 1000)}",
                exit_code=process.returncode, duration_seconds=duration, repair_pass=repair_pass,
                sanitized_output=sanitized,
            )
        payload: dict[str, Any] | None = None
        parse_error: ValueError | None = None
        try:
            payload = parse_claude_json_output(stdout)
        except ValueError as exc:
            parse_error = exc
        envelope_subtype = str(payload.get("_claude_subtype") or "") if payload else ""
        explicit_envelope_success = bool(
            payload
            and envelope_subtype == "success"
            and payload.get("_claude_is_error") is not True
        )
        bounded_turn_completion = bool(
            payload and envelope_subtype == "error_max_turns" and changed
        )
        if process.returncode != 0 and not (explicit_envelope_success or bounded_turn_completion):
            detail = _claude_error_detail(stdout, stderr, workspace=workspace)
            return ImplementationResult(
                ImplementationStatus.CLI_ERROR, self.name, self.model,
                f"Claude Code exited with code {process.returncode}."
                + (f" {detail}" if detail else ""),
                changed_paths=changed, diff_digest=digest,
                exit_code=process.returncode, duration_seconds=duration, repair_pass=repair_pass,
                sanitized_output=sanitized,
            )
        if payload is None:
            return ImplementationResult(
                ImplementationStatus.MALFORMED_OUTPUT, self.name, self.model,
                str(parse_error or "Claude Code returned no structured result"),
                changed_paths=changed, diff_digest=digest,
                exit_code=process.returncode, duration_seconds=duration, repair_pass=repair_pass,
                sanitized_output=sanitized,
            )
        tests_raw = payload.get("tests")
        reported_tests = (
            tuple(item for item in tests_raw if isinstance(item, dict))
            if isinstance(tests_raw, list) else ()
        )
        try:
            verified_tests = self._run_candidate_tests(workspace, request.test_commands)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return ImplementationResult(
                ImplementationStatus.CANDIDATE_TEST_FAILURE, self.name, self.model,
                f"LocalPilot could not verify candidate tests: {type(exc).__name__}: {_bounded(exc, 500)}",
                changed_paths=changed, diff_digest=digest, exit_code=process.returncode,
                duration_seconds=duration, repair_pass=repair_pass, sanitized_output=sanitized,
            )
        tests = verified_tests or reported_tests
        tests_ok = bool(tests) and all(bool(item.get("passed")) for item in tests)
        status = ImplementationStatus.COMPLETED if tests_ok else ImplementationStatus.CANDIDATE_TEST_FAILURE
        usage = payload.get("_claude_usage") if isinstance(payload.get("_claude_usage"), dict) else {}
        return ImplementationResult(
            status, self.name, self.model,
            _bounded(payload.get("summary") or "Claude Code completed the implementation stage.", 2000),
            changed_paths=changed, diff_digest=digest, tests=tests,
            session_id=_bounded(payload.get("_claude_session_id"), 200),
            exit_code=process.returncode, usage=dict(usage), duration_seconds=duration,
            repair_pass=repair_pass, sanitized_output=sanitized,
        )
