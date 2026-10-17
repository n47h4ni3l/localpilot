"""Nestra's local Claude Code tool, with proposals kept off the running checkout.

Claude Code is a tool harness here, not a second model/provider. A foreground
request receives a fresh, filtered snapshot of committed source and a private
CLI home. Changes remain proposals for review; this module never applies,
commits to, or pushes the owner's repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

import psutil

from localpilot.config import Config
from localpilot.implementation_backend import (
    ClaudeCodeBackend,
    ImplementationRequest,
    ImplementationStatus,
    sanitize_process_output,
)
from localpilot.process import hidden_process_creation_flags
from localpilot.resource import ResourceGovernor
from localpilot.safety import RiskLevel, SafetyPolicy
from localpilot.selfdev_results import _ALLOWED_SUFFIXES, _IGNORE_NAMES
from localpilot.tools.repository import RepositoryReader


_PRIVATE_PREFIXES = (
    "training/evals", "training/outputs", "training/reports",
    "training/checkpoints", "training/evolution_execution/acceptance",
)
_PRIVATE_NAMES = {".claude", ".codex", ".gitmodules", ".gitconfig", ".lfsconfig", "localpilot.toml"}
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
_MAX_FILE_BYTES = 1_000_000
_MAX_SNAPSHOT_FILES = 5000
_MAX_PROPOSAL_BYTES = 8 * 1024 * 1024


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise ValueError("Claude Code paths must be explicit bounded relative file paths.")
    normalized = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    parts = normalized.split("/")
    if (
        windows.drive or windows.root or normalized.startswith("/")
        or any(part in {"", ".", ".."} or part.endswith((".", " ")) for part in parts)
        or any(ord(char) < 32 or char in ':*?[]()\"<>|' for char in normalized)
        or any(PureWindowsPath(part).is_reserved() for part in parts)
    ):
        raise ValueError("Claude Code paths must not contain traversal, drives, patterns or special names.")
    return PurePosixPath(normalized).as_posix()


def _private(relative: str) -> bool:
    path = Path(relative)
    folded = relative.casefold()
    return (
        RepositoryReader._is_sensitive(path)
        or bool({part.casefold() for part in path.parts} & (_PRIVATE_NAMES | _IGNORE_NAMES))
        or any(folded == prefix or folded.startswith(prefix + "/") for prefix in _PRIVATE_PREFIXES)
    )


def _no_links(path: Path) -> None:
    cursor = path
    while True:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("Claude Code storage must not traverse a link or junction.")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def _storage(root: Path, config: Config) -> Path:
    relative = _relative(str(config.agent.data_dir))
    if any(part.casefold() in {".git", ".claude", ".codex"} for part in Path(relative).parts):
        raise ValueError("Claude Code storage must use the configured application data directory.")
    path = root / relative / "claude-code"
    _no_links(path)
    return path


@contextmanager
def claude_code_run_lock(data_dir: str | Path) -> Iterator[None]:
    """One CLI job per application store, including other broker/worker processes.

    Callers pass the application's data directory, not a candidate directory.
    Dead-owner locks are retained as evidence; a live/reused/unknown PID never
    authorizes starting a second job.
    """
    directory = Path(data_dir) / "claude-code"
    _no_links(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "active.json"
    token = uuid.uuid4().hex
    owner = {"pid": os.getpid(), "started_at": psutil.Process().create_time(), "token": token}
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if attempt:
                raise RuntimeError("Claude Code is already running; wait for the existing tool call.")
            try:
                if lock.stat().st_size > 4096:
                    raise ValueError("oversized lock")
                previous = json.loads(lock.read_text(encoding="utf-8"))
                process = psutil.Process(int(previous["pid"]))
                # A reused PID is also treated as busy until its identity can
                # be established; do not delete another live process's lock.
                process.create_time()
            except psutil.NoSuchProcess:
                lock.rename(directory / f"active.stale-{uuid.uuid4().hex}.json")
                continue
            except (OSError, ValueError, KeyError, TypeError, psutil.Error):
                raise RuntimeError("Claude Code's existing run identity could not be verified.") from None
            raise RuntimeError("Claude Code is already running; wait for the existing tool call.")
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(owner, stream, sort_keys=True)
            break
    try:
        yield
    finally:
        try:
            current = json.loads(lock.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock.unlink()
        except (OSError, ValueError):
            pass


class _PrivateClaudeCodeBackend(ClaudeCodeBackend):
    def __init__(self, *, private_home: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.private_home = private_home

    def _environment(self) -> dict[str, str]:
        env = super()._environment()
        home = self.private_home
        for directory in (home, home / "AppData/Roaming", home / "AppData/Local", home / "tmp", home / ".claude"):
            directory.mkdir(parents=True, exist_ok=True)
        env.update({
            "HOME": str(home), "USERPROFILE": str(home),
            "APPDATA": str(home / "AppData/Roaming"), "LOCALAPPDATA": str(home / "AppData/Local"),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "TEMP": str(home / "tmp"), "TMP": str(home / "tmp"),
            "DISABLE_AUTOUPDATER": "1",
        })
        return env

    def _run_probe(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.executable_argv, *args], capture_output=True, text=True, timeout=20,
            check=False, shell=False, env=self._environment(), creationflags=hidden_process_creation_flags(),
        )


class ClaudeCodeTool:
    def __init__(self, project_root: str | Path, config: Config) -> None:
        self.root = Path(project_root).resolve()
        self.config = config
        self.governor = ResourceGovernor(config.resource)

    @staticmethod
    def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1"})
        completed = subprocess.run(
            ["git", "-C", str(root), "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false", *args],
            input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            timeout=30, check=False, shell=False, creationflags=hidden_process_creation_flags(),
        )
        if completed.returncode:
            raise RuntimeError("Claude Code source snapshot failed: " + completed.stderr.decode("utf-8", errors="replace")[:500])
        return completed.stdout

    def _resource_guard(self, *, during_inference: bool = False) -> None:
        sample = self.governor.sample(interval=0.05)
        if during_inference:
            # The CLI's own CPU load is expected, but memory remains a hard stop.
            if sample.memory_percent > self.config.resource.max_memory_percent_for_background:
                raise RuntimeError("Claude Code stopped at the configured memory boundary.")
        elif not sample.capacity_allowed:
            raise RuntimeError("Claude Code deferred: " + sample.capacity_reason)

    def _backend(self, private_home: Path) -> ClaudeCodeBackend:
        cfg = self.config.selfdev
        return _PrivateClaudeCodeBackend(
            private_home=private_home, executable=cfg.implementation_executable,
            model=self.config.model.name, context_tokens=cfg.implementation_context_tokens,
            max_turns=cfg.implementation_max_turns, timeout_seconds=cfg.implementation_timeout_seconds,
            max_output_tokens=cfg.implementation_max_output_tokens, max_output_chars=cfg.implementation_max_output_chars,
            base_url=cfg.implementation_base_url,
            resource_guard=lambda: self._resource_guard(during_inference=True),
        )

    def _snapshot(self, destination: Path) -> tuple[str, int]:
        revision = self._git(self.root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        rows = self._git(self.root, "ls-tree", "-rlz", "--full-tree", revision).split(b"\0")
        selected: list[tuple[str, str, int]] = []
        total = omitted = 0
        for row in rows:
            if not row:
                continue
            metadata, raw_path = row.split(b"\t", 1)
            mode, kind, digest, raw_size = metadata.split()
            try:
                relative = _relative(raw_path.decode("utf-8", errors="strict"))
            except (ValueError, UnicodeError):
                omitted += 1
                continue
            if kind != b"blob" or mode not in {b"100644", b"100755"} or _private(relative):
                omitted += 1
                continue
            size = int(raw_size)
            if size > _MAX_FILE_BYTES:
                omitted += 1
                continue
            total += size
            if total > _MAX_SNAPSHOT_BYTES or len(selected) >= _MAX_SNAPSHOT_FILES:
                raise RuntimeError("Claude Code's bounded source snapshot limit was exceeded.")
            selected.append((relative, digest.decode("ascii"), size))
        # Each blob is read by identity, never through a working-tree filter,
        # smudge command, submodule, symlink, Git hook or source repository config.
        destination.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_LAZY_FETCH": "1"})
        process = subprocess.Popen(
            ["git", "-C", str(self.root), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, shell=False, creationflags=hidden_process_creation_flags(),
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            for relative, digest, size in selected:
                process.stdin.write((digest + "\n").encode("ascii"))
                process.stdin.flush()
                header = process.stdout.readline().split()
                if header != [digest.encode("ascii"), b"blob", str(size).encode("ascii")]:
                    raise RuntimeError("A source blob changed during Claude Code snapshot preparation.")
                content = process.stdout.read(size)
                if len(content) != size or process.stdout.read(1) != b"\n":
                    raise RuntimeError("Incomplete source blob in Claude Code snapshot.")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            process.stdin.close()
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError("Claude Code source blob reader failed.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if self._git(self.root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() != revision:
            raise RuntimeError("Source HEAD changed during Claude Code snapshot preparation.")
        self._git(destination, "init", "--quiet", "--template=")
        self._git(destination, "add", "--force", "--", ".")
        self._git(destination, "-c", "user.name=Nestra proposal", "-c", "user.email=nestra@localhost", "-c", "commit.gpgsign=false", "commit", "--quiet", "--no-verify", "-m", "Isolated source snapshot")
        return revision, omitted

    def claude_code(self, task: str, allowed_paths: list[str] | None = None) -> dict:
        """Ask local Nestra through Claude Code; optional explicit paths permit proposal edits only.

        Uses committed HEAD source; unsaved/uncommitted working-tree edits are
        excluded and preserved. Without allowed_paths this is a read-only question.
        With paths, edits stay in a new isolated proposal and are not applied
        to the live repository. Never pass absolute paths or private files.
        """
        if not isinstance(task, str) or not task.strip() or len(task) > 12000:
            raise ValueError("Claude Code needs a non-empty task of at most 12000 characters.")
        if allowed_paths is not None and (not isinstance(allowed_paths, list) or len(allowed_paths) > 64):
            raise ValueError("Claude Code allowed_paths must be a list of at most 64 explicit relative files.")
        allowed = tuple(sorted({_relative(item) for item in (allowed_paths or [])}))
        for relative in allowed:
            suffix = Path(relative).suffix.lower() if Path(relative).name != ".gitignore" else ".gitignore"
            if _private(relative) or suffix not in _ALLOWED_SUFFIXES or suffix == ".zip":
                raise ValueError("Claude Code cannot write a protected, private or unsupported path: " + relative)
        cfg = self.config
        policy = SafetyPolicy(cfg.safety.auto_allow_read_only, cfg.safety.auto_allow_reversible, cfg.safety.require_confirmation_for_destructive)
        if not policy.permits_without_confirmation(RiskLevel.REVERSIBLE):
            raise PermissionError("Claude Code requires the existing reversible-tool permission.")
        if cfg.model.provider != "ollama" or not cfg.model.name.strip() or any(
            name != cfg.model.name for name in (cfg.selfdev.implementation_model, cfg.selfdev.developer_model)
        ):
            raise ValueError("Claude Code requires the exact accepted local model for chat, development and implementation; no substitution is allowed.")
        self._resource_guard()
        storage = _storage(self.root, cfg)
        with claude_code_run_lock(storage.parent):
            run = storage / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex)
            run.mkdir()
            workspace = run / "workspace"
            started = time.monotonic()
            revision, omitted = self._snapshot(workspace)
            for relative in allowed:
                if (self.root / relative).exists() and not (workspace / relative).is_file():
                    raise ValueError("Claude Code cannot edit source omitted from its bounded snapshot: " + relative)
            self._resource_guard()
            prompt = (
                "You are Nestra using Claude Code as a local tool harness with the same accepted model. "
                "Answer the bounded task from this isolated source snapshot. "
                f"The source is commit {revision}; uncommitted working-tree edits are not included. "
                "Never claim to have inspected uncommitted owner changes. Never access parent/absolute paths, "
                "private data, credentials, the web, package managers or external services. Never commit, push, "
                "change branches, delete files, weaken tests or claim changes were applied to production. "
                "Only these exact files may be edited: " + json.dumps(allowed) + ". "
                + ("This request is read-only; report a concise useful answer, with no invented test runs.\n" if not allowed else
                   "Edits are proposals awaiting independent review. Report only actual tests and concise results.\n")
                + "Task:\n" + task.strip()
            )
            backend = self._backend(run / "private-home")
            result = backend.run(ImplementationRequest(workspace=workspace, prompt=prompt, allowed_paths=allowed, read_only=not allowed))
            proposal_path = ""
            proposal_hash = ""
            if result.changed_paths and result.status != ImplementationStatus.CONFINEMENT_VIOLATION:
                # Intent-to-add represents newly proposed files in the patch
                # without committing them or touching the running repository.
                self._git(workspace, "add", "--intent-to-add", "--force", "--", *result.changed_paths)
                patch = self._git(workspace, "diff", "--binary", "HEAD", "--", *allowed)
                if len(patch) > _MAX_PROPOSAL_BYTES:
                    raise RuntimeError("Claude Code proposal exceeded the bounded diff limit; isolated evidence was retained.")
                proposal = run / "proposal.diff"
                proposal.write_bytes(patch)
                proposal_path, proposal_hash = str(proposal), hashlib.sha256(patch).hexdigest()
            summary = result.summary
            if proposal_path and result.status == ImplementationStatus.CANDIDATE_TEST_FAILURE and not result.tests:
                summary = "Edit proposal saved; test verification is incomplete. " + summary
            payload = {
                "status": result.status.value, "backend": result.backend, "model": result.model,
                "summary": sanitize_process_output(summary, workspace=workspace)[:2000],
                "source_revision": revision, "snapshot_omitted_files": omitted,
                "source_basis": "committed_head_snapshot", "working_tree_included": False,
                "read_only": not allowed, "changed_paths": list(result.changed_paths),
                "diff_digest": result.diff_digest, "proposal_path": proposal_path, "proposal_sha256": proposal_hash,
                "workspace": str(workspace), "applied_to_production": False,
                "review_status": "not_applicable" if not allowed else "pending_review",
                "pending_review": bool(allowed),
                "tests": list(result.tests)[:20], "tests_independently_verified": False,
                "tests_status": "not_run" if not allowed else "unverified",
                "tests_not_run": bool(result.usage.get("tests_not_run", False)),
                "transport": result.usage.get("transport", {}),
                "duration_seconds": round(time.monotonic() - started, 3),
            }
            (run / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return payload
