from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import subprocess
import time
from typing import BinaryIO
import uuid

import psutil

from localpilot.process import current_process_elevated
from localpilot.process import hidden_process_creation_flags


def _option(arguments: list[str], name: str) -> str | None:
    values = []
    for index, value in enumerate(arguments):
        if value == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"Missing {name} value")
            values.append(arguments[index + 1])
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1:])
    if len(values) > 1:
        raise ValueError(f"Ambiguous {name} value")
    return values[0] if values else None


def _cli_desktop(arguments: list[str]) -> bool:
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--config":
            index += 2
        elif value.startswith("--config="):
            index += 1
        else:
            return value == "desktop"
    return False


def assert_no_legacy_desktop(root: Path, *, retired_owners: dict[int, float] | None = None) -> None:
    """Do not create a second UI alongside a version predating the owner lock."""
    excluded = {os.getpid()}
    try:
        excluded.update(process.pid for process in psutil.Process().parents())
    except psutil.Error:
        pass
    modules = {
        "localpilot.native_avatar_companion", "localpilot.native_avatar",
        "localpilot.native_avatar_legacy", "localpilot.webview_app", "localpilot.desktop",
    }
    for process in psutil.process_iter():
        if process.pid in excluded:
            continue
        try:
            retired = (retired_owners or {}).get(process.pid)
            if retired is not None and abs(process.create_time() - retired) < 0.001:
                continue
            argv = [str(value) for value in process.cmdline()]
            module_index = next(index + 1 for index, value in enumerate(argv[:-1]) if value == "-m")
            module = argv[module_index]
            if module not in modules and not (module == "localpilot.cli" and _cli_desktop(argv[module_index + 1:])):
                continue
            selected_root = _option(argv[module_index + 1:], "--root")
            if selected_root is not None:
                selected = Path(selected_root)
                if not selected.is_absolute():
                    selected = Path(process.cwd()) / selected
            else:
                selected = Path(process.cwd())
            if selected.resolve() == root.resolve():
                raise RuntimeError(
                    "An older LocalPilot desktop is already open. Finish the reply and save your draft, "
                    "then close that desktop before opening the updated version."
                )
        except (psutil.Error, OSError, ValueError, IndexError, StopIteration):
            continue


class DesktopInstance:
    """One desktop owner per checkout, released by the OS even after a crash.

    A repeated launch only requests activation. It never replaces a running UI,
    so its active response and composer stay in the same process.
    """

    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        chosen_config = Path(config_path or os.environ.get("LOCALPILOT_CONFIG", "localpilot.toml"))
        self.config_path = (chosen_config if chosen_config.is_absolute() else self.root / chosen_config).resolve()
        self.elevated = current_process_elevated()
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True,
                text=True, timeout=5, creationflags=hidden_process_creation_flags(),
            )
            self.source_sha = head.stdout.strip() if head.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            self.source_sha = None
        self.directory = self.root / "localpilot-data" / "desktop-instance"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = self.directory / "owner.json"
        self.token = uuid.uuid4().hex
        self._handle: BinaryIO | None = None
        self._owned = False
        self.retired_owners: dict[int, float] = {}

    def acquire(self) -> bool:
        handle = (self.directory / "owner.lock").open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        self._handle = handle
        self._owned = True
        temporary = self.directory / f"owner-{self.token}.tmp"
        try:
            temporary.write_text(
                json.dumps({
                    "root": str(self.root),
                    "config_path": str(self.config_path),
                    "elevated": self.elevated,
                    "source_sha": self.source_sha,
                    "pid": os.getpid(),
                    "created_at": psutil.Process().create_time(),
                    "token": self.token,
                }),
                encoding="utf-8",
            )
            temporary.replace(self.metadata)
        except BaseException:
            self.close()
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def request_activation(self, *, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            try:
                owner = json.loads(self.metadata.read_text(encoding="utf-8"))
                token = owner["token"]
                process = psutil.Process(int(owner["pid"]))
                valid = (
                    Path(owner["root"]).resolve() == self.root
                    and isinstance(token, str)
                    and len(token) == 32
                    and all(character in "0123456789abcdef" for character in token)
                    and abs(process.create_time() - float(owner["created_at"])) < 0.001
                )
                if valid:
                    if Path(owner.get("config_path", "")).resolve() != self.config_path:
                        raise RuntimeError(
                            "LocalPilot is already open with a different configuration. "
                            "Finish the reply and save your draft before closing it and changing configuration."
                        )
                    if self.elevated is True and owner.get("elevated") is not True:
                        raise RuntimeError(
                            "The existing LocalPilot desktop does not have verified administrator rights. "
                            "Finish the reply and save your draft, then close it and use the administrator shortcut."
                        )
                    replacement = bool(self.source_sha and owner.get("source_sha") and self.source_sha != owner["source_sha"])
                    if replacement:
                        self.retired_owners[process.pid] = process.create_time()
                        # The old owner releases its lock just before Python
                        # exits. Retire its verified launcher ancestors too so
                        # a still-unwinding venv redirector is not a legacy UI.
                        try:
                            for parent in process.parents():
                                self.retired_owners[parent.pid] = parent.create_time()
                        except psutil.Error:
                            pass
                    action = "replace" if replacement else "activate"
                    (self.directory / f"{action}-{token}-{uuid.uuid4().hex}.request").touch(exist_ok=False)
                    return not replacement
            except (OSError, psutil.Error, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def activation_requested(self) -> bool:
        return self._consume_requests("activate")

    def replacement_requested(self) -> bool:
        return self._consume_requests("replace")

    def _consume_requests(self, action: str) -> bool:
        if not self._owned:
            return False
        requested = False
        for path in self.directory.glob(f"{action}-{self.token}-*.request"):
            try:
                path.unlink()
                requested = True
            except FileNotFoundError:
                pass
        return requested

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self.activation_requested()
            self.replacement_requested()
            self.metadata.unlink(missing_ok=True)
        finally:
            self._owned = False
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
