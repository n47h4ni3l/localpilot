from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


class RuntimeSupervisor:
    """Supervise the replaceable operator/PowerShell worker process behind the broker."""

    def __init__(
        self,
        root: str | Path,
        *,
        config_path: str | Path | None = None,
        restart_limit: int = 5,
        on_message: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config_path = str(Path(config_path).resolve()) if config_path else None
        self.restart_limit = max(1, int(restart_limit))
        self.on_message = on_message or (lambda message: None)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stopping = False
        self._consecutive_failures = 0

    @property
    def pid(self) -> int | None:
        with self._lock:
            process = self._process
            return process.pid if process is not None and process.poll() is None else None

    @property
    def running(self) -> bool:
        return self.pid is not None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stopping = False
            self._launch_locked(restarting=False)

    def _launch_locked(self, *, restarting: bool) -> None:
        argv = [sys.executable, "-m", "localpilot.runtime_worker", "--root", str(self.root)]
        if self.config_path:
            argv.extend(["--config", self.config_path])
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            argv,
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            creationflags=creationflags,
        )
        self._process = process
        self.on_message(
            {
                "kind": "supervisor",
                "type": "runtime.restarting" if restarting else "runtime.starting",
                "payload": {"pid": process.pid},
            }
        )
        threading.Thread(target=self._read_stdout, args=(process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process,), daemon=True).start()
        threading.Thread(target=self._watch, args=(process,), daemon=True).start()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.on_message(
                    {
                        "kind": "supervisor",
                        "type": "runtime.protocol_error",
                        "payload": {"reason": "worker emitted invalid JSON"},
                    }
                )
                continue
            if message.get("kind") == "ready":
                self._consecutive_failures = 0
            self.on_message(message)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            if line.strip():
                self.on_message(
                    {
                        "kind": "supervisor",
                        "type": "runtime.stderr",
                        "payload": {"diagnostic": True},
                    }
                )

    def _watch(self, process: subprocess.Popen[str]) -> None:
        returncode = process.wait()
        with self._lock:
            if self._process is not process:
                return
            self._process = None
            if self._stopping:
                return
            self._consecutive_failures += 1
            failure = self._consecutive_failures
        self.on_message(
            {
                "kind": "supervisor",
                "type": "runtime.exited",
                "payload": {"returncode": returncode, "restart_attempt": failure},
            }
        )
        if failure > self.restart_limit:
            self.on_message(
                {
                    "kind": "supervisor",
                    "type": "runtime.unavailable",
                    "payload": {"restart_limit": self.restart_limit},
                }
            )
            return
        time.sleep(min(0.25 * (2 ** (failure - 1)), 4.0))
        with self._lock:
            if not self._stopping and self._process is None:
                self._launch_locked(restarting=True)

    def send(self, message: dict[str, Any]) -> None:
        if not self.running:
            self.start()
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._write_lock:
            with self._lock:
                process = self._process
                if process is None or process.stdin is None or process.poll() is not None:
                    raise RuntimeError("LocalPilot runtime is unavailable")
                process.stdin.write(encoded + "\n")
                process.stdin.flush()

    def restart(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
