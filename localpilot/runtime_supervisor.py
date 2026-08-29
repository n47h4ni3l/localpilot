from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from localpilot.audit import AuditLog


_COORDINATED_RESTART_SOURCES = {
    "broker_requested_restart",
    "explicit_api_restart",
    "watchdog",
    "update",
    "fatal_condition",
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class RuntimeSupervisor:
    """Supervise the replaceable operator/PowerShell worker process behind the broker."""

    def __init__(
        self,
        root: str | Path,
        *,
        config_path: str | Path | None = None,
        restart_limit: int = 5,
        on_message: Callable[[dict[str, Any]], None] | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config_path = str(Path(config_path).resolve()) if config_path else None
        self.restart_limit = max(1, int(restart_limit))
        self.on_message = on_message or (lambda message: None)
        self.audit = AuditLog(audit_path or (self.root / "localpilot-data" / "audit.jsonl"))
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stopping = False
        self._consecutive_failures = 0
        self._process_started_at: str | None = None
        self._launch_context: dict[str, Any] = {}
        self._planned_restart: dict[str, Any] | None = None
        self._requests: dict[str, dict[str, Any]] = {}
        self._active_request_id: str | None = None

    def _record_lifecycle(self, transition: str, **fields: Any) -> dict[str, Any]:
        row = self.audit.write("runtime_lifecycle", transition=transition, **fields)
        self.on_message(
            {
                "kind": "supervisor",
                "type": "runtime.lifecycle",
                "payload": {key: value for key, value in row.items() if key != "event"},
            }
        )
        return row

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
            self._launch_locked(
                restarting=False,
                context={"source": "broker_startup", "reason": "broker_started"},
            )

    def _launch_locked(self, *, restarting: bool, context: dict[str, Any]) -> None:
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
        self._process_started_at = _timestamp()
        self._launch_context = dict(context)
        lifecycle = {
            "old_pid": context.get("old_pid"),
            "new_pid": process.pid,
            "process_started_at": self._process_started_at,
            "reason": context.get("reason"),
            "return_code": context.get("return_code"),
            "signal": context.get("signal"),
            "request_id": context.get("request_id"),
            "session_id": context.get("session_id"),
            "message_id": context.get("message_id"),
            "affected_requests": context.get("affected_requests", []),
            "source": context.get("source"),
        }
        self._record_lifecycle("started", **lifecycle)
        self.on_message(
            {
                "kind": "supervisor",
                "type": "runtime.restarting" if restarting else "runtime.starting",
                "payload": {"pid": process.pid, **lifecycle},
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
                with self._lock:
                    if self._process is process:
                        context = dict(self._launch_context)
                        started_at = self._process_started_at
                    else:
                        context = {}
                        started_at = None
                self._record_lifecycle(
                    "ready",
                    old_pid=context.get("old_pid"),
                    new_pid=process.pid,
                    process_started_at=started_at,
                    reason=context.get("reason"),
                    return_code=context.get("return_code"),
                    signal=context.get("signal"),
                    request_id=context.get("request_id"),
                    session_id=context.get("session_id"),
                    message_id=context.get("message_id"),
                    affected_requests=context.get("affected_requests", []),
                    source=context.get("source"),
                )
            request_id = str(message.get("request_id") or "")
            if request_id:
                with self._lock:
                    if message.get("kind") == "event":
                        self._active_request_id = request_id
                    elif message.get("kind") in {"result", "error", "protocol_error"}:
                        self._requests.pop(request_id, None)
                        if self._active_request_id == request_id:
                            self._active_request_id = None
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
            old_pid = process.pid
            started_at = self._process_started_at
            planned = self._planned_restart
            active = dict(self._requests.get(self._active_request_id or "") or {})
            affected = [dict(item) for item in self._requests.values()]
            self._process = None
            self._process_started_at = None
            self._launch_context = {}
            self._planned_restart = None
            self._requests.clear()
            self._active_request_id = None
            if self._stopping:
                return
            if planned is None:
                self._consecutive_failures += 1
                failure = self._consecutive_failures
            else:
                failure = 0
        signal = -returncode if returncode < 0 else None
        context = dict(planned or {})
        context.update(
            {
                "old_pid": old_pid,
                "process_started_at": started_at,
                "return_code": returncode,
                "signal": signal,
                "source": context.get("source") or "crash_recovery",
                "reason": context.get("reason") or "unexpected_worker_exit",
                "request_id": active.get("request_id") or context.get("request_id"),
                "session_id": active.get("session_id") or context.get("session_id"),
                "message_id": active.get("message_id") or context.get("message_id"),
                "affected_requests": affected,
            }
        )
        self._record_lifecycle("exited", new_pid=None, **context)
        self.on_message(
            {
                "kind": "supervisor",
                "type": "runtime.exited",
                "payload": {
                    "returncode": returncode,
                    "restart_attempt": failure,
                    **context,
                },
            }
        )
        if planned is None and failure > self.restart_limit:
            self.on_message(
                {
                    "kind": "supervisor",
                    "type": "runtime.unavailable",
                    "payload": {"restart_limit": self.restart_limit},
                }
            )
            return
        if planned is None:
            time.sleep(min(0.25 * (2 ** (failure - 1)), 4.0))
        with self._lock:
            if not self._stopping and self._process is None:
                self._launch_locked(restarting=True, context=context)

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
                if message.get("kind") == "ask":
                    request_id = str(message.get("request_id") or "")
                    if request_id:
                        self._requests[request_id] = {
                            "request_id": request_id,
                            "session_id": str(message.get("session_id") or "") or None,
                            "message_id": message.get("message_id"),
                        }

    def restart(
        self,
        *,
        source: str = "broker_requested_restart",
        reason: str = "coordinated_restart",
        request_id: str | None = None,
        session_id: str | None = None,
        message_id: int | None = None,
    ) -> bool:
        if source not in _COORDINATED_RESTART_SOURCES:
            raise ValueError(f"Unsupported whole-runtime restart source: {source}")
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return False
            if self._planned_restart is not None:
                return False
            self._planned_restart = {
                "source": source,
                "reason": str(reason),
                "request_id": request_id,
                "session_id": session_id,
                "message_id": message_id,
            }
        self._record_lifecycle(
            "restart_requested",
            old_pid=process.pid,
            new_pid=None,
            process_started_at=self._process_started_at,
            reason=str(reason),
            return_code=None,
            signal=None,
            request_id=request_id,
            session_id=session_id,
            message_id=message_id,
            affected_requests=[dict(item) for item in self._requests.values()],
            source=source,
        )
        process.terminate()
        return True

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        old_pid = process.pid
        started_at = self._process_started_at
        process.terminate()
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=5)
        self._record_lifecycle(
            "stopped",
            old_pid=old_pid,
            new_pid=None,
            process_started_at=started_at,
            reason="broker_stopped",
            return_code=returncode,
            signal=-returncode if returncode < 0 else None,
            request_id=None,
            session_id=None,
            message_id=None,
            affected_requests=[],
            source="broker_shutdown",
        )
