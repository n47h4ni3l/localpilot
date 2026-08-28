from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO

from localpilot.audit import AuditLog
from localpilot.config import Config, load_config
from localpilot.selfdev import SelfDeveloper


DEFAULT_INTERVAL_SECONDS = 30.0
_LOCK_FILENAME = "background-worker.lock"
_PID_FILENAME = "background-worker.pid"
_STOP_FILENAME = "background-worker.stop"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


class WorkerLock:
    """Cross-process worker lock whose OS lock survives stale PID files safely."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.pid_path = self.path.with_suffix(".pid")
        self._handle: IO[bytes] | None = None
        self.owner: dict[str, Any] = {}
        self.stale_owner: dict[str, Any] = {}

    @staticmethod
    def _try_lock(handle: IO[bytes]) -> bool:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    @staticmethod
    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self, *, root: Path) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        if os.fstat(descriptor).st_size == 0:
            handle.write(b"0")
            handle.flush()
        if not self._try_lock(handle):
            handle.close()
            self.owner = _read_json(self.pid_path)
            return False

        self.stale_owner = _read_json(self.pid_path)
        self.owner = {
            "pid": os.getpid(),
            "started_at": _utc_now(),
            "root": str(root),
        }
        temporary_pid_path = self.pid_path.with_suffix(".pid.tmp")
        temporary_pid_path.write_text(
            json.dumps(self.owner, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_pid_path, self.pid_path)
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            self.pid_path.write_text("{}\n", encoding="utf-8")
            self._unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> WorkerLock:
        if not self.acquire(root=self.path.parent):
            raise RuntimeError("LocalPilot background worker is already running")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class BackgroundWorker:
    """Run the established autonomous evolve cycle on one persistent process."""

    def __init__(
        self,
        root: str | Path,
        *,
        config_path: str | Path | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        config: Config | None = None,
        cycle_runner: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        waiter: Callable[[float], bool] | None = None,
    ) -> None:
        if float(interval_seconds) <= 0:
            raise ValueError("interval_seconds must be positive")
        self.root = Path(root).resolve()
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config = config or load_config(self.config_path)
        self.interval_seconds = float(interval_seconds)
        self.data_dir = (self.root / self.config.agent.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.data_dir / "audit.jsonl")
        self.lock = WorkerLock(self.data_dir / _LOCK_FILENAME)
        self.stop_path = self.data_dir / _STOP_FILENAME
        self.stop_event = threading.Event()
        self.stop_reason = "requested"
        self._cycle_runner = cycle_runner or self._run_evolve_cycle
        self._monotonic = monotonic
        self._waiter = waiter or self.stop_event.wait
        self._previous_signal_handlers: dict[int, Any] = {}

    def _run_evolve_cycle(self) -> Any:
        # This is deliberately the same unforced entry point used by the old
        # scheduled task. SelfDeveloper retains every existing gate and audit.
        return SelfDeveloper(self.config, self.root).run_once(force=False)

    def request_stop(self, reason: str) -> None:
        self.stop_reason = reason
        self.stop_event.set()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(signum)
                signal.signal(
                    signum,
                    lambda _signum, _frame, name=signal.Signals(signum).name: self.request_stop(
                        f"signal:{name}"
                    ),
                )
            except (OSError, ValueError):
                continue
            self._previous_signal_handlers[signum] = previous

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                pass
        self._previous_signal_handlers.clear()

    def _consume_stop_request(self) -> bool:
        request = _read_json(self.stop_path)
        if not request:
            return False
        target_pid = request.get("target_pid")
        if target_pid != os.getpid():
            try:
                self.stop_path.unlink()
            except FileNotFoundError:
                pass
            return False
        try:
            self.stop_path.unlink()
        except FileNotFoundError:
            pass
        self.request_stop("stop_request")
        return True

    def _wait_until(self, deadline: float) -> bool:
        while not self.stop_event.is_set():
            self._consume_stop_request()
            if self.stop_event.is_set():
                break
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            if self._waiter(min(remaining, 0.25)):
                return True
        return True

    def run(self, *, max_cycles: int | None = None) -> int:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        if not self.lock.acquire(root=self.root):
            self.audit.write(
                "background_worker_duplicate",
                pid=os.getpid(),
                owner_pid=self.lock.owner.get("pid"),
                lock_path=str(self.lock.path),
            )
            return 0

        stale_pid = self.lock.stale_owner.get("pid")
        if stale_pid == os.getpid():
            stale_pid = None
        self._install_signal_handlers()
        cycles = 0
        try:
            self._consume_stop_request()
            self.audit.write(
                "background_worker_start",
                pid=os.getpid(),
                interval_seconds=self.interval_seconds,
                lock_path=str(self.lock.path),
                recovered_stale_pid=stale_pid,
            )
            next_deadline = self._monotonic()
            while not self.stop_event.is_set():
                if self._wait_until(next_deadline):
                    break
                cycles += 1
                cycle_started = self._monotonic()
                self.audit.write(
                    "background_worker_cycle_start",
                    pid=os.getpid(),
                    sequence=cycles,
                    interval_seconds=self.interval_seconds,
                )
                try:
                    result = self._cycle_runner()
                except Exception as exc:
                    self.audit.write(
                        "background_worker_cycle_error",
                        pid=os.getpid(),
                        sequence=cycles,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        duration_seconds=round(self._monotonic() - cycle_started, 3),
                    )
                else:
                    self.audit.write(
                        "background_worker_cycle_end",
                        pid=os.getpid(),
                        sequence=cycles,
                        status=str(getattr(result, "status", "completed")),
                        duration_seconds=round(self._monotonic() - cycle_started, 3),
                    )
                if max_cycles is not None and cycles >= max_cycles:
                    self.stop_reason = "max_cycles"
                    break
                self._consume_stop_request()
                next_deadline += self.interval_seconds
                now = self._monotonic()
                if next_deadline < now:
                    skipped = int((now - next_deadline) // self.interval_seconds) + 1
                    next_deadline += skipped * self.interval_seconds
                    self.audit.write(
                        "background_worker_cadence_overrun",
                        pid=os.getpid(),
                        sequence=cycles,
                        skipped_intervals=skipped,
                    )
            return 0
        finally:
            self.audit.write(
                "background_worker_stop",
                pid=os.getpid(),
                reason=self.stop_reason,
                completed_cycles=cycles,
            )
            try:
                request = _read_json(self.stop_path)
                if request.get("target_pid") == os.getpid():
                    self.stop_path.unlink(missing_ok=True)
            finally:
                self._restore_signal_handlers()
                self.lock.release()


def request_worker_stop(root: str | Path, config_path: str | Path | None = None) -> int | None:
    root_path = Path(root).resolve()
    config = load_config(config_path)
    data_dir = (root_path / config.agent.data_dir).resolve()
    owner = _read_json(data_dir / _PID_FILENAME)
    target_pid = owner.get("pid")
    audit = AuditLog(data_dir / "audit.jsonl")
    if not isinstance(target_pid, int) or target_pid <= 0:
        audit.write("background_worker_stop_not_running")
        return None
    stop_path = data_dir / _STOP_FILENAME
    stop_path.write_text(
        json.dumps({"target_pid": target_pid, "requested_at": _utc_now()}) + "\n",
        encoding="utf-8",
    )
    audit.write("background_worker_stop_requested", target_pid=target_pid)
    return target_pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-background-worker")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--stop", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stop:
        request_worker_stop(args.root, args.config)
        return
    worker: BackgroundWorker | None = None
    try:
        worker = BackgroundWorker(
            args.root,
            config_path=args.config,
            interval_seconds=args.interval_seconds,
        )
        raise SystemExit(worker.run())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            audit = (
                worker.audit
                if worker is not None
                else AuditLog(Path(args.root).resolve() / "localpilot-data" / "audit.jsonl")
            )
            audit.write(
                "background_worker_crash",
                pid=os.getpid(),
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
