from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from localpilot.background_worker import BackgroundWorker, WorkerLock, request_worker_stop
from localpilot.config import Config


def _audit_rows(root: Path) -> list[dict[str, object]]:
    path = root / "localpilot-data" / "audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_worker_calls_the_existing_unforced_selfdev_entry_point(tmp_path, monkeypatch):
    observed: dict[str, object] = {}

    class FakeDeveloper:
        def __init__(self, config, root):
            observed["config"] = config
            observed["root"] = root

        def run_once(self, *, force):
            observed["force"] = force
            return SimpleNamespace(status="deferred")

    monkeypatch.setattr("localpilot.background_worker.SelfDeveloper", FakeDeveloper)
    config = Config()
    worker = BackgroundWorker(tmp_path, config=config)

    result = worker._run_evolve_cycle()

    assert result.status == "deferred"
    assert observed == {"config": config, "root": tmp_path.resolve(), "force": False}


def test_worker_runs_on_a_fixed_non_overlapping_cadence(tmp_path):
    starts: list[float] = []
    now = 0.0

    def monotonic():
        return now

    def waiter(seconds):
        nonlocal now
        now += seconds
        return False

    def cycle():
        starts.append(monotonic())
        return SimpleNamespace(status="deferred")

    worker = BackgroundWorker(
        tmp_path,
        config=Config(),
        interval_seconds=30,
        cycle_runner=cycle,
        monotonic=monotonic,
        waiter=waiter,
    )
    assert worker.run(max_cycles=3) == 0

    assert len(starts) == 3
    assert starts == [0.0, 30.0, 60.0]
    rows = _audit_rows(tmp_path)
    assert [row["sequence"] for row in rows if row["event"] == "background_worker_cycle_start"] == [1, 2, 3]


def test_duplicate_worker_is_rejected_by_the_os_lock(tmp_path):
    lock_path = tmp_path / "worker.lock"
    first = WorkerLock(lock_path)
    second = WorkerLock(lock_path)
    assert first.acquire(root=tmp_path)
    try:
        assert not second.acquire(root=tmp_path)
        assert second.owner["pid"] == os.getpid()
    finally:
        first.release()


def test_stale_pid_is_recovered_after_process_crash(tmp_path):
    lock_path = tmp_path / "worker.lock"
    code = (
        "import os; from pathlib import Path; "
        "from localpilot.background_worker import WorkerLock; "
        f"lock=WorkerLock(Path({str(lock_path)!r})); "
        f"assert lock.acquire(root=Path({str(tmp_path)!r})); "
        "os._exit(23)"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 23
    stale = json.loads(lock_path.with_suffix(".pid").read_text(encoding="utf-8"))

    recovered = WorkerLock(lock_path)
    assert recovered.acquire(root=tmp_path)
    try:
        assert recovered.stale_owner["pid"] == stale["pid"]
        assert recovered.owner["pid"] == os.getpid()
    finally:
        recovered.release()


def test_stop_request_waits_for_cycle_boundary_and_cleans_up(tmp_path):
    cycle_started = threading.Event()
    release_cycle = threading.Event()

    def cycle():
        cycle_started.set()
        assert release_cycle.wait(timeout=2)
        return SimpleNamespace(status="deferred")

    worker = BackgroundWorker(
        tmp_path,
        config=Config(),
        interval_seconds=30,
        cycle_runner=cycle,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert cycle_started.wait(timeout=2)

    assert request_worker_stop(tmp_path) == os.getpid()
    assert thread.is_alive()
    release_cycle.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert not worker.stop_path.exists()
    assert json.loads(worker.lock.pid_path.read_text(encoding="utf-8")) == {}
    rows = _audit_rows(tmp_path)
    assert rows[-1]["event"] == "background_worker_stop"
    assert rows[-1]["reason"] == "stop_request"


def test_cycle_failure_is_logged_and_next_cycle_still_runs(tmp_path):
    attempts = 0

    def cycle():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("test failure")
        return SimpleNamespace(status="deferred")

    worker = BackgroundWorker(
        tmp_path,
        config=Config(),
        interval_seconds=0.01,
        cycle_runner=cycle,
    )
    assert worker.run(max_cycles=2) == 0

    rows = _audit_rows(tmp_path)
    assert attempts == 2
    assert any(row["event"] == "background_worker_cycle_error" for row in rows)
    assert any(row["event"] == "background_worker_cycle_end" for row in rows)
