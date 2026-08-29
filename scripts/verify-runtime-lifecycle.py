from __future__ import annotations

import json
import time
from pathlib import Path

from localpilot.broker import BrokerApp
from localpilot.config import Config


class _NoopTimer:
    def cancel(self) -> None:
        return


def _wait_for(predicate, *, seconds: float = 15.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("live runtime validation timed out")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config()
    config.agent.data_dir = "localpilot-data-live-validation"
    config.systemsense.enabled = False
    config.library.enabled = False
    app = BrokerApp(root, config)
    app.start()
    try:
        initial_pid = _wait_for(lambda: app.runtime.pid)
        _wait_for(
            lambda: any(
                event["type"] == "runtime.ready"
                for event in app.store.events_after(0)
            )
        )

        session = app.store.create_session("Live lifecycle verification")
        assistant = app.store.add_message(
            session["id"], "assistant", "", status="streaming"
        )
        request_id = "live-soft-timeout"
        with app._lock:
            app._pending[request_id] = {
                "session_id": session["id"],
                "message_id": assistant["id"],
                "content": "",
                "timer": _NoopTimer(),
            }
        app._expire_request(request_id)
        timeout_pid = app.runtime.pid
        timeout_message = app.store.message(assistant["id"])

        with app._lock:
            pending = app._pending.pop(request_id)
        pending["timer"].cancel()
        app.store.update_message(assistant["id"], "live timeout proof complete", status="complete")

        process = app.runtime._process
        if process is None:
            raise RuntimeError("runtime process disappeared before crash verification")
        process.kill()
        replacement_pid = _wait_for(
            lambda: app.runtime.pid if app.runtime.pid not in {None, initial_pid} else None
        )
        lifecycle = app.runtime.audit.recent("runtime_lifecycle", limit=12)
        crash_exit = next(
            row
            for row in lifecycle
            if row.get("transition") == "exited"
            and row.get("old_pid") == initial_pid
            and row.get("source") == "crash_recovery"
        )
        crash_start = next(
            row
            for row in lifecycle
            if row.get("transition") == "started"
            and row.get("new_pid") == replacement_pid
            and row.get("source") == "crash_recovery"
        )
        print(
            json.dumps(
                {
                    "soft_timeout": {
                        "pid_before": initial_pid,
                        "pid_after": timeout_pid,
                        "pid_stable": initial_pid == timeout_pid,
                        "message_status": timeout_message["status"],
                    },
                    "crash_recovery": {
                        "old_pid": initial_pid,
                        "new_pid": replacement_pid,
                        "pid_replaced": initial_pid != replacement_pid,
                        "exit_reason": crash_exit.get("reason"),
                        "exit_source": crash_exit.get("source"),
                        "return_code": crash_exit.get("return_code"),
                        "replacement_source": crash_start.get("source"),
                    },
                },
                indent=2,
            )
        )
    finally:
        app.stop()


if __name__ == "__main__":
    main()
