import io
import subprocess

import pytest

from localpilot.runtime_supervisor import RuntimeSupervisor


class SlowWorker:
    pid = 321

    def __init__(self, *, stdin=True):
        self.stdin = io.StringIO() if stdin else None
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode

    def terminate(self):
        raise AssertionError("Do not terminate a worker during cooperative shutdown")

    def kill(self):
        raise AssertionError("Do not kill a worker during cooperative shutdown")


@pytest.mark.parametrize("stdin", [True, False])
def test_stop_timeout_preserves_worker_and_blocks_replacement(tmp_path, monkeypatch, stdin):
    supervisor = RuntimeSupervisor(tmp_path)
    worker = SlowWorker(stdin=stdin)
    supervisor._process = worker
    supervisor._process_started_at = "original-start"
    launches = []
    monkeypatch.setattr(supervisor, "_launch_locked", lambda **kwargs: launches.append(kwargs))

    with pytest.raises(RuntimeError, match="321.*still stopping"):
        supervisor.stop()

    assert supervisor._process is worker
    assert supervisor._process_started_at == "original-start"
    assert supervisor.pid == 321
    if stdin:
        assert worker.stdin.closed
    supervisor.start()
    with pytest.raises(RuntimeError, match="runtime is stopping"):
        supervisor.send({"kind": "ask", "prompt": "must not run"})
    assert supervisor.restart(source="update") is False
    assert launches == []
    rows = supervisor.audit.recent("runtime_lifecycle", limit=10)
    assert [row["transition"] for row in rows] == ["stop_timeout"]
    assert rows[0]["old_pid"] == 321


def test_timed_out_worker_can_exit_later_without_automatic_restart(tmp_path, monkeypatch):
    supervisor = RuntimeSupervisor(tmp_path)
    worker = SlowWorker()
    supervisor._process = worker
    launches = []
    monkeypatch.setattr(supervisor, "_launch_locked", lambda **kwargs: launches.append(kwargs))

    with pytest.raises(RuntimeError, match="still stopping"):
        supervisor.stop()
    worker.returncode = 0
    supervisor._watch(worker)

    assert supervisor._process is None
    assert supervisor.pid is None
    assert launches == []
    supervisor.start()
    assert len(launches) == 1
    assert supervisor._stopping is False
