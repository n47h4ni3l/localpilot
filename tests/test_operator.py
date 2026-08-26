import sys
import unittest
from types import SimpleNamespace

import pytest

from localpilot.operator import CommandRunner, CommandSpec, OperationRisk


class TestOperator(unittest.TestCase):
    def test_shell_metacharacters(self):
        runner = CommandRunner()
        spec = CommandSpec(
            argv=[sys.executable, '-c', 'import sys; print("|".join(sys.argv[1:]))', '&&', 'echo', 'second'],
            risk=OperationRisk.READ_ONLY,
            timeout=5,
        )
        result = runner.run(spec)
        self.assertIn('&&', result['stdout'])
        self.assertIn('second', result['stdout'])

    def test_timeout_returns_structured_result(self):
        runner = CommandRunner()
        spec = CommandSpec(
            argv=[sys.executable, '-c', 'import time; time.sleep(2)'],
            risk=OperationRisk.READ_ONLY,
            timeout=0.05,
        )
        result = runner.run(spec)
        self.assertEqual('', result['stderr'])
        self.assertIsNone(result['returncode'])

    def test_destructive_operation_denied(self):
        runner = CommandRunner()
        spec = CommandSpec(argv=['rm', '-rf', '/'], risk=OperationRisk.DESTRUCTIVE, timeout=5)
        with self.assertRaises(RuntimeError):
            runner.run(spec)

    def test_harmless_approved_command(self):
        def approval_callback():
            return True
        runner = CommandRunner(approval_callback)
        spec = CommandSpec(
            argv=[sys.executable, '-c', 'print("approved")'],
            risk=OperationRisk.DESTRUCTIVE,
            timeout=5,
        )
        result = runner.run(spec)
        self.assertIn('approved', result['stdout'])


def test_nonblocking_action_uses_argv_shell_false_and_audits(monkeypatch):
    calls = []
    events = []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr("localpilot.operator.subprocess.Popen", fake_popen)
    runner = CommandRunner(audit_callback=events.append)

    result = runner.run(
        CommandSpec(
            argv=["notepad.exe"],
            risk=OperationRisk.REVERSIBLE,
            timeout=10,
            action="open_windows_app:notepad",
            wait=False,
        )
    )

    assert result["status"] == "started"
    assert result["process_id"] == 4321
    assert calls[0][0] == ["notepad.exe"]
    assert calls[0][1]["shell"] is False
    assert events[0]["action"] == "open_windows_app:notepad"
    assert events[0]["risk"] == "reversible"
    assert events[0]["status"] == "started"
    assert "argv" not in events[0]


def test_destructive_approval_receives_exact_spec_and_denial_is_audited():
    seen = []
    events = []
    spec = CommandSpec(
        argv=[sys.executable, "-c", "raise SystemExit(99)"],
        risk=OperationRisk.DESTRUCTIVE,
        timeout=5,
        action="blocked-test",
    )
    runner = CommandRunner(
        approval_callback=lambda item: seen.append(item) or False,
        audit_callback=events.append,
    )

    with pytest.raises(RuntimeError, match="not approved"):
        runner.run(spec)

    assert seen == [spec]
    assert events[0]["status"] == "denied"


@pytest.mark.parametrize(
    "argv,timeout",
    [([], 5), ([""], 5), (["echo"], 0), (["echo"], 61)],
)
def test_invalid_command_specs_fail_before_process_creation(argv, timeout):
    with pytest.raises(ValueError):
        CommandRunner().run(
            CommandSpec(
                argv=argv,
                risk=OperationRisk.REVERSIBLE,
                timeout=timeout,
            )
        )
