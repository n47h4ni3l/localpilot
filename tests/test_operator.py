import sys
import unittest
from localpilot.operator import CommandRunner, CommandSpec, OperationRisk


class TestOperator(unittest.TestCase):
    def test_shell_metacharacters(self):
        runner = CommandRunner()
        spec = CommandSpec(argv=['echo', '&&', 'echo', 'second'], risk=OperationRisk.READ_ONLY, timeout=5)
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

    def test_destructive_operation_approved(self):
        def approval_callback():
            return True
        runner = CommandRunner(approval_callback)
        spec = CommandSpec(argv=['echo', 'approved'], risk=OperationRisk.DESTRUCTIVE, timeout=5)
        result = runner.run(spec)
        self.assertIn('approved', result['stdout'])


if __name__ == '__main__':
    unittest.main()
