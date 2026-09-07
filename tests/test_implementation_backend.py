from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.implementation_backend import (
    ClaudeCodeBackend,
    ImplementationPreflight,
    ImplementationRequest,
    ImplementationResult,
    ImplementationStatus,
    parse_claude_json_output,
)
from localpilot.selfdev import CandidateTools, SelfDeveloper


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Backend Test")
    _git(root, "config", "user.email", "backend-test@invalid.local")
    _git(root, "add", "module.py")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root


def _fake_cli(tmp_path: Path, behavior: str) -> tuple[str, ...]:
    script = tmp_path / f"fake_claude_{behavior}.py"
    script.write_text(
        """
import json, os, pathlib, sys, time
if '--version' in sys.argv:
    print('2.1.263')
    raise SystemExit(0)
if '--help' in sys.argv:
    print('--allowedTools --disallowedTools --max-turns --model --output-format')
    raise SystemExit(0)
behavior = sys.argv[1]
prompt = sys.stdin.read()
if behavior == 'timeout':
    time.sleep(2)
elif behavior == 'cli_error':
    print('failed', file=sys.stderr)
    raise SystemExit(7)
elif behavior == 'malformed':
    print('not json')
elif behavior in {'success', 'nonzero_success', 'plain_success', 'null_success', 'max_turns', 'escape'}:
    target = pathlib.Path('module.py' if behavior != 'escape' else 'other.py')
    target.write_text('VALUE = 2\\n', encoding='utf-8')
    result = {
        'summary': 'implemented and tested',
        'tests': [{'command': 'python -m pytest -q', 'passed': True, 'exit_code': 0,
                   'output_digest': 'a' * 64}],
    }
    envelope_result = (
        result if behavior == 'nonzero_success'
        else None if behavior == 'null_success'
        else 'implementation finished' if behavior in {'plain_success', 'max_turns'}
        else json.dumps(result)
    )
    envelope_key = 'structured_output' if behavior == 'nonzero_success' else 'result'
    print(json.dumps({envelope_key: envelope_result, 'session_id': 'session-1',
                      'usage': {'input_tokens': 10, 'output_tokens': 5},
                      'subtype': 'error_max_turns' if behavior == 'max_turns' else 'success',
                      'is_error': behavior in {'null_success', 'max_turns'},
                      'stop_reason': 'stop_sequence'}))
    if behavior in {'nonzero_success', 'plain_success', 'null_success', 'max_turns'}:
        raise SystemExit(1)
""".lstrip(),
        encoding="utf-8",
    )
    return (sys.executable, str(script), behavior)


def _backend(tmp_path: Path, behavior: str, **kwargs) -> ClaudeCodeBackend:
    backend = ClaudeCodeBackend(
        executable_argv=_fake_cli(tmp_path, behavior), timeout_seconds=kwargs.get("timeout", 5),
        max_output_chars=kwargs.get("max_output", 120_000),
    )
    backend.preflight = lambda: ImplementationPreflight(
        True, "claude_code", "gpt-oss:20b", sys.executable,
        version="2.1.263", context_tokens=65536,
    )
    backend._cancel_ollama_inference = lambda: None
    return backend


def test_json_parser_handles_claude_envelope_and_usage():
    raw = json.dumps(
        {
            "result": json.dumps({"summary": "done", "tests": []}),
            "session_id": "abc",
            "usage": {"input_tokens": 4},
        }
    )
    parsed = parse_claude_json_output(raw)
    assert parsed["summary"] == "done"
    assert parsed["_claude_session_id"] == "abc"
    assert parsed["_claude_usage"] == {"input_tokens": 4}


@pytest.mark.parametrize(("allocated", "healthy"), [(65536, True), (32768, False)])
def test_preflight_loads_model_at_target_context_and_verifies_allocation(
    monkeypatch, allocated: int, healthy: bool
):
    backend = ClaudeCodeBackend(executable_argv=(sys.executable,))
    flags = " ".join(backend._REQUIRED_FLAGS)
    backend._run_probe = lambda *args: subprocess.CompletedProcess(
        args, 0, stdout="2.1.263" if args == ("--version",) else flags, stderr=""
    )
    monkeypatch.setattr(
        "localpilot.implementation_backend.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="installed", stderr=""),
    )
    calls: list[tuple[str, dict | None]] = []
    trims: list[bool] = []
    monkeypatch.setattr(
        "localpilot.implementation_backend._trim_windows_gpu_runner_working_sets",
        lambda: trims.append(True) or 1,
    )

    def ollama_json(path, payload=None):
        calls.append((path, payload))
        if path == "/api/ps":
            return {
                "models": [{
                    "name": "gpt-oss:20b", "context_length": allocated,
                    "size": 100, "size_vram": 100,
                }]
            }
        return {"done": True}

    backend._ollama_json = ollama_json
    result = backend.preflight()
    assert result.healthy is healthy
    assert calls[0][0] == "/api/generate"
    assert calls[0][1]["options"]["num_ctx"] == 65536
    assert calls[1][0] == "/api/ps"
    assert trims == ([True] if healthy else [])
    if not healthy:
        assert "allocated 32768" in "; ".join(result.messages)


def test_real_process_wrapper_edits_only_candidate_and_returns_structured_evidence(tmp_path: Path):
    root = _repo(tmp_path)
    result = _backend(tmp_path, "success").run(
        ImplementationRequest(root, "implement", ("module.py",))
    )
    assert result.status == ImplementationStatus.COMPLETED
    assert result.changed_paths == ("module.py",)
    assert len(result.diff_digest) == 64
    assert result.session_id == "session-1"
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}
    assert result.tests[0]["passed"] is True


def test_explicit_success_envelope_retains_nonzero_exit_as_evidence(tmp_path: Path):
    root = _repo(tmp_path)
    result = _backend(tmp_path, "nonzero_success").run(
        ImplementationRequest(root, "implement", ("module.py",))
    )
    assert result.status == ImplementationStatus.COMPLETED
    assert result.exit_code == 1


@pytest.mark.parametrize("behavior", ["plain_success", "null_success", "max_turns"])
def test_localpilot_test_verification_recovers_bounded_custom_model_envelope(
    tmp_path: Path, behavior: str
):
    root = _repo(tmp_path)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_module.py").write_text(
        "import unittest\nfrom module import VALUE\n\n"
        "class ModuleTests(unittest.TestCase):\n"
        "    def test_value(self):\n        self.assertEqual(VALUE, 2)\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_module.py")
    _git(root, "commit", "--quiet", "-m", "test fixture")
    result = _backend(tmp_path, behavior).run(
        ImplementationRequest(
            root,
            "implement",
            ("module.py",),
            test_commands=((
                sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"
            ),),
        )
    )
    assert result.status == ImplementationStatus.COMPLETED
    assert result.exit_code == 1
    assert result.tests[0]["passed"] is True
    assert len(result.tests[0]["output_digest"]) == 64


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("cli_error", ImplementationStatus.CLI_ERROR),
        ("malformed", ImplementationStatus.MALFORMED_OUTPUT),
        ("escape", ImplementationStatus.CONFINEMENT_VIOLATION),
    ],
)
def test_wrapper_distinguishes_failure_classes(tmp_path: Path, behavior: str, expected: ImplementationStatus):
    root = _repo(tmp_path)
    result = _backend(tmp_path, behavior).run(
        ImplementationRequest(root, "implement", ("module.py",))
    )
    assert result.status == expected
    if behavior == "cli_error":
        assert "stderr=failed" in result.summary


def test_wrapper_times_out_and_never_uses_shell(tmp_path: Path, monkeypatch):
    root = _repo(tmp_path)
    observed: dict[str, object] = {}
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        observed["shell"] = kwargs.get("shell")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr("localpilot.implementation_backend.subprocess.Popen", recording_popen)
    backend = _backend(tmp_path, "timeout", timeout=0.2)
    cancellations: list[bool] = []
    backend._cancel_ollama_inference = lambda: cancellations.append(True)
    result = backend.run(
        ImplementationRequest(root, "implement", ("module.py",))
    )
    assert result.status == ImplementationStatus.TIMEOUT
    assert observed["shell"] is False
    assert cancellations == [True]


def test_command_has_narrow_tools_and_no_permission_bypass(tmp_path: Path):
    command = _backend(tmp_path, "success")._command(allowed_paths=("module.py",))
    joined = " ".join(command)
    assert "--dangerously-skip-permissions" not in joined
    assert "--allowedTools" in command
    assert "--disallowedTools" in command
    assert "Bash(git commit *)" in command
    assert "WebSearch" in command
    assert "--no-session-persistence" in command
    assert "Edit" not in command
    assert "Write" not in command
    assert "Edit(./module.py)" in command
    assert "Write(./module.py)" in command
    env = _backend(tmp_path, "success")._environment()
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"
    assert env["MAX_THINKING_TOKENS"] == "0"
    assert env["CLAUDE_CODE_DISABLE_THINKING"] == "1"
    assert env["CLAUDE_CODE_DISABLE_TERMINAL_TITLE"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    if sys.platform == "win32":
        assert env["SYSTEMDRIVE"]
        assert env["PROGRAMDATA"]


def test_localpilot_rejection_drives_one_bounded_claude_rework_pass(tmp_path: Path, monkeypatch):
    root = _repo(tmp_path)
    config = Config()
    config.agent.data_dir = "data"

    class RepairingBackend:
        name = "claude_code"

        def __init__(self) -> None:
            self.calls = 0

        def preflight(self):
            return ImplementationPreflight(
                True, self.name, "gpt-oss:20b", "fake-test-double",
                version="test", context_tokens=65536,
            )

        def run(self, request, *, repair_pass=0):
            self.calls += 1
            (request.workspace / "module.py").write_text(
                f"VALUE = {1 + self.calls}\n", encoding="utf-8"
            )
            passed = self.calls > 1
            return ImplementationResult(
                ImplementationStatus.COMPLETED if passed else ImplementationStatus.CANDIDATE_TEST_FAILURE,
                self.name, "gpt-oss:20b", "repair result", changed_paths=("module.py",),
                diff_digest=str(self.calls) * 64,
                tests=({"command": "python -m pytest", "passed": passed, "exit_code": 0 if passed else 1,
                        "output_digest": "b" * 64},),
                session_id="same-session",
            )

    backend = RepairingBackend()
    developer = SelfDeveloper(config, root, implementation_backend=backend)
    responses = iter(
        [
            {"content": json.dumps({"approved": False, "feedback": ["fix regression"], "summary": "reject"})},
            {"content": json.dumps({"approved": True, "feedback": [], "summary": "accept"})},
        ]
    )
    review_calls: list[dict] = []

    def review_chat(*args, **kwargs):
        review_calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(developer, "_developer_chat", review_chat)
    tools = CandidateTools(root)
    task = {
        "id": "fixture", "title": "repair fixture", "acceptance": ["tests pass"],
        "hypothesis": "repair works",
        "evaluation": {"metric": "tests", "baseline": "failing", "measurement_method": "pytest"},
    }
    result = developer._run_claude_code_implementation(
        chat=lambda **kwargs: None, developer_model="gpt-oss:20b", task=task,
        branch="candidate/test", workspace=root, tools=tools, cycle_id=1,
        research="evidence", grounding_plan={"referenced_paths": ["module.py"], "new_runtime_paths": []},
        grounding_evidence=["module.py"], evolution_context="bounded", lessons=[], force=True,
    )
    assert backend.calls == 2
    assert len(review_calls) == 2
    assert review_calls[0]["format"]["required"] == ["approved", "feedback", "summary"]
    assert review_calls[0]["options"]["num_predict"] == 1024
    assert "LocalPilot review approved" in result
    evidence = [
        json.loads(line) for line in (root / "data" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if '"event": "selfdev_implementation_backend"' in line
    ]
    assert evidence[-1]["review_repair_pass_count"] == 1
    assert evidence[-1]["final_status"] == "completed"
