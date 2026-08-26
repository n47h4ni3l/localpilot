from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from ollama import Client

from localpilot.agent import LocalPilotAgent
from localpilot.config import Config
from localpilot.operator import CommandRunner, OperationRisk
from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.windows_actions import WindowsActions


def _chunk(*, content="", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking="",
            tool_calls=list(tool_calls or []),
        )
    )


def _call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments)
    )


class RecordingRunner:
    def __init__(self):
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        return {
            "action": spec.action,
            "risk": spec.risk.value,
            "status": "started",
            "process_id": 123,
        }


class StatefulPowerRunner:
    balanced = "381b4222-f694-41f0-9685-ff5bb260df2e"
    high = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    saver = "a1841308-3541-4fab-bc81-f71556f20b4a"

    def __init__(self, *, available=None, ignore_changes=False):
        self.active = self.balanced
        self.available = set(available or {self.balanced, self.high, self.saver})
        self.ignore_changes = ignore_changes
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        arguments = spec.argv[1:]
        stdout = ""
        if arguments == ["/GETACTIVESCHEME"]:
            stdout = f"Power Scheme GUID: {self.active}  (active)"
        elif arguments == ["/LIST"]:
            stdout = "\n".join(
                f"Power Scheme GUID: {guid}" for guid in sorted(self.available)
            )
        elif arguments[:1] == ["/SETACTIVE"]:
            if arguments[1] not in self.available:
                return {"status": "failed", "returncode": 1, "stdout": ""}
            if not self.ignore_changes:
                self.active = arguments[1]
        return {"status": "succeeded", "returncode": 0, "stdout": stdout}


def test_allowlisted_app_and_settings_actions_build_fixed_reversible_argv(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = RecordingRunner()
    actions = WindowsActions(runner)

    app = actions.open_windows_app("calculator")
    settings = actions.open_windows_settings("windows_update")

    assert app["status"] == settings["status"] == "started"
    assert runner.specs[0].argv == ["calc.exe"]
    assert runner.specs[1].argv == ["explorer.exe", "ms-settings:windowsupdate"]
    assert all(spec.risk is OperationRisk.REVERSIBLE for spec in runner.specs)
    assert all(spec.wait is False for spec in runner.specs)


@pytest.mark.parametrize(
    "method,value",
    [("open_windows_app", "powershell"), ("open_windows_settings", "registry")],
)
def test_action_arguments_outside_the_allowlist_are_rejected(monkeypatch, method, value):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    actions = WindowsActions(RecordingRunner())

    with pytest.raises(ValueError, match="Unsupported Windows"):
        getattr(actions, method)(value)


def test_non_windows_hosts_fail_before_command_execution(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "posix")
    runner = RecordingRunner()

    with pytest.raises(RuntimeError, match="only on Windows"):
        WindowsActions(runner).open_windows_app("notepad")
    assert runner.specs == []


def test_registry_exposes_only_the_complete_reversible_action_set(tmp_path):
    tools = registry(tmp_path, command_runner=CommandRunner())

    assert tools["open_windows_app"].risk is RiskLevel.REVERSIBLE
    assert tools["open_windows_settings"].risk is RiskLevel.REVERSIBLE
    assert tools["set_active_power_plan"].risk is RiskLevel.REVERSIBLE
    assert tools["restore_power_plan"].risk is RiskLevel.REVERSIBLE
    assert not {
        "terminate_process",
        "run_command",
    } & tools.keys()


def test_ollama_client_generates_schemas_for_registered_reversible_actions(
    tmp_path, monkeypatch
):
    request_json = {}

    def fake_request(_client, _response_type, *_args, **kwargs):
        request_json.update(kwargs["json"])
        return {}

    monkeypatch.setattr(Client, "_request", fake_request)
    registered = registry(tmp_path, command_runner=CommandRunner())
    reversible = [
        spec.fn for spec in registered.values() if spec.risk is RiskLevel.REVERSIBLE
    ]

    Client().chat(model="test-model", messages=[], tools=reversible)

    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in request_json["tools"]
    }
    assert set(schemas) == {
        "open_windows_app",
        "open_windows_settings",
        "set_active_power_plan",
        "restore_power_plan",
    }
    assert {
        name: set(schema["properties"])
        for name, schema in schemas.items()
    } == {
        "open_windows_app": {"app"},
        "open_windows_settings": {"page"},
        "set_active_power_plan": {"plan"},
        "restore_power_plan": {"rollback_token"},
    }


def test_power_plan_change_is_verified_and_has_one_use_exact_rollback(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = StatefulPowerRunner()
    actions = WindowsActions(runner)

    changed = actions.set_active_power_plan("high_performance")

    assert changed["status"] == "changed"
    assert changed["previous_guid"] == runner.balanced
    assert runner.active == runner.high
    token = changed["rollback_token"]
    restored = actions.restore_power_plan(token)
    assert restored["status"] == "restored"
    assert runner.active == runner.balanced
    with pytest.raises(ValueError, match="already-used"):
        actions.restore_power_plan(token)
    assert all(spec.argv[0] == "powercfg.exe" for spec in runner.specs)
    assert all(spec.wait is True for spec in runner.specs)


def test_power_plan_change_refuses_missing_target_before_mutation(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = StatefulPowerRunner(available={StatefulPowerRunner.balanced})

    with pytest.raises(RuntimeError, match="not installed"):
        WindowsActions(runner).set_active_power_plan("high_performance")

    assert runner.active == runner.balanced
    assert not any("/SETACTIVE" in spec.argv for spec in runner.specs)


def test_power_rollback_refuses_to_overwrite_an_independent_change(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = StatefulPowerRunner()
    actions = WindowsActions(runner)
    token = actions.set_active_power_plan("high_performance")["rollback_token"]
    runner.active = runner.saver

    with pytest.raises(RuntimeError, match="changed after"):
        actions.restore_power_plan(token)

    assert runner.active == runner.saver


def test_later_verified_change_invalidates_older_rollback_token(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = StatefulPowerRunner()
    actions = WindowsActions(runner)
    old_token = actions.set_active_power_plan("high_performance")["rollback_token"]
    current_token = actions.set_active_power_plan("power_saver")["rollback_token"]

    with pytest.raises(ValueError, match="Unknown"):
        actions.restore_power_plan(old_token)
    restored = actions.restore_power_plan(current_token)
    assert restored["status"] == "restored"
    assert runner.active == runner.high


def test_failed_power_plan_verification_restores_and_verifies_prior_plan(monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    runner = StatefulPowerRunner(ignore_changes=True)

    with pytest.raises(RuntimeError, match="prior plan was restored and verified"):
        WindowsActions(runner).set_active_power_plan("high_performance")

    assert runner.active == runner.balanced


def test_agent_policy_visibility_and_audit_are_wired_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    monkeypatch.setattr(
        "localpilot.operator.subprocess.Popen",
        lambda _argv, **_kwargs: SimpleNamespace(pid=2468),
    )
    enabled = LocalPilotAgent(Config(), tmp_path / "enabled")
    result = enabled.tools["open_windows_app"].fn("notepad")

    assert result["status"] == "started"
    event = enabled.audit.latest("operator_command_executed")
    assert event["action"] == "open_windows_app:notepad"
    assert event["risk"] == "reversible"
    assert event["process_id"] == 2468
    assert enabled.tools["open_windows_app"].fn in enabled._functions()

    disabled_config = Config()
    disabled_config.safety.auto_allow_reversible = False
    disabled = LocalPilotAgent(disabled_config, tmp_path / "disabled")
    assert disabled.tools["open_windows_app"].fn not in disabled._functions()


def test_power_rollback_capability_is_redacted_from_durable_audit_preview():
    result = {
        "status": "changed",
        "rollback_token": "one-use-secret-token",
        "active_plan": "balanced",
    }

    preview = LocalPilotAgent._tool_result_audit_preview(
        "set_active_power_plan", result
    )

    assert "one-use-secret-token" not in preview
    assert "<redacted>" in preview
    assert result["rollback_token"] == "one-use-secret-token"
    arguments = LocalPilotAgent._tool_arguments_for_audit(
        "restore_power_plan",
        {"rollback_token": "one-use-secret-token"},
    )
    assert arguments == {"rollback_token": "<redacted>"}


def test_model_tool_loop_executes_allowlisted_reversible_action(tmp_path, monkeypatch):
    monkeypatch.setattr("localpilot.tools.windows_actions.os.name", "nt")
    launches = []
    monkeypatch.setattr(
        "localpilot.operator.subprocess.Popen",
        lambda argv, **kwargs: launches.append((argv, kwargs))
        or SimpleNamespace(pid=1357),
    )
    streams = iter(
        [
            [_chunk(tool_calls=[_call("open_windows_app", {"app": "calculator"})])],
            [_chunk(content="I have enough evidence to report the action result.")],
            [_chunk(content="Calculator opened successfully.")],
        ]
    )

    def fake_chat(**_kwargs):
        return iter(next(streams))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    agent = LocalPilotAgent(Config(), tmp_path)

    answer = agent.ask("Open Calculator.")

    assert answer == "Calculator opened successfully."
    assert launches[0][0] == ["calc.exe"]
    assert launches[0][1]["shell"] is False
    assert agent.audit.latest("tool_call")["risk"] == "reversible"
    assert agent.audit.latest("operator_command_executed")["status"] == "started"
