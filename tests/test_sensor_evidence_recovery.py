"""Production regressions: selection, recovery coverage, and source persistence."""
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from ollama import ResponseError

from localpilot.agent import LocalPilotAgent
from localpilot.answer_contract import AnswerContract, AnswerField
from localpilot.config import Config
from localpilot.systemsense_selection import SensorRequest
from localpilot.tools.systemsense import SystemSenseReader
from test_systemsense_truth_views import _sense, _sensor


PROMPT = (
    "Inspect current raw SystemSense sensors and report CPU Tctl/Tdie and CCD1, "
    "plus GPU core, memory and hot spot temperatures. Identify the provider and "
    "distinguish its readings from Windows thermal-zone errors."
)
ANSWER = (
    "LibreHardwareMonitorLib reports CPU Tctl/Tdie 51.25 C, CPU CCD1 50.25 C; "
    "GPU core 39 C, GPU memory 65 C, GPU hot spot 55 C. "
    "The separate Windows thermal-zone query failed (thermal:com_error); "
    "that does not invalidate the provider readings."
)
NAMES = ["CPU Tctl/Tdie", "CPU CCD1", "GPU core", "GPU memory", "GPU hot spot"]


def _rows(padding=20):
    rows = [_sensor(f"/fan/{i}", f"Fan {i}", "Control", 20, "/fan", "SuperIO", "Board")
            for i in range(padding)]
    for name, value, component in [
        ("Core (Tctl/Tdie)", 51.25, "Cpu"), ("CCD1 (Tdie)", 50.25, "Cpu"),
        ("GPU Core", 39, "GpuAmd"), ("GPU Memory", 65, "GpuAmd"),
        ("GPU Hot Spot", 55, "GpuAmd"),
    ]:
        rows.append(_sensor(f"/{component}/temperature/{len(rows)}", name,
                            "Temperature", value, f"/{component}", component, component))
    return rows


def _payload(rows=None):
    return {"captured_at": "2026-10-06T00:00:00Z", "hardware_truth": {
        "sensors": _rows() if rows is None else rows,
        "sensor_provider": {"source": "LibreHardwareMonitorLib", "available": True, "errors": []},
        "windows_performance": {"thermal_zones": [], "errors": ["thermal:com_error"]},
    }}


def _chunk(content="", calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, thinking="", tool_calls=list(calls)))


def _call(name, **arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _agent(tmp_path, monkeypatch, responses):
    cfg = Config()
    cfg.agent.research_soft_tool_rounds = 4
    cfg.agent.research_hard_tool_rounds = 4
    agent = LocalPilotAgent(cfg, tmp_path)
    agent.governor = SimpleNamespace(sample=lambda interval: SimpleNamespace(background_allowed=False),
                                    apply_process_priority=lambda idle: None)
    snapshots = []
    responses = iter(responses)
    def chat(**kwargs):
        snapshots.append([dict(m) for m in kwargs["messages"]])
        response = next(responses)
        if isinstance(response, Exception):
            def failed_stream():
                raise response
                yield  # Stream failures occur during iteration, as in production.
            return failed_stream()
        return iter([response])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat, ResponseError=ResponseError))
    return agent, snapshots


@pytest.mark.parametrize("padding", [20, 10000])
def test_named_sensor_selection_precedes_row_limit(tmp_path, padding):
    sense = _sense(tmp_path)
    provider = {"source": "LibreHardwareMonitorLib", "available": True,
                "errors": [], "sensors": _rows(padding)}
    sense.sensors.collect = lambda: provider
    sense.collect_dynamic()
    result = json.loads(SystemSenseReader(sense).inspect_raw_system_sense(
        category="sensors", limit=10, requested_sensors=NAMES))
    assert len(result["items"]) == 5
    assert result["selection"]["missing"] == []
    assert result["selection"]["truncated"] is False
    assert result["sensor_provider"]["source"] == "LibreHardwareMonitorLib"
    assert result["captured_at"]


def test_failed_history_does_not_clear_pc_requirement_and_summary_satisfies_it(tmp_path, monkeypatch):
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider")]),
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider_sensors")]),
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    answer = agent.ask(PROMPT)
    assert answer == ANSWER
    state = agent.audit.latest("model_evidence_state")
    assert state["required"] == ["Windows/PC state"]
    assert state["succeeded"] == ["Windows/PC state"]
    assert state["tool_rounds"] == 3


def test_generic_recovery_cannot_drop_requested_fields(tmp_path, monkeypatch):
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk("The system is running smoothly overall."),
        _chunk("The system is running smoothly overall."),
        _chunk(ANSWER),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    review = agent.audit.latest("model_same_context_postvalidation_complete")
    assert review["accepted"] is False
    assert review["missing_requested_fields"]
    assert PROMPT in snapshots[-1][-1]["content"]


def test_repeated_failure_ends_bounded_without_unsupported_synthesis(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider")]),
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider_sensors")]),
        _chunk("The system is running smoothly overall."),
        _chunk("The system is running smoothly overall."),
        _chunk("The system is running smoothly overall."),
    ])
    answer = agent.ask(PROMPT)
    assert "unverified" in answer
    assert "smoothly" not in answer
    assert agent.audit.latest("model_evidence_acquisition_failed")["missing"] == ["Windows/PC state"]
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] <= 4


@pytest.mark.parametrize("aliases", [NAMES, ["Tctl/Tdie", "CCD1 (Tdie)", "GPU Core", "GPU Memory", "GPU hotspot"]])
def test_aliases_and_missing_metrics_are_explicit(tmp_path, aliases):
    sense = _sense(tmp_path)
    provider = {"source": "LibreHardwareMonitorLib", "available": True,
                "errors": [], "sensors": _rows()[:-1]}
    sense.sensors.collect = lambda: provider
    sense.collect_dynamic()
    result = json.loads(SystemSenseReader(sense).inspect_raw_system_sense(
        category="sensors", requested_sensors=aliases, sensor_type="Temperature", components=["cpu", "gpu"]))
    assert len(result["items"]) == 4
    assert result["selection"]["missing"] == [aliases[-1]]
    gaps, contract = SensorRequest.from_prompt(PROMPT).evaluate(json.dumps(result))
    assert "GPU hot spot" in gaps


def test_small_limit_reports_incomplete_selection(tmp_path):
    sense = _sense(tmp_path)
    sense.sensors.collect = lambda: {"source": "LibreHardwareMonitorLib", "available": True,
                                    "sensors": _rows()}
    sense.collect_dynamic()
    result = json.loads(SystemSenseReader(sense).inspect_raw_system_sense(
        category="sensors", limit=2, requested_sensors=NAMES, sensor_type="Temperature"))
    assert len(result["items"]) == 2
    assert result["selection"]["truncated"] is True
    assert len(result["selection"]["missing"]) == 3
    assert "complete requested sensor selection" in SensorRequest.from_prompt(PROMPT).evaluate(json.dumps(result))[0]


def test_named_request_binds_raw_arguments_before_default_slice(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("inspect_raw_system_sense", category="temperature_sensors", limit=10)]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    sense = _sense(tmp_path / "sense")
    sense.sensors.collect = lambda: {"source": "LibreHardwareMonitorLib", "available": True,
                                    "errors": [], "sensors": _rows(10000)}
    sense.performance.collect = lambda: {"thermal_zones": [], "errors": ["thermal:com_error"]}
    sense.collect_dynamic()
    agent.tools["inspect_raw_system_sense"] = replace(
        agent.tools["inspect_raw_system_sense"], fn=SystemSenseReader(sense).inspect_raw_system_sense)
    assert agent.ask(PROMPT) == ANSWER
    call = agent.audit.latest("tool_call")
    assert call["args"]["requested_sensors"] == NAMES
    assert call["args"]["limit"] == 10
    assert agent.audit.latest("model_requested_evidence_coverage")["complete"] is True
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 1


def test_inventory_cannot_satisfy_named_temperatures(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("inspect_hardware_inventory")]),
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    agent.tools["inspect_hardware_inventory"] = replace(
        agent.tools["inspect_hardware_inventory"], fn=lambda **kw: '{"available":true}')
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    results = list(reversed(agent.audit.recent("tool_result", limit=100)))
    assert results[0]["ok"] is False
    assert results[1]["ok"] is True


def test_behavior_recovery_retains_fields_and_original_request(tmp_path, monkeypatch):
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk("| Status | Reading |\n| --- | --- |\n| CPU | 51 C |"),
        _chunk("| Status | Reading |\n| --- | --- |\n| CPU | 51 C |"),
        _chunk("The system is running smoothly overall."),
        _chunk(ANSWER),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    for snapshot in snapshots[3:]:
        assert PROMPT in snapshot[-1]["content"]
        assert all(name in snapshot[-1]["content"] for name in NAMES)


def test_generic_answer_contract_is_not_specific_to_sensors(tmp_path, monkeypatch):
    agent, snapshots = _agent(tmp_path, monkeypatch, [_chunk("Everything is fine."),
        _chunk("Everything is fine."), _chunk("Firmware vendor Acme, schema version 3.")])
    contract = AnswerContract((AnswerField("vendor", ("vendor",), "Acme"),
                               AnswerField("schema version", ("schema version",), 3)))
    answer = agent._continue_high_reasoning_answer(
        sys.modules["ollama"].chat, prompt="Report the firmware vendor and schema version.",
        round_no=1, after_tools=True, answer_contract=contract)
    assert answer == "Firmware vendor Acme, schema version 3."
    assert "schema version" in snapshots[-1][-1]["content"]


@pytest.mark.parametrize("answer", ["The system is running smoothly overall.",
    ANSWER.replace("GPU memory 65 C", "GPU memory 39 C"), ANSWER.replace("GPU hot spot 55 C. ", "")])
def test_incomplete_or_wrong_fields_are_rejected(answer):
    _, contract = SensorRequest.from_prompt(PROMPT).evaluate(json.dumps(_payload()))
    assert contract.gaps(answer)
    assert not contract.gaps(ANSWER)


def test_failed_history_and_repository_attempts_preserve_pc_until_terminal_failure(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider")]),
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider_sensors")]),
        _chunk(calls=[_call("search_repository", query="unknown_history")]),
        _chunk(calls=[_call("search_repository", query="unknown_sensors")]),
        _chunk("Everything is healthy."),
    ])
    agent.tools["search_repository"] = replace(agent.tools["search_repository"],
                                               fn=lambda **kw: "Tool error: no repository evidence")
    answer = agent.ask(PROMPT)
    assert "unverified" in answer
    assert "healthy" not in answer
    assert agent.audit.latest("model_evidence_state")["required"] == ["Windows/PC state"]
    assert agent.audit.latest("model_research_stagnation_adaptation")["outstanding"] == ["Windows/PC state"]
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] <= 4


def test_repeated_recovery_cannot_accept_generic_answer(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        *[_chunk("The system is running smoothly overall.") for _ in range(4)],
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT).startswith("[LocalPilot withheld")
    correction = agent.audit.latest("model_same_context_authority_correction_complete")
    assert correction["accepted"] is False
    assert correction["attempts"] == 2
    assert correction["remaining_gaps"]


def test_bounded_reacquisition_after_incomplete_metrics(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(calls=[_call("inspect_raw_system_sense", category="sensors", limit=10)]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload(_rows()[:-1])))
    agent.tools["inspect_raw_system_sense"] = replace(
        agent.tools["inspect_raw_system_sense"], fn=lambda **kw: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    states = list(reversed(agent.audit.recent("model_evidence_state")))
    assert states[0]["succeeded"] == []
    assert states[1]["succeeded"] == ["Windows/PC state"]
    assert states[1]["tool_rounds"] == 2


def test_read_after_soft_budget_uses_same_bound_selection_as_checkpoint(tmp_path, monkeypatch):
    checkpoint = _call("update_research_notebook", evidence_refs=["obs-001", "obs-002"],
        unresolved_fact="current named CPU and GPU temperatures", proposed_tool="inspect_raw_system_sense",
        proposed_arguments={"category": "sensors", "limit": 10},
        result_that_would_change_the_conclusion="Current raw temperatures would answer the owner's request.")
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider")]),
        _chunk(calls=[_call("get_system_sense_history", metric="hardware_provider_sensors")]),
        _chunk(calls=[checkpoint]),
        _chunk(calls=[_call("inspect_raw_system_sense", category="sensors", limit=10)]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    agent.config.agent.research_soft_tool_rounds = 2
    agent.tools["inspect_raw_system_sense"] = replace(
        agent.tools["inspect_raw_system_sense"], fn=lambda **kw: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 3


def test_reset_recovery_carries_answer_contract_into_recursive_validation(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk("Hello! How can I help you today?"),
        _chunk("The system is running smoothly overall."),
        _chunk(ANSWER),
    ])
    _, contract = SensorRequest.from_prompt(PROMPT).evaluate(json.dumps(_payload()))
    assert agent._continue_high_reasoning_answer(
        sys.modules["ollama"].chat, prompt=PROMPT, round_no=1, after_tools=True,
        successful_tools=frozenset({"get_system_sense_summary"}), answer_contract=contract) == ANSWER
    assert agent.audit.latest("model_same_context_postvalidation_complete")["accepted"] is False


def test_raw_tool_schema_exposes_categories_and_named_selection(tmp_path, monkeypatch):
    from ollama import Tool
    agent, _ = _agent(tmp_path, monkeypatch, [])
    schema = next(tool for tool in agent._functions() if isinstance(tool, dict)
                  and tool["function"]["name"] == "inspect_raw_system_sense")
    schema = Tool.model_validate(schema).model_dump()
    properties = schema["function"]["parameters"]["properties"]
    assert properties["category"]["enum"] == ["dynamic", "sensors", "inventory", "backend"]
    assert "requested_sensors" in properties and "sensor_type" in properties


@pytest.mark.parametrize("payload", [[], [{"pid": 123, "cpu_percent": 2}], "healthy", 1, None])
def test_unrelated_json_shapes_cannot_satisfy_temperature_request(payload):
    missing, contract = SensorRequest.from_prompt(PROMPT).evaluate(json.dumps(payload))
    assert missing
    assert contract.fields == ()


@pytest.mark.parametrize("prompt", [PROMPT,
    "What CPU and GPU temperatures can you see in the current raw SystemSense hardware-provider sensors? "
    "Distinguish those readings from Windows thermal-zone errors."])
def test_owner_metric_selection_overrides_one_row_slice(tmp_path, monkeypatch, prompt):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("inspect_raw_system_sense", category="sensors", limit=1)]),
        _chunk(ANSWER), _chunk(ANSWER),
    ])
    sense = _sense(tmp_path / "sense")
    sense.sensors.collect = lambda: {"source": "LibreHardwareMonitorLib", "available": True,
                                    "errors": [], "sensors": _rows(10000)}
    sense.performance.collect = lambda: {"thermal_zones": [], "errors": ["thermal:com_error"]}
    sense.collect_dynamic()
    agent.tools["inspect_raw_system_sense"] = replace(
        agent.tools["inspect_raw_system_sense"], fn=SystemSenseReader(sense).inspect_raw_system_sense)
    assert agent.ask(prompt) == ANSWER
    coverage = agent.audit.latest("model_requested_evidence_coverage")
    assert coverage["complete"] and len(coverage["evidence_fields"]) == 7
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 1
    assert agent.audit.latest("tool_call")["args"]["limit"] <= 500


def test_failed_sensor_read_then_process_list_does_not_clear_requirement(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(calls=[_call("get_top_processes", limit=5)]),
        _chunk(calls=[_call("get_storage_summary")]),
        _chunk(calls=[_call("get_system_summary")]),
        _chunk("The requested GPU sensors are unavailable."),
    ])
    agent.tools["get_system_sense_summary"] = replace(agent.tools["get_system_sense_summary"],
                                                       fn=lambda: "Tool error: sensor read failed")
    for tool in ("get_top_processes", "get_storage_summary", "get_system_summary"):
        agent.tools[tool] = replace(agent.tools[tool], fn=lambda **kw: json.dumps([{"unrelated": True}]))
    assert agent.ask(PROMPT) == "The requested GPU sensors are unavailable."
    state = agent.audit.latest("model_evidence_state")
    assert state["required"] == ["Windows/PC state"] and state["succeeded"] == []
    assert state["tool_rounds"] == 4


@pytest.mark.parametrize("operator_error", [False, True])
def test_answer_protocol_recovery_keeps_acquired_fields_and_shared_retry_bound(tmp_path, monkeypatch, operator_error):
    from ollama import ResponseError
    without_provider = ANSWER.replace("LibreHardwareMonitorLib", "The provider")
    responses = ([ResponseError("error parsing tool call: malformed arguments")] if operator_error else []) + [
        _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(without_provider), _chunk(without_provider),
        ResponseError("error parsing tool call: malformed arguments"), _chunk(ANSWER),
    ]
    agent, snapshots = _agent(tmp_path, monkeypatch, responses)
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    retry = agent.audit.latest("model_answer_tool_call_protocol_recovery_retry")
    assert retry["attempt"] == (2 if operator_error else 1)
    assert retry["retry_limit"] == 2
    context = snapshots[-1]
    assert not any(message.get("role") == "tool" or message.get("tool_calls") for message in context)
    assert "LibreHardwareMonitorLib" in str(context) and "51.25" in str(context)
    assert PROMPT in context[-1]["content"]
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 1


def test_answer_protocol_exhaustion_is_bounded_and_never_becomes_unsupported_answer(tmp_path, monkeypatch):
    from ollama import ResponseError
    without_provider = ANSWER.replace("LibreHardwareMonitorLib", "The provider")
    bad = lambda: ResponseError("error parsing tool call: malformed arguments")
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        bad(), _chunk(calls=[_call("get_system_sense_summary")]),
        _chunk(without_provider), _chunk(without_provider), bad(), bad(),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    answer = agent.ask(PROMPT)
    assert answer.startswith("[LocalPilot") and "bounded" in answer
    exhausted = agent.audit.latest("model_answer_tool_call_protocol_recovery_exhausted")
    assert exhausted["retries"] == exhausted["retry_limit"] == 2
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 1


def test_answer_behavior_recovery_protocol_error_retains_original_fields(tmp_path, monkeypatch):
    from ollama import ResponseError
    table = "| Sensor | Temperature |\n| --- | --- |\n| CPU | 51.25 C |"
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]), _chunk(table), _chunk(table),
        ResponseError("error parsing tool call: malformed arguments"), _chunk(ANSWER),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    assert agent.audit.latest("model_answer_tool_call_protocol_recovery_retry")["attempt"] == 1
    assert PROMPT in snapshots[-1][-1]["content"]


def test_protocol_failure_at_tool_ceiling_still_renders_acquired_fields(tmp_path, monkeypatch):
    agent, snapshots = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]),
        ResponseError("error parsing tool call: malformed arguments"), _chunk(ANSWER),
    ])
    agent.config.agent.research_soft_tool_rounds = 1
    agent.config.agent.research_hard_tool_rounds = 1
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    assert agent.ask(PROMPT) == ANSWER
    assert agent.audit.latest("model_evidence_state")["tool_rounds"] == 1
    assert agent.audit.latest("model_answer_tool_call_protocol_recovery_retry")["attempt"] == 1
    assert not any(message.get("role") == "tool" or message.get("tool_calls") for message in snapshots[-1])


def test_unrelated_server_error_during_answer_correction_still_surfaces(tmp_path, monkeypatch):
    without_provider = ANSWER.replace("LibreHardwareMonitorLib", "The provider")
    agent, _ = _agent(tmp_path, monkeypatch, [
        _chunk(calls=[_call("get_system_sense_summary")]), _chunk(without_provider),
        _chunk(without_provider), ResponseError("Internal Server Error", 500),
    ])
    agent.tools["get_system_sense_summary"] = replace(
        agent.tools["get_system_sense_summary"], fn=lambda: json.dumps(_payload()))
    with pytest.raises(ResponseError, match="Internal Server Error"):
        agent.ask(PROMPT)
    assert agent.audit.latest("model_answer_tool_call_protocol_recovery_retry") is None
