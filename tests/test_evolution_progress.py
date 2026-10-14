from types import SimpleNamespace

import pytest

from localpilot.config import Config
from localpilot import selfdev


def developer(tmp_path, messages):
    config = Config()
    config.agent.data_dir = "data"
    return selfdev.SelfDeveloper(config, tmp_path, progress=messages.append)


def test_preflight_and_resource_deferral_are_visible_without_model_work(tmp_path, monkeypatch):
    messages = []
    dev = developer(tmp_path, messages)
    dev.github.sync_trusted_main = lambda: SimpleNamespace(ok=True, updated=False, summary="current")
    dev._reconcile_candidates = lambda: None
    dev.governor = SimpleNamespace(
        sample=lambda: SimpleNamespace(allows_selfdev=lambda **kw: False,
                                       blocking_reason=lambda **kw: "RAM above admission limit"),
        apply_process_priority=lambda **kw: None,
    )
    dev._select_developer_model = lambda: pytest.fail("Deferred cycle must not load a model")
    result = dev.run_once(force=True)
    assert result.status == "deferred"
    assert "RAM above admission limit" in result.summary
    assert messages == ["Starting self-development preflight", "Checking trusted main", "Evolution result: deferred"]


def test_wait_heartbeat_is_throttled_and_never_emits_reasoning(tmp_path, monkeypatch):
    messages = []
    dev = developer(tmp_path, messages)
    checks = []
    dev._check_resources = lambda *args, **kw: checks.append((args, kw))
    ticks = iter([0, 1, 2, 29, 30, 31, 60])
    monkeypatch.setattr(selfdev.time, "monotonic", lambda: next(ticks))

    def model_call(chat, **kwargs):
        for _ in range(6):
            kwargs["stream_guard"]()
        return "private model reasoning"

    monkeypatch.setattr(selfdev, "developer_chat", model_call)
    answer = dev._developer_chat(None, force=True, branch="candidate", model="test")
    assert answer == "private model reasoning"
    assert messages == ["Waiting for the developer model response"] + ["Still waiting for the developer model response"] * 2
    assert len(checks) == 6
    assert all(kw == {"during_inference": True} for _, kw in checks)
    assert "private" not in str(messages)


def test_crash_reports_stage_without_leaking_exception_payload(tmp_path, monkeypatch):
    messages = []
    dev = developer(tmp_path, messages)
    def fail(**kw):
        raise ValueError("private response payload")
    dev._run_once = fail
    with pytest.raises(ValueError):
        dev.run_once(force=True)
    assert messages == ["Starting self-development preflight", "Evolution stopped: ValueError"]
