"""Offline contract tests for opt-in hosted mentor consultations."""
from __future__ import annotations

import json
import sys
import urllib.request
from types import SimpleNamespace

import pytest

from localpilot.config import Config, load_config
from localpilot.agent import LocalPilotAgent
from localpilot.agent_tools import _forbidden_tools, _tool_arguments_for_audit, _tool_result_audit_preview
from localpilot.tools import registry
from localpilot.tools.external_mentor import ExternalMentor


def test_disabled_in_strict_and_default_configs():
    cfg = Config()
    assert cfg.mentor.provider == "groq"
    assert cfg.mentor.model == "openai/gpt-oss-120b"
    assert "consult_external_mentor" not in registry(config=cfg)
    cfg.mentor.enabled = True
    assert "consult_external_mentor" not in registry(config=cfg)
    cfg.agent.scaffold_mode = "guide_first"
    assert "consult_external_mentor" in registry(config=cfg)


def test_opt_in_config_validation(tmp_path):
    file = tmp_path / "localpilot.toml"
    file.write_text('[agent]\nscaffold_mode = "guide_first"\n[mentor]\nenabled = true\nprovider = "groq"\nmodel = "openai/gpt-oss-120b"\nmax_requests_per_session = 2\n', encoding="utf-8")
    cfg = load_config(file)
    assert cfg.mentor.enabled and cfg.mentor.max_requests_per_session == 2
    assert cfg.mentor.provider == "groq"
    file.write_text("[mentor]\nmax_requests_per_session = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_requests_per_session"):
        load_config(file)
    file.write_text('[mentor]\nprovider = "openrouter"\nmodel = "openai/gpt-oss-120b"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":free"):
        load_config(file)


def test_preserves_strict_web_prohibition_and_redacts_audit():
    assert _forbidden_tools("Don't use the internet.") == frozenset({
        "search_public_web", "fetch_public_https"
    })
    assert _tool_arguments_for_audit("consult_external_mentor", {"question": "private words"}) == {"question_chars": 13}
    assert "secret" not in _tool_result_audit_preview("consult_external_mentor", "secret")


def test_owner_no_web_excludes_external_mentor_from_tool_invocation(tmp_path, monkeypatch):
    cfg = Config()
    cfg.agent.scaffold_mode = "guide_first"
    cfg.mentor.enabled = True
    cfg.systemsense.enabled = False
    agent = LocalPilotAgent(cfg, tmp_path)
    offered = []
    def fake_chat(**kwargs):
        offered.append(kwargs.get("tools") or [])
        return iter([SimpleNamespace(message=SimpleNamespace(
            content="I can reason without contacting the web.",
            thinking="", tool_calls=[]
        ))])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=fake_chat))
    assert "without contacting the web" in agent.ask(
        "Investigate a hypothetical algorithm carefully but do not use the public web."
    )
    assert offered
    assert "consult_external_mentor" not in str(offered)


def test_missing_key_and_sensitive_material_never_make_requests(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    mentor = ExternalMentor(model="openai/gpt-oss-120b", provider="groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        mentor.consult_external_mentor("How should I investigate a race condition?")
    monkeypatch.setenv("GROQ_API_KEY", "test-local-secret")
    for question in (
        "What can I do about test@example.com appearing in an error report?",
        "Why does this leak happen? password=superprivate",
        "Can I quote this snippet? " + chr(96)*3 + "py" + chr(96)*3,
    ):
        with pytest.raises(ValueError):
            mentor.consult_external_mentor(question)


@pytest.mark.parametrize("provider,model,key_name,endpoint", [
    ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY",
     "https://api.groq.com/openai/v1/chat/completions"),
    ("openrouter", "openai/gpt-oss-120b:free", "OPENROUTER_API_KEY",
     "https://openrouter.ai/api/v1/chat/completions"),
    ("openai", "gpt-5.5", "OPENAI_API_KEY",
     "https://api.openai.com/v1/responses"),
])
def test_hosted_advice_bounded_and_never_requires_local_model(
        monkeypatch, provider, model, key_name, endpoint):
    monkeypatch.setenv(key_name, "test-local-secret")
    captured = []
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, max_bytes):
            if provider == "openai":
                response = {"output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "Compare two hypotheses before changing code."}
                ]}]}
            else:
                response = {"choices": [{"message": {
                    "content": "Compare two hypotheses before changing code."
                }}]}
            return json.dumps(response).encode("utf-8")
    class FakeOpener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            return FakeResponse()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: FakeOpener())
    mentor = ExternalMentor(provider=provider, model=model, max_requests_per_session=1)
    answer = mentor.consult_external_mentor(
        "How should I debug a hypothetical cache invalidation bug?"
    )
    assert "UNTRUSTED" in answer and "Compare two hypotheses" in answer
    request, timeout = captured[0]
    assert request.full_url == endpoint
    assert request.get_method() == "POST" and timeout == 45
    assert request.get_header("Authorization") == "Bearer test-local-secret"
    payload = json.loads(request.data)
    assert payload["model"] == model
    if provider == "openai":
        assert payload["store"] is False and payload["tools"] == []
        assert payload["input"].startswith("How should")
    else:
        assert payload["messages"][-1]["content"].startswith("How should")
        assert "tools" not in payload
    assert "test-local-secret" not in request.data.decode()
    with pytest.raises(RuntimeError, match="session request limit"):
        mentor.consult_external_mentor("Could I ask an additional question about this?")
    assert len(captured) == 1


def test_openrouter_rejects_paid_variant():
    with pytest.raises(ValueError, match="free model"):
        ExternalMentor(provider="openrouter", model="openai/gpt-oss-120b")
