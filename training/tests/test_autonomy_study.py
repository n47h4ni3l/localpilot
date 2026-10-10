from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.scripts import run_autonomy_study as study
from training.scripts import review_autonomy_study as review
from training.scripts.autonomy_tasks import authored_tasks


def test_fresh_48_cell_plan_has_matched_trios_and_balanced_positions():
    tasks = authored_tasks()
    planned = study.plan(tasks)
    assert len(tasks) == 8 and len(planned) == 48
    assert len({c["id"] for c in planned}) == 48
    assert {s: sum(t["stratum"] == s for t in tasks) for s in
            ("supplied_source", "live_source")} == {"supplied_source": 4, "live_source": 4}
    for task in tasks:
        for repeat in (1, 2):
            trio = [c for c in planned if c["task_id"] == task["id"] and c["repeat"] == repeat]
            assert {c["arm"] for c in trio} == set(study.ARMS)
            assert len({c["seed"] for c in trio}) == 1
    for arm in study.ARMS:
        positions = [sum(planned[i + pos]["arm"] == arm for i in range(0, 48, 3)) for pos in range(3)]
        assert max(positions) - min(positions) <= 1
    assert all("lp-postrepair" not in t["id"] and "fresh-" not in t["id"] for t in tasks)


def test_budget_is_resource_only_and_unknown_usage_stops_extra_calls():
    clock = [0.0]
    b = study.Budget({**study.BUDGETS, "generated_tokens": 10000}, study.OPTIONS, lambda: clock[0])
    assert b.call()["num_predict"] == 8192
    b.complete({"eval_count": 7000}, 8192)
    assert b.call()["num_predict"] == 3000
    b.complete(None, 3000)
    assert b.tokens == 10000 and b.unmetered
    with pytest.raises(study.StudyBudgetExceeded):
        b.call()


def test_budget_limits_calls_tools_and_elapsed_time():
    clock = [0.0]
    b = study.Budget({**study.BUDGETS, "model_calls": 1, "tool_calls": 1}, study.OPTIONS, lambda: clock[0])
    b.call()
    b.complete({"eval_count": 0}, 8192)
    with pytest.raises(study.StudyBudgetExceeded):
        b.call()
    b.tool()
    with pytest.raises(study.StudyBudgetExceeded):
        b.tool()
    fresh = study.Budget(study.BUDGETS, study.OPTIONS, lambda: clock[0])
    clock[0] = 601
    with pytest.raises(study.StudyBudgetExceeded):
        fresh.call()


def test_sources_are_fully_captured_and_drift_is_rejected():
    text = "A" * 60000
    sha = hashlib.sha256(text.encode()).hexdigest()

    def fetch(url, max_chars, start_char):
        end = min(start_char + max_chars, len(text))
        return (f"Source coverage: chars={start_char}-{end}/{len(text)}; truncated=true\n"
                f"Source text SHA-256: {sha}\n\n{text[start_char:end]}")

    source = study.capture_source("https://example.org", fetch)
    assert source["text"] == text and len(source["pages"]) == 2

    def drifting(url, max_chars, start_char):
        raw = fetch(url, max_chars, start_char)
        return raw.replace(sha, "0" * 64) if start_char else raw

    with pytest.raises(RuntimeError, match="changed during"):
        study.capture_source("https://example.org", drifting)


def test_packet_sections_avoid_toc_and_fail_for_missing_section():
    text = "Start End\n\nStart\n" + "genuine source passage " * 10 + "End"
    section = study.packet_section({"text": text}, "Start", "End")
    assert "genuine source passage" in section
    with pytest.raises(RuntimeError, match="not found"):
        study.packet_section({"text": text}, "Absent", "End")


def test_isolation_never_loads_owner_configuration_or_enables_mentor(tmp_path, monkeypatch):
    from localpilot import config

    monkeypatch.setattr(config, "load_config", lambda *a: pytest.fail("Owner config loaded"))
    for arm in study.ARMS:
        cfg = study.isolated_config(tmp_path / "runtime", tmp_path / "state" / arm, arm,
                                   {"model": study.MODEL, "think": "high", "options": study.OPTIONS, "keep_alive": "30m"})
        assert not any((cfg.mentor.enabled, cfg.library.enabled, cfg.systemsense.enabled,
                        cfg.selfdev.enabled, cfg.github.enabled, cfg.model.memory_embeddings_enabled,
                        cfg.agent.feedback_coaching_enabled, cfg.agent.feedback_auto_observations_enabled))
        assert not cfg.safety.auto_allow_reversible
        assert cfg.safety.require_confirmation_for_destructive
        assert str(tmp_path / "state") in cfg.agent.data_dir


def test_child_environment_removes_credentials_and_config_overrides(monkeypatch):
    for name in study.ENV_DROP:
        monkeypatch.setenv(name, "do-not-leak")
    env = study.child_environment()
    assert not (set(k.upper() for k in env) & study.ENV_DROP)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_tool_surface_cannot_expose_actions_or_mentor(tmp_path):
    from localpilot.safety import ToolSpec, RiskLevel

    budget = study.Budget(study.BUDGETS, study.OPTIONS)
    for name, risk in (("fetch_public_https", RiskLevel.REVERSIBLE),
                       ("consult_external_mentor", RiskLevel.READ_ONLY),
                       ("read_repository_file", RiskLevel.READ_ONLY)):
        with pytest.raises(RuntimeError, match="tool surface"):
            study.wrapped_tools({name: ToolSpec(name, "", risk, lambda: None)}, {}, tmp_path / "row.json", budget)


class FakeClient:
    def list(self):
        return {"models": [{"model": study.MODEL, "digest": study.P1_DIGEST}]}

    def chat(self, **kwargs):
        self.request = kwargs
        yield {"message": {"content": "  Unsupported unsafe advice.\n", "thinking": "raw thought"}, "done": False}
        yield {"message": {"content": ""}, "done": True, "eval_count": 42, "prompt_eval_count": 100}


def _protocol():
    return {"model": study.MODEL, "model_digest": study.P1_DIGEST, "think": "high",
            "options": study.OPTIONS, "budgets": study.BUDGETS, "keep_alive": "30m",
            "boundary": study.BOUNDARY, "tasks": [], "plan": [], "review":
            {"dimensions": ["accuracy", "utility", "unsafe_advice"]}}


def test_transport_matches_effective_parameters_and_preserves_raw(tmp_path):
    client, row = FakeClient(), {"turns": []}
    b = study.Budget(study.BUDGETS, study.OPTIONS)
    chat = study.controlled_chat(client, row, tmp_path / "cell.json", b, _protocol(), 13)
    result = list(chat(model="wrong-model", think="low", options={"num_predict": 5},
                       messages=[{"role": "user", "content": "same question"}], stream=False))
    assert client.request["think"] == "high" and client.request["model"] == study.MODEL
    assert client.request["options"] == {**study.OPTIONS, "seed": 13}
    assert row["turns"][0]["requested"]["think"] == "low"
    raw = [json.loads(s) for s in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert raw[0]["chunk"]["message"]["content"] == result[0]["message"]["content"]
    assert b.tokens == 42 and not b.unmetered


def test_partial_stream_is_saved_and_usage_not_silently_zero(tmp_path):
    class Broken(FakeClient):
        def chat(self, **kwargs):
            yield {"message": {"content": "partial draft"}, "done": False}
            raise RuntimeError("stream failed")

    row, b = {"turns": []}, study.Budget(study.BUDGETS, study.OPTIONS)
    with pytest.raises(RuntimeError, match="stream failed"):
        list(study.controlled_chat(Broken(), row, tmp_path / "cell.json", b, _protocol(), 1)(messages=[]))
    assert "partial draft" in next(tmp_path.glob("*.jsonl")).read_text()
    assert b.unmetered and b.tokens == 8192
    with pytest.raises(study.StudyBudgetExceeded):
        b.call()


@pytest.mark.parametrize("answer", ["", "  Wrong answer.\n", "DECLINE: my decision"])
def test_direct_control_has_no_answer_review_recovery_or_rewriting(answer):
    calls, row = [], {}

    def chat(**kwargs):
        calls.append(kwargs)
        yield {"message": {"content": answer, "thinking": "never substitute this for delivery"}}

    assert study.direct_answer(chat, {"question": "q"}, {}, row, _protocol()) == answer
    assert len(calls) == 1


def _saved_study(tmp_path: Path):
    root = tmp_path / "study"
    root.mkdir()
    protocol = _protocol()
    tasks = authored_tasks()
    for t in tasks:
        t["question_sha256"] = hashlib.sha256(t["question"].encode()).hexdigest()
        t["source_provenance"] = []
    protocol.update(tasks=tasks, plan=study.plan(tasks))
    study.save(root / "protocol.json", protocol)
    entry = protocol["plan"][0]
    task = next(t for t in tasks if t["id"] == entry["task_id"])
    study.save(root / "cells" / f"{entry['id']}.json", {**entry,
        "question_sha256": task["question_sha256"], "response": "", "error": {"type": "timeout"},
        "seconds": 5, "turns": [], "tool_results": []})
    return root, protocol


def test_blind_cards_preserve_empty_failed_missing_cells_without_metadata(tmp_path):
    root, protocol = _saved_study(tmp_path)
    bundle, key_path = tmp_path / "blind", tmp_path / "key.json"
    review.cards(root, bundle, key_path)
    cards = [json.loads(s) for s in (bundle / "cards.jsonl").read_text().splitlines()]
    assert len(cards) == 48
    assert all(c["response"] == "" and c["accuracy"] is None for c in cards)
    assert all(not {"arm", "repeat", "seconds", "error", "turns", "tool_results"} & set(c) for c in cards)
    assert sum(c["collected"] for c in study.read(key_path)["cards"].values()) == 1
    summary = review.summarize(root, key_path, None, tmp_path / "summary.json")
    assert summary["planned"] == 48 and summary["collected"] == 1
    assert summary["independently_graded_cells"] == 0
    assert all(g["grades"]["accuracy"]["mean"] is None for g in summary["groups"])
    with pytest.raises(ValueError, match="overwrite"):
        review.cards(root, bundle, key_path)


def test_offline_grading_requires_independent_attestation_and_response_integrity(tmp_path):
    root, _ = _saved_study(tmp_path)
    bundle, key = tmp_path / "blind", tmp_path / "key.json"
    review.cards(root, bundle, key)
    mapping = study.read(key)["cards"]
    cards = [json.loads(s) for s in (bundle / "cards.jsonl").read_text().splitlines()]
    card = next(c for c in cards if mapping[c["card_id"]]["collected"])
    card.update(accuracy=0, utility=0, unsafe_advice=0, reviewer="test-independent-reviewer",
                rationale="Empty delivery", source_citations=["supplied question"], independent_of_execution=False)
    path = tmp_path / "grades.jsonl"
    path.write_text(json.dumps(card) + "\n")
    with pytest.raises(ValueError, match="independent"):
        review.summarize(root, key, path, tmp_path / "summary.json")
    card["independent_of_execution"] = True
    path.write_text(json.dumps(card) + "\n")
    assert review.summarize(root, key, path, tmp_path / "summary.json")["independently_graded_cells"] == 1
    card["response"] = "injected answer"
    path.write_text(json.dumps(card) + "\n")
    with pytest.raises(ValueError, match="changed"):
        review.summarize(root, key, path, tmp_path / "other.json")


def test_arm_key_cannot_be_inside_blind_bundle(tmp_path):
    root, _ = _saved_study(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        review.cards(root, tmp_path / "blind", tmp_path / "blind" / "key.json")


def test_historical_strict_archive_preserves_original_runtime_bytes(tmp_path):
    runtime = study.archive(study.STRICT_REVISION, tmp_path / "strict")
    import subprocess

    for name in ("agent.py", "agent_prompt.py", "authority.py", "safety.py"):
        original = subprocess.check_output(["git", "-C", str(study.ROOT), "show",
                                           f"{study.STRICT_REVISION}:localpilot/{name}"])
        assert (tmp_path / "strict" / "localpilot" / name).read_bytes() == original
    assert not (tmp_path / "strict" / "training").exists()
    assert not (tmp_path / "strict" / "localpilot.toml").exists()
    assert study.tree_hashes(tmp_path / "strict") == runtime["files"]


def test_launch_blocked_runtime_does_not_start_cell_or_modify_service(tmp_path, monkeypatch):
    protocol = _protocol()
    monkeypatch.setattr(study, "validate_freeze", lambda *a: protocol)
    monkeypatch.setattr(study, "readiness", lambda: {"blockers": ["worker running"], "ready": False})
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **kw: pytest.fail("Inference dispatched"))
    monkeypatch.setattr(study, "production_invariants", lambda *a: pytest.fail("No isolation yet"))
    review_path = tmp_path / "review.json"
    study.save(review_path, {"protocol_sha256": "frozen", "reviewer": "code second review",
                            "implementation_and_protocol_approved": True})
    with pytest.raises(RuntimeError, match="Runtime not ready"):
        study.run(tmp_path / "out", "frozen", review_path, tmp_path / "production", tmp_path / "models")
    assert not (tmp_path / "out" / "run-started.json").exists()


def test_freeze_detects_mutated_source_archive_and_client(tmp_path, monkeypatch):
    output = tmp_path / "freeze"
    output.mkdir()
    tasks = authored_tasks()
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("evaluator")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "git", lambda *a: "revision" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: "client-version")
    runtime = output / "runtime" / "strict"
    runtime.mkdir(parents=True)
    (runtime / "control.py").write_text("original")
    source = output / "sources" / "primary.json"
    study.save(source, {"text": "original"})
    protocol = {"tasks": tasks, "task_digest": study.digest(tasks), "plan": study.plan(tasks),
        "evaluator_revision": "revision", "runtimes": {"strict":
        {"revision": study.STRICT_REVISION, "files": study.tree_hashes(runtime)}},
        "source_files": {"primary": study.file_hash(source)},
        "evaluator_files": {"eval.py": study.file_hash(evaluator)},
        "client_runtime": {"python": study.sys.version, "platform": study.sys.platform,
                           "ollama_library": "client-version"}}
    study.save(output / "protocol.json", protocol)
    expected = study.file_hash(output / "protocol.json")
    assert study.validate_freeze(output, expected) == protocol
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen source"):
        study.validate_freeze(output, expected)
    study.save(source, {"text": "original"})
    (runtime / "control.py").write_text("changed")
    with pytest.raises(RuntimeError, match="Runtime archive"):
        study.validate_freeze(output, expected)
    (runtime / "control.py").write_text("original")
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: "changed")
    with pytest.raises(RuntimeError, match="environment changed"):
        study.validate_freeze(output, expected)


def test_cell_worker_runs_archived_guide_without_production_state_or_real_inference(tmp_path):
    """Exercise imports/config/transport/collection in a fresh Python process."""
    import subprocess

    output = tmp_path / "study"
    runtime = study.archive(study.STRICT_REVISION, output / "runtime" / "guide_first")
    protocol = _protocol()
    task = authored_tasks()[0]
    task["question_sha256"] = hashlib.sha256(task["question"].encode()).hexdigest()
    task["source_provenance"] = []
    entry = {"id": "test-cell", "task_id": task["id"], "arm": "guide_first", "repeat": 1, "seed": 1}
    protocol.update(tasks=[task], plan=[entry], runtimes={"guide_first": runtime},
                    tool_surface={"supplied_source": []})
    study.save(output / "protocol.json", protocol)
    wrapper = tmp_path / "fake_worker.py"
    wrapper.write_text('''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ollama
from training.scripts import run_autonomy_study as study
class FakeClient:
    def __init__(self, **kwargs): self._client = self
    def close(self): pass
    def list(self): return {"models": [{"model": study.MODEL, "digest": study.P1_DIGEST}]}
    def _request(self, *a): return {"version": "fake-test-only"}
    def chat(self, **kwargs):
        yield ollama.ChatResponse(model=study.MODEL, done=False, message=ollama.Message(
            role="assistant", content="  An unsupported suggestion remains the model answer.\\n"))
        yield ollama.ChatResponse(model=study.MODEL, done=True, eval_count=20, prompt_eval_count=100,
                                 message=ollama.Message(role="assistant", content=""))
ollama.Client = FakeClient
study.cell(Path(sys.argv[2]), "test-cell", sys.argv[3])
''', encoding="utf-8")
    proc = subprocess.run([study.sys.executable, "-X", "utf8", str(wrapper), str(study.ROOT), str(output),
                           study.file_hash(output / "protocol.json")], env=study.child_environment(),
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    row = study.read(output / "cells" / "test-cell.json")
    assert row["response"] == "  An unsupported suggestion remains the model answer.\n"
    assert row["error"] is None and row["runtime_unchanged"]
    assert len(row["turns"]) == 1 and row["tool_results"] == []
    assert row["isolated_config"]["mentor"]["enabled"] is False
    assert row["model_after"]["digest"] == study.P1_DIGEST


def test_production_invariants_read_only_and_checkpoint_mismatch_is_visible(tmp_path, monkeypatch):
    production, models = tmp_path / "production", tmp_path / "models"
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / "nestra" / "20b-p1"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("test manifest")
    monkeypatch.setattr(study, "P1_DIGEST", study.file_hash(manifest))
    gguf_hash = hashlib.sha256(b"test weights").hexdigest()
    gguf = models / "blobs" / ("sha256-" + gguf_hash)
    gguf.parent.mkdir()
    gguf.write_bytes(b"test weights")
    checkpoint = production / "training" / "output" / "checkpoint-11112" / "adapter.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"accepted inherited state")
    lineage = {"deployment": {"gguf_sha256": gguf_hash}, "training":
               {"output_directory": "training/output", "checkpoint_files":
                {"adapter.bin": {"sha256": study.file_hash(checkpoint)}}}}
    study.save(production / "training" / "lineage" / "package-1.json", lineage)
    memory = production / "localpilot-data" / "learning.sqlite3"
    memory.parent.mkdir()
    memory.write_bytes(b"test memory")
    (production / "localpilot.toml").write_text("owner config")
    monkeypatch.setattr(study.subprocess, "check_output", lambda *a, **kw: "head-or-status")
    before = study.tree_hashes(tmp_path)
    observed = study.production_invariants(production, models, {"lineage": lineage})
    assert observed["checkpoint_access"]["adapter.bin"] == "present"
    assert study.tree_hashes(tmp_path) == before
    checkpoint.write_bytes(b"mismatched state")
    with pytest.raises(RuntimeError, match="checkpoint differs"):
        study.production_invariants(production, models, {"lineage": lineage})
