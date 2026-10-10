"""Diagnoses keep exhausted cells and drafts; no scoring or source mutations."""
import json

import pytest

from training.scripts.diagnose_autonomy_traces import Evidence, diagnose, sha


def evidence_at(tmp_path):
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return sha(path.read_bytes())

    protocol = {
        "runtimes": {"strict": {"revision": "frozen-control", "files": {}}},
        "source_files": {}, "budgets": {"model_calls": 1, "generated_tokens": 24576, "tool_calls": 10},
        "plan": [{"id": "empty-direct", "arm": "direct"}, {"id": "cap-strict", "arm": "strict"}],
    }
    protocol_hash = put("protocol.json", protocol)
    put("freeze.json", {"protocol_sha256": protocol_hash})
    for cell_id, arm, message, tokens, reason, error in (
        ("empty-direct", "direct", {"thinking": "private reasoning"}, 8192, "length", None),
        ("cap-strict", "strict", {"content": "Model-authored draft"}, 200, "stop",
         {"type": "StudyBudgetExceeded", "message": "Per-cell inference resource ceiling reached"}),
    ):
        chunks_name = cell_id + "--call-1.jsonl"
        put("cells/" + chunks_name, {"chunk": {"message": message}})
        put("cells/" + cell_id + ".json", {
            "id": cell_id, "arm": arm, "stratum": "supplied_source", "response": "", "error": error,
            "metering": {"calls": 1, "tools": 0, "generated_tokens_reserved_or_observed": tokens},
            "turns": [{"chunks_file": chunks_name,
                       "requested": {"think": "high", "options": {"num_predict": 3072}},
                       "effective": {"think": "high", "options": {"num_predict": 8192}},
                       "terminal": {"done_reason": reason, "eval_count": tokens, "prompt_eval_count": 800}}],
            "tool_results": [],
        })
    return Evidence(tmp_path)


def test_empty_high_reasoning_and_exhausted_draft_remain_distinct(tmp_path):
    evidence = evidence_at(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = diagnose(evidence)
    assert report["totals"]["cells"] == 2
    assert report["totals"]["nonempty"] == 0
    assert report["totals"]["ceilings"] == {"model_calls": 1}
    assert report["totals"]["reasoning_only_length_calls"] == 1
    empty, exhausted = report["cells"]
    assert empty["error"] is None and empty["ceiling"] is None
    assert exhausted["nonempty_draft_call_numbers"] == [1]
    call = exhausted["calls"][0]
    assert call["requested_num_predict"] == 3072 and call["effective_num_predict"] == 8192
    assert "private reasoning" not in json.dumps(report)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_changed_protocol_is_not_diagnosed_as_frozen_evidence(tmp_path):
    evidence = evidence_at(tmp_path)
    with (tmp_path / "protocol.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        diagnose(evidence)


def test_missing_token_usage_remains_unknown(tmp_path):
    evidence = evidence_at(tmp_path)
    path = tmp_path / "cells/empty-direct.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["turns"][0]["terminal"].pop("eval_count")
    path.write_text(json.dumps(row), encoding="utf-8")
    report = diagnose(evidence)
    assert report["totals"]["unknown_token_calls"] == 1
    assert report["cells"][0]["calls"][0]["generated_tokens"] is None


@pytest.mark.parametrize("path", ["../outside", "/outside", "C:outside", "runtime\\outside"])
def test_untrusted_evidence_paths_cannot_read_outside_source(tmp_path, path):
    evidence = Evidence(tmp_path)
    with pytest.raises(ValueError, match="Invalid evidence path"):
        evidence.bytes(path)
