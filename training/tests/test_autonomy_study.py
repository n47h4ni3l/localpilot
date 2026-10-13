from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from training.scripts import autonomy_study_artifacts as study
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


def _protocol():
    return {"model": study.MODEL, "model_digest": study.P1_DIGEST, "think": "high",
            "options": study.OPTIONS, "budgets": study.BUDGETS, "keep_alive": "30m",
            "boundary": study.BOUNDARY, "tasks": [], "plan": [], "review":
            {"dimensions": ["accuracy", "utility", "unsafe_advice"]}}


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
    sources = [json.loads(s) for s in (bundle / "observed-sources.jsonl").read_text().splitlines()]
    assert len(sources) == 48
    assert all(set(s) == {"card_id", "public_reads"} for s in sources)
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


def test_blind_review_includes_frozen_reference_and_observed_read_without_arm_metadata(tmp_path):
    root, protocol = _saved_study(tmp_path)
    source = root / "sources" / "primary.json"
    study.save(source, {"url": "https://example.org", "text": "frozen primary reference"})
    protocol["source_files"] = {"primary": study.file_hash(source)}
    study.save(root / "protocol.json", protocol)
    cell_path = next((root / "cells").glob("*.json"))
    row = study.read(cell_path)
    row["tool_results"] = [{"tool": "fetch_public_https", "started_at": "arm-order-clue",
                           "result": "exact bounded public evidence", "result_sha256": "hash"}]
    study.save(cell_path, row)
    bundle = tmp_path / "blind"
    review.cards(root, bundle, tmp_path / "key.json")
    assert study.read(bundle / "references" / "primary.json")["text"] == "frozen primary reference"
    observed = (bundle / "observed-sources.jsonl").read_text()
    assert "exact bounded public evidence" in observed
    assert "arm-order-clue" not in observed and "fetch_public_https" not in observed
