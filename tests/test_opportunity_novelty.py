from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from localpilot.evolution_orchestrator import OpportunityLedger, opportunity_similarity


def _retrieval_task(task_id: str = "retrieval-lexical-weights") -> dict:
    work = (
        "multi-step tasks requiring verified operational guidance, recovery constraints, "
        "causal explanations, durable decisions, and remembered user preferences"
    )
    return {
        "id": task_id,
        "title": "Tune memory retrieval weights",
        "capability_target": "Reliable retrieval from durable learning memory",
        "question": f"Can lexical retrieval weights reduce missed relevant lessons for {work}?",
        "observed_limitation": f"Relevant stored lessons are sometimes omitted from planning during {work}.",
        "hypothesis": "Tuning lexical retrieval weights reduces missed relevant lessons.",
        "evidence": ["localpilot/learning.py owns existing memory retrieval."],
        "evaluation": {
            "metric": "relevant lesson recall",
            "baseline": "Measure current recall on the held-out planning queries.",
            "success_criterion": "Improve recall without reducing precision.",
            "measurement_method": "Run the same held-out queries before and after the change.",
        },
    }


def _reranker_task(task_id: str = "retrieval-semantic-reranker") -> dict:
    task = _retrieval_task(task_id)
    task["title"] = "Evaluate a semantic reranker for existing memory retrieval"
    task["question"] = task["question"].replace("lexical retrieval weights", "a semantic reranker")
    task["hypothesis"] = "Introducing a semantic reranker stage reduces missed relevant lessons."
    return task


def test_existing_feature_improvement_can_use_the_same_benchmark(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    lexical = _retrieval_task()
    reranker = _reranker_task()
    assert lexical["capability_target"] == reranker["capability_target"]
    assert lexical["evaluation"] == reranker["evaluation"]

    assert ledger.enqueue(lexical, score=7)
    assert opportunity_similarity(lexical, reranker) < ledger.similarity_threshold
    assert ledger.enqueue(reranker, score=10)
    assert OpportunityLedger(ledger.path).next_task()["id"] == reranker["id"]


@pytest.mark.parametrize("old_algorithm,new_algorithm", [("BM25", "TFIDF"), ("HTTP2", "HTTP3")])
def test_one_different_algorithm_under_the_same_benchmark_is_eligible(
    tmp_path: Path, old_algorithm: str, new_algorithm: str
):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task(f"retrieval-{old_algorithm}")
    new = _retrieval_task(f"retrieval-{new_algorithm}")
    old["hypothesis"] = f"Tuning {old_algorithm} weights reduces missed relevant lessons."
    new["hypothesis"] = f"Tuning {new_algorithm} weights reduces missed relevant lessons."

    assert old["evaluation"] == new["evaluation"]
    assert ledger.enqueue(old, score=7)
    assert ledger.enqueue(new, score=9)


def test_one_different_concrete_failure_in_the_same_feature_is_eligible(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task("retrieval-cache-failure")
    new = _retrieval_task("retrieval-timeout-failure")
    old["observed_limitation"] += " Planning requests fail because of cache failures."
    new["observed_limitation"] += " Planning requests fail because of timeout failures."

    assert old["evaluation"] == new["evaluation"]
    assert ledger.enqueue(old, score=7)
    assert ledger.enqueue(new, score=9)


def test_numeric_baseline_is_a_complete_measurement_contract(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    new = _reranker_task()
    old["evaluation"]["baseline"] = "62%"
    new["evaluation"]["baseline"] = "62%"

    assert ledger.enqueue(old, score=7)
    assert ledger.enqueue(new, score=9)


def test_feature_name_alone_never_establishes_a_duplicate():
    old = {"capability_target": "Reliable durable memory retrieval"}
    new = {"capability_target": "Reliable durable memory retrieval"}

    assert opportunity_similarity(old, new) < 0.5


def test_newly_observed_limitation_of_the_same_feature_remains_eligible(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    new = _retrieval_task("retrieval-stale-lessons")
    new["observed_limitation"] = "Retrieved lessons contain obsolete entries after source corrections."
    new["evaluation"] = {
        "metric": "obsolete lesson selection rate",
        "baseline": "Record obsolete entries returned after correcting a lesson source.",
        "success_criterion": "Return corrected lessons without selecting obsolete entries.",
        "measurement_method": "Revise stored lessons and replay the affected planning requests.",
    }

    assert ledger.enqueue(old, score=7)
    assert ledger.enqueue(new, score=9)


def test_paraphrased_objective_is_still_a_duplicate(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    paraphrase = _retrieval_task("renamed-retrieval-experiment")
    paraphrase["title"] = "Reduce omitted lessons in memory planning"
    paraphrase["question"] = "Can adjusted lexical retrieval weights cut omitted applicable lessons?"
    paraphrase["observed_limitation"] = "Applicable durable lessons are sometimes missed from planning."
    paraphrase["hypothesis"] = "Adjusting lexical retrieval weights will cut omitted applicable lessons."

    assert ledger.enqueue(old, score=7)
    assert opportunity_similarity(old, paraphrase) >= ledger.similarity_threshold
    assert not ledger.enqueue(paraphrase, score=12)


def test_exact_identity_blocks_even_a_different_measured_mechanism(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    new = _reranker_task(old["id"])

    assert ledger.enqueue(old, score=7)
    duplicate, score = ledger.duplicate(new)
    assert duplicate["task"]["id"] == old["id"]
    assert score == 1.0
    assert not ledger.enqueue(new, score=12)


@pytest.mark.parametrize("cosmetic_change", ["title", "target", "metric", "metadata", "baseline"])
def test_cosmetic_changes_do_not_create_a_new_objective(tmp_path: Path, cosmetic_change: str):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    renamed = _retrieval_task("another-task-id")
    if cosmetic_change == "title":
        renamed["title"] = "Completely new capability project"
    elif cosmetic_change == "target":
        renamed["capability_target"] = "Planning lesson retrieval reliability"
    elif cosmetic_change == "metric":
        renamed["evaluation"]["metric"] = "durable information usefulness score"
    elif cosmetic_change == "baseline":
        renamed["evaluation"]["baseline"] = "Recall measured today is 62 percent instead of 51 percent."
    else:
        renamed["created_at"] = "2026-10-14T12:00:00Z"
        renamed["revision"] = "c2a74bc9"
        for key in ("question", "observed_limitation", "hypothesis"):
            renamed[key] += " revision: c2a74bc9 timestamp: 2026-10-14T12:00:00Z version: v2"

    assert ledger.enqueue(old, score=7)
    assert not ledger.enqueue(renamed, score=12)


@pytest.mark.parametrize("missing_field", ["question", "observed_limitation", "hypothesis", "evaluation"])
@pytest.mark.parametrize("legacy_side", ["existing", "candidate"])
def test_incomplete_legacy_objectives_remain_conservative(
    tmp_path: Path, missing_field: str, legacy_side: str
):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    new = _reranker_task()
    del (old if legacy_side == "existing" else new)[missing_field]

    assert ledger.enqueue(old, score=7)
    assert not ledger.enqueue(new, score=12)


@pytest.mark.parametrize("status", ["rejected", "failed", "completed"])
def test_historical_outcomes_still_block_the_same_objective(tmp_path: Path, status: str):
    ledger = OpportunityLedger(tmp_path / "opportunities.json")
    old = _retrieval_task()
    assert ledger.enqueue(old, score=7)
    ledger.update(old["id"], status, "Preserve the measured outcome.")

    assert not ledger.enqueue(_retrieval_task("renamed-history"), score=12)
    assert ledger.enqueue(_reranker_task(), score=11)
    entries = ledger.context()
    assert entries[0]["status"] == status
    assert entries[0]["outcome"] == "Preserve the measured outcome."


def test_distinct_objectives_do_not_expand_the_bounded_queue(tmp_path: Path):
    ledger = OpportunityLedger(tmp_path / "opportunities.json", max_entries=2)
    first = _retrieval_task()
    second = _reranker_task()
    third = deepcopy(first)
    third["id"] = "retrieval-stale-source"
    third["observed_limitation"] = "Retrieved entries contain obsolete content after source corrections."

    assert ledger.enqueue(first, score=7)
    assert ledger.enqueue(second, score=10)
    assert ledger.enqueue(third, score=9)
    assert [entry["task_id"] for entry in ledger.context()] == [second["id"], third["id"]]
