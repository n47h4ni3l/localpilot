import json
import sqlite3
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.evolution import capability_task_id, legacy_capability_task_id, parse_capability_proposals
from localpilot.evolution_repository_review import (
    backlog_repository_context, objective_fingerprint, review_objective,
    review_sources_unchanged, validate_repository_review,
)
from localpilot.github_integration import CommandResult, MainSyncResult
from localpilot.learning import LearningMemory
from localpilot.resource import ResourceState
from localpilot.selfdev import CandidateTools, DeveloperModelSelection, SelfDeveloper


def task():
    return {"id": "capability-search-ranking", "title": "Improve existing search",
            "capability_target": "library search", "hypothesis": "A reranker will improve recall.",
            "observed_limitation": "Relevant passages are omitted.",
            "question": "Can a reranker improve existing retrieval?",
            "evidence": ["library implementation"], "source": "capability_discovery",
            "evaluation": {"metric": "recall", "baseline": "unmeasured current search",
                           "success_criterion": "raise recall without lowering precision",
                           "measurement_method": "same held-out queries before and after"}}


def repository(root):
    (root / "localpilot").mkdir(parents=True, exist_ok=True)
    (root / "localpilot/search.py").write_bytes(b"def search(query):\r\n    return query.split()\r\n")
    (root / "selfdev-backlog.json").write_text('{"tasks": []}')
    return {"localpilot/search.py", "selfdev-backlog.json"}


def payload():
    return {"repository_review": {
        "objective_kind": "improve_existing", "current_behavior": "Search splits a query into terms.",
        "remaining_gap": "Ranking quality has not been measured for held-out queries.",
        "proposed_change": "Compare a reranker using the same held-out query set.",
        "source_evidence": [{"path": "localpilot/search.py", "excerpt": "def search(query):\n    return query.split()"}],
        "change_plan": {"referenced_symbols": ["localpilot.search:search"],
                        "referenced_config_fields": [], "referenced_paths": ["localpilot/search.py"],
                        "required_test_contracts": [], "integration_points": ["localpilot.search:search"],
                        "expected_call_relationships": [], "planned_subsystems": [], "new_runtime_paths": []},
    }}


def proposal_data():
    value = task()
    value.update({"evolution_class": "improve_cognition", "alternatives": ["Lexical tuning", "Semantic reranking"],
                  "mission_alignment": "Improve reusable retrieval while preserving owner control.",
                  "current_frontier": "Existing retrieval has an unmeasured ranking gap.",
                  "why_high_leverage": "Many future tasks depend on accurate retrieval.",
                  "capability_unlocked": "Better evidence reuse across future tasks.",
                  "next_frontier": "Measure transfer to a second held-out task set."})
    return value


def validate(raw, root, tracked, inspected=None):
    prefixes = {path: path.read_text(encoding="utf-8") for path in
                (inspected if inspected is not None else {root / "localpilot/search.py"})}
    return validate_repository_review(json.dumps(raw), root=root, tracked_paths=tracked,
                                      inspected_sources=prefixes)


def developer(root, monkeypatch):
    tracked = repository(root)
    cfg = Config(); cfg.agent.data_dir = "data"
    dev = SelfDeveloper(cfg, root)
    monkeypatch.setattr(dev.github, "tracked_project_paths", lambda: tracked)
    monkeypatch.setattr(dev.github, "_run", lambda *args, **kwargs: CommandResult(True, "revision-1", "", 0))
    return dev


def test_existing_feature_is_a_baseline_not_a_completion_veto(tmp_path):
    tracked = repository(tmp_path)
    review = validate(payload(), tmp_path, tracked)
    assert review["status"] == "source_anchored"
    assert review["objective_kind"] == "improve_existing"
    assert not review["objective_completed"] and not review["performance_verified"]
    assert review_sources_unchanged(review, tmp_path)
    (tmp_path / "localpilot/search.py").write_text("def search(query):\n    return []\n")
    assert not review_sources_unchanged(review, tmp_path)


@pytest.mark.parametrize("case,code", [
    ("backlog", "not_repository_source"), ("circular_proposal", "not_repository_source"),
    ("uninspected", "source_not_inspected"), ("false_excerpt", "source_excerpt_not_inspected"),
    ("false_symbol", "nonexistent_api"), ("private_file", "not_repository_source"),
    ("traversal", "not_repository_source"),
])
def test_unverified_or_false_evidence_requires_research(tmp_path, case, code):
    tracked = repository(tmp_path)
    value = payload(); review = value["repository_review"]; inspected = {tmp_path / "localpilot/search.py"}
    if case == "backlog": review["source_evidence"] = [{"path": "selfdev-backlog.json", "excerpt": '{"tasks": []}'}]
    if case == "circular_proposal": review["source_evidence"][0]["path"] = "durable_memory/frontiers/another-proposal"
    if case == "uninspected": inspected = set()
    if case == "false_excerpt": review["source_evidence"][0]["excerpt"] = "return magic_knowledge_graph(query)"
    if case == "false_symbol": review["change_plan"]["referenced_symbols"] = ["localpilot.search:ImaginaryRetriever"]
    if case == "private_file":
        (tmp_path / "private.py").write_text("private material")
        review["source_evidence"] = [{"path": "private.py", "excerpt": "private material"}]
    if case == "traversal": review["source_evidence"][0]["path"] = "../localpilot/search.py"
    result = validate(value, tmp_path, tracked, inspected)
    assert result["status"] == "needs_research"
    assert code in {issue["code"] for issue in result["issues"]}


def test_backlog_reconciliation_preserves_rejection_and_improvement_opportunity():
    tasks = [{"id": "old", "title": "Build search", "status": "todo",
              "acceptance": ["Create localpilot/search.py and tests/test_search.py"]},
             {"id": "rejected", "status": "todo"}]
    result = backlog_repository_context(tasks, {"localpilot/search.py"}, {"old"}, {"rejected"}, set())
    assert result[0]["stored_status"] == "todo"
    assert result[0]["history_state"] == "completed_original_objective"
    assert result[0]["referenced_paths_present"] == ["localpilot/search.py"]
    assert result[0]["referenced_paths_absent"] == ["tests/test_search.py"]
    assert not result[0]["file_presence_proves_completion"]
    assert result[0]["existing_feature_can_be_improved"]
    assert result[1]["history_state"] == "owner_rejected_objective"


def test_cached_review_is_bound_to_objective_revision_and_current_source(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch); calls = []
    def stage(**kwargs):
        calls.append(kwargs)
        assert {fn.__name__ for fn in kwargs["functions"]} == {"list_project_files", "read_project_file"}
        kwargs["functions"][1]("localpilot/search.py")
        return json.dumps(payload())
    monkeypatch.setattr(dev, "_tool_stage", stage)
    first = review_objective(dev, task(), developer_model="fake", force=False)
    assert first["status"] == "source_anchored"
    assert review_objective(dev, task(), developer_model="fake", force=False) == first
    assert len(calls) == 1
    altered = task(); altered["observed_limitation"] = "A newly observed error affects empty queries."
    review_objective(dev, altered, developer_model="fake", force=False)
    assert len(calls) == 2
    monkeypatch.setattr(dev.github, "_run", lambda *args, **kwargs: CommandResult(True, "revision-2", "", 0))
    review_objective(dev, task(), developer_model="fake", force=False)
    assert len(calls) == 3
    (tmp_path / "localpilot/search.py").write_text("def search(query):\n    return []\n")
    result = review_objective(dev, task(), developer_model="fake", force=False)
    assert result["status"] == "needs_research" and len(calls) == 5
    # Every attempt is preserved, and the final failure is not retried forever.
    review_objective(dev, task(), developer_model="fake", force=False)
    assert len(calls) == 5
    with sqlite3.connect(dev.memory.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evolution_repository_reviews").fetchone()[0] == 5


def test_preparation_is_blocked_before_cycle_creation(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch)
    monkeypatch.setattr(dev.github, "sync_trusted_main", lambda: MainSyncResult(True, False, "current"))
    monkeypatch.setattr(dev.governor, "sample", lambda: ResourceState(1000, 5, 20, True, "idle", True, True))
    monkeypatch.setattr(dev.governor, "apply_process_priority", lambda idle: None)
    monkeypatch.setattr(dev, "_reconcile_candidates", lambda: None)
    monkeypatch.setattr(dev, "_repair_local_candidate", lambda **kwargs: None)
    monkeypatch.setattr(dev, "_repair_failed_candidate", lambda **kwargs: None)
    monkeypatch.setattr(dev, "_load_next_task", lambda: task())
    monkeypatch.setattr(dev, "_select_developer_model", lambda: DeveloperModelSelection("fake", 1, 1, "fits"))
    monkeypatch.setattr(dev, "_tool_stage", lambda **kwargs: json.dumps(payload()))  # source was never inspected
    monkeypatch.setattr(dev, "_candidate_workspace", lambda *args: pytest.fail("checkout must wait for evidence"))
    result = dev._run_once(force=True)
    assert result.status == "needs_research"
    assert dev.memory.latest_experiment() is None
    with sqlite3.connect(dev.memory.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM development_cycles").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM candidate_preparations").fetchone()[0] == 0


def test_held_backlog_objective_can_research_after_revision_changes(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch); original = task()
    monkeypatch.setattr(dev, "_backlog_tasks", lambda: [original])
    dev.memory.record_repository_review(original["id"], objective_fingerprint(original), "revision-1",
                                        {"status": "needs_research"})
    assert dev._load_next_task() is None
    monkeypatch.setattr(dev.github, "_run", lambda *args, **kwargs: CommandResult(True, "revision-2", "", 0))
    assert dev._load_next_task()["id"] == original["id"]


def test_frontier_proposals_are_not_reported_as_verified_capabilities(tmp_path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    memory.record_experiment(task())
    frontier = memory.discovery_context()["frontiers"][0]
    assert frontier["experiment_status"] == "proposed"
    assert frontier["evidence_state"] == "unverified_objective"


def test_readonly_source_listing_does_not_scan_private_trees(tmp_path, monkeypatch):
    tools = CandidateTools(tmp_path, readable_paths={"localpilot/search.py", "tests/test_search.py"})
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("allowlisted listing must avoid private tree traversal"))
    assert tools.list_project_files() == "localpilot/search.py\ntests/test_search.py"


def test_objective_identity_distinguishes_a_new_gap_and_preserves_legacy_lookup():
    original = parse_capability_proposals(json.dumps(proposal_data()))[0]
    changed = proposal_data(); changed["observed_limitation"] = "Queries fail because of a timeout."
    improvement = parse_capability_proposals(json.dumps(changed))[0]
    assert legacy_capability_task_id(original) == "capability-library-search-e516cf0b5e"
    assert legacy_capability_task_id(improvement) == legacy_capability_task_id(original)
    assert capability_task_id(improvement) != capability_task_id(original)
    cosmetic = proposal_data(); cosmetic["title"] = "A new label"; cosmetic["evaluation"]["baseline"] = "67%"
    assert capability_task_id(parse_capability_proposals(json.dumps(cosmetic))[0]) == capability_task_id(original)


def test_same_legacy_objective_cannot_reenter_under_the_new_identity(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch)
    proposal = parse_capability_proposals(json.dumps(proposal_data()))[0]
    original = proposal.task(legacy_capability_task_id(proposal))
    dev.memory.record_experiment(original)
    dev.memory.update_experiment_outcome(original["id"], status="validated")
    monkeypatch.setattr(dev, "_tool_stage", lambda **kwargs: json.dumps({"proposals": [proposal_data()]}))
    with pytest.raises(ValueError, match="previously attempted hypotheses"):
        dev._discover_capability_task(developer_model="fake", force=True)
    assert dev.memory.experiment_for_task(capability_task_id(proposal)) is None


def test_new_concrete_limitation_is_not_a_coarse_legacy_identity_collision(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch)
    value = proposal_data(); value["observed_limitation"] = "Planning requests fail because of cache failures."
    proposal = parse_capability_proposals(json.dumps(value))[0]
    old = proposal.task(legacy_capability_task_id(proposal))
    dev.memory.record_experiment(old)
    dev.memory.update_experiment_outcome(old["id"], status="failed")
    before = dev.memory.experiment_for_task(old["id"])
    value["observed_limitation"] = "Planning requests fail because of timeout failures."
    new = parse_capability_proposals(json.dumps(value))[0]
    monkeypatch.setattr(dev, "_tool_stage", lambda **kwargs: json.dumps({"proposals": [value]}))
    selected = dev._discover_capability_task(developer_model="fake", force=True)
    assert selected["id"] == capability_task_id(new) and selected["id"] != old["id"]
    assert dev.memory.experiment_for_task(old["id"]) == before


def test_completed_history_outside_the_queue_blocks_a_rephrased_objective(tmp_path, monkeypatch):
    dev = developer(tmp_path, monkeypatch)
    value = proposal_data(); proposal = parse_capability_proposals(json.dumps(value))[0]
    historical = proposal.task("legacy-objective-no-longer-in-queue")
    dev.memory.record_experiment(historical)
    dev.memory.update_experiment_outcome(historical["id"], status="rejected_by_human")
    value["hypothesis"] = "A semantic reranker could improve recall."
    new = parse_capability_proposals(json.dumps(value))[0]
    assert legacy_capability_task_id(new) != legacy_capability_task_id(proposal)
    monkeypatch.setattr(dev, "_tool_stage", lambda **kwargs: json.dumps({"proposals": [value]}))
    with pytest.raises(ValueError, match="previously attempted hypotheses"):
        dev._discover_capability_task(developer_model="fake", force=True)
    assert dev.memory.experiment_for_task(capability_task_id(new)) is None


def test_source_quote_must_have_been_returned_not_merely_exist_later_in_file(tmp_path):
    tracked = repository(tmp_path)
    path = tmp_path / "localpilot/search.py"
    path.write_text("# An unrelated source comment.\n" * 100 + "def search(query):\n    return query.split()\n")
    tools = CandidateTools(tmp_path, readable_paths=tracked)
    returned = tools.read_project_file("localpilot/search.py", max_chars=1000)
    assert "def search" not in returned
    result = validate_repository_review(json.dumps(payload()), root=tmp_path, tracked_paths=tracked,
                                         inspected_sources=tools.source_read_prefixes)
    assert result["status"] == "needs_research"
    assert "source_excerpt_not_inspected" in {issue["code"] for issue in result["issues"]}
    tools.read_project_file("localpilot/search.py", max_chars=5000)
    assert validate_repository_review(json.dumps(payload()), root=tmp_path, tracked_paths=tracked,
                                      inspected_sources=tools.source_read_prefixes)["status"] == "source_anchored"
