"""Inspect exact source before accepting numeric or cross-file implementation claims."""
import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.source_answer_checks import (
    inspected_repository_lines, repository_answer_risks,
)


def read(path="localpilot/tools/repository.py", first=40):
    return {
        "role": "tool", "tool_name": "read_repository_file",
        "content": (
            f"Repository file: {path} lines {first}-{first + 3}\n"
            f"{first}: def search_repository(query, max_results=40):\n"
            f"{first+1}:     max_results = max(1, min(int(max_results), 100))\n"
            f"{first+2}:     return results[:max_results]\n"
            f"{first+3}:     # Bounds are local to this function"
        ),
    }


def test_named_numeric_constant_must_be_on_the_same_inspected_source_line():
    prompt = "Inspect the repository and cite the actual max_results limit in source code."
    evidence = [read()]
    correct = "The inspected max_results is 100 (localpilot/tools/repository.py:41)."
    wrong = "The inspected max_results is 500 (localpilot/tools/repository.py:41)."
    assert repository_answer_risks(prompt, correct, evidence) == ()
    assert repository_answer_risks(prompt, wrong, evidence) == (
        "named_numeric_constant_not_supported_by_source",
    )
    assert repository_answer_risks(
        prompt, "The inspected max_results is 100 (localpilot/tools/repository.py:999).", evidence
    ) == ("repository_line_citation_unverified",)
    # A number from another file or a different line is not source support.
    separate = {
        "role": "tool", "tool_name": "read_repository_file",
        "content": "Repository file: docs/notes.md lines 1-1\n1: An unrelated example uses 500.",
    }
    assert "named_numeric_constant_not_supported_by_source" in repository_answer_risks(
        prompt, wrong, [*evidence, separate]
    )


def test_location_request_requires_observed_line_locator():
    prompt = "Explain from the repository and cite implementation locations."
    evidence = [read()]
    assert repository_answer_risks(prompt, "max_results is bounded by the inspected source.", evidence) == (
        "requested_repository_line_citations_missing",
    )
    assert repository_answer_risks(
        prompt, "The inspected bound appears in localpilot/tools/repository.py:41.", evidence
    ) == ()
    assert repository_answer_risks(
        prompt, "The implementation remains unresolved because a required file was not inspected.", evidence
    ) == ()


def test_inspected_numbered_search_results_count_as_verifiable_locations():
    result = {
        "role": "tool", "tool_name": "search_repository",
        "content": "Repository search: 'max_results'\nlocalpilot/tools/repository.py:41: max_results = max(1, min(int(max_results), 100))",
    }
    index = inspected_repository_lines([result])
    assert index["localpilot/tools/repository.py"][41].endswith("100))")
    prompt = "Cite the exact repository value and line of max_results."
    assert repository_answer_risks(
        prompt, "max_results is 100 (localpilot/tools/repository.py:41).", [result]
    ) == ()


def test_complete_cross_module_trace_requires_observed_hops_not_just_one_file():
    prompt = "Trace the integration path across modules in this repository."
    one = [read()]
    assert repository_answer_risks(
        prompt, "I verified the complete call chain.", one
    ) == ("cross_module_completeness_not_established",)
    assert repository_answer_risks(
        prompt, "I inspected only localpilot/tools/repository.py:41; other links remain unverified.", one
    ) == ()
    other = {
        "role": "tool", "tool_name": "read_repository_file",
        "content": "Repository file: localpilot/agent.py lines 90-90\n90: from localpilot.tools.repository import RepositoryReader",
    }
    assert repository_answer_risks(
        prompt,
        "I verified the complete call chain from localpilot/agent.py:90 to localpilot/tools/repository.py:41.",
        [*one, other],
    ) == ()


def test_explicit_hypothetical_numeric_premise_is_not_reported_as_live_repo_mismatch():
    prompt = "Suppose max_results is 500 in a hypothetical repository. Discuss the consequence."
    assert repository_answer_risks(
        prompt, "Under that premise, max_results is 500; that is not a live observation.",
        [read()],
    ) == ()


def test_non_repo_and_tool_failure_do_not_activate_numeric_source_binding():
    assert repository_answer_risks(
        "What manufacturer's filament drying temperature should I use?",
        "Drying temperature is 80 C.",
        [read()],
    ) == ()
    failed = {
        "role": "tool", "tool_name": "read_repository_file",
        "content": "Tool error: Access denied\nRepository file: localpilot/tools/repository.py lines 40-40\n40: LIMIT = 900",
    }
    assert inspected_repository_lines([failed]) == {}
    assert repository_answer_risks(
        "Inspect repository and cite source line.",
        "LIMIT = 900.",
        [failed],
    ) == ()


@pytest.mark.parametrize("text", [
    "For our example, this is not established: max_results is 900.",
    "The source does not establish that max_results is 500.",
    "If max_results is 500, that would be a hypothetical rather than observed.",
    "You should not set max_results = 500 without checking the source.",
])
def test_explicitly_negative_claims_not_rewritten_to_positive(text):
    assert "named_numeric_constant_not_supported_by_source" not in repository_answer_risks(
        "Inspect repository behavior.", text, [read()]
    )


def test_agent_contextual_gate_applies_checks_to_delivered_answers():
    prompt = "Inspect repository source code, explain max_results and cite exact implementation locations."
    bad = "max_results is 500 (localpilot/tools/repository.py:41)."
    safe = "max_results is 100 (localpilot/tools/repository.py:41)."
    assert "named_numeric_constant_not_supported_by_source" in LocalPilotAgent._contextual_evidence_risks(
        prompt, bad, frozenset({"read_repository_file"}), [read()]
    )
    assert "named_numeric_constant_not_supported_by_source" not in LocalPilotAgent._contextual_evidence_risks(
        prompt, safe, frozenset({"read_repository_file"}), [read()]
    )
