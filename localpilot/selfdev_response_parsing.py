"""Change-plan parsing/application and model-response-outcome parsing for
the self-development pipeline.

Extracted verbatim (no logic changes) from selfdev.py: parsing a model's
JSON change plan or grounding-claim manifest (_json_object,
parse_change_plan, parse_grounding_plan, PlannedChange, ChangePlan,
_GROUNDING_PLAN_FIELDS), applying a parsed plan through CandidateTools
(apply_change_plan), building bounded read/repair context
(build_read_context, build_static_repair_context, StaticRepairResult),
and turning a model's free-text response into a structured outcome
(the six small statics carried over from SelfDeveloper as shims:
_outcome, _research_handoff, _checkpoint_outcome, _evaluation_report,
_candidate_pr_body, _check_summary).

apply_change_plan/build_read_context/build_static_repair_context take a
CandidateTools instance -- CandidateTools stays in selfdev.py, so (as in
selfdev_results.py) that's a TYPE_CHECKING-guarded import. The original
code already used quoted "CandidateTools" forward references for these
three, so this was already safe at runtime; the guarded import just adds
proper static-analysis resolution, consistent with selfdev_results.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from localpilot.evolution import evolution_status_fields, normalize_evolution_task

if TYPE_CHECKING:
    from localpilot.selfdev import CandidateTools


@dataclass(frozen=True, slots=True)
class PlannedChange:
    path: str
    content: str
    reason: str


@dataclass(frozen=True, slots=True)
class ChangePlan:
    summary: str
    reusable_lesson: str
    changes: tuple[PlannedChange, ...]


@dataclass(frozen=True, slots=True)
class StaticRepairResult:
    check_result: str
    passed: bool
    final_text: str
    attempts_used: int


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object.")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Change plan must be a JSON object.")
    return value


def parse_change_plan(text: str, max_files: int = 8) -> ChangePlan:
    value = _json_object(text)
    changes = value.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Change plan must contain at least one change.")
    if len(changes) > max_files:
        raise ValueError("Change plan exceeds the candidate file limit.")
    parsed: list[PlannedChange] = []
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError("Each planned change must be an object.")
        path, content = item.get("path"), item.get("content")
        if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
            raise ValueError("Each planned change requires string path and content fields.")
        parsed.append(PlannedChange(path.strip(), content, str(item.get("reason") or "")))
    return ChangePlan(
        str(value.get("summary") or "Structured fallback plan applied."),
        str(value.get("reusable_lesson") or "Use a structured write plan when direct tool editing stalls."),
        tuple(parsed),
    )

_GROUNDING_PLAN_FIELDS = (
    "referenced_symbols",
    "referenced_config_fields",
    "referenced_paths",
    "required_test_contracts",
    "integration_points",
    "expected_call_relationships",
    "planned_subsystems",
    "new_runtime_paths",
)


def parse_grounding_plan(text: str) -> dict[str, list[Any]]:
    """Parse the repository-claim manifest produced before implementation."""
    value = _json_object(text)
    candidate = value.get("change_plan", value)
    if not isinstance(candidate, dict):
        raise ValueError("Grounding change_plan must be a JSON object.")
    plan: dict[str, list[Any]] = {}
    for field in _GROUNDING_PLAN_FIELDS:
        items = candidate.get(field)
        if not isinstance(items, list):
            raise ValueError(f"Grounding change_plan field {field!r} must be a list.")
        plan[field] = items[:50]
    return plan


def apply_change_plan(plan: ChangePlan, tools: "CandidateTools") -> list[str]:
    """Preflight the complete fallback plan, then apply it through CandidateTools.

    Validation is deliberately a separate first pass. Deterministic safety
    failures therefore reject the whole plan before any file is written. An
    unexpected filesystem failure during the write pass is recorded by
    CandidateTools and blocks candidate delivery for the current cycle.
    """
    if len(plan.changes) > tools.max_files:
        raise ValueError("Change plan exceeds the candidate file limit.")
    tools.validate_write_plan(plan.changes)
    return [tools.write_project_file(change.path, change.content) for change in plan.changes]


def build_read_context(
    tools: "CandidateTools",
    *,
    max_files: int = 8,
    max_chars_per_file: int = 16000,
) -> str:
    """Return bounded source context only for files already inspected through CandidateTools."""
    rows: list[str] = []
    paths = sorted(
        tools.files_read,
        key=lambda item: item.relative_to(tools.workspace).as_posix(),
    )

    for file_path in paths[:max_files]:
        try:
            relative = file_path.relative_to(tools.workspace).as_posix()
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )[:max_chars_per_file]
        except Exception:
            continue

        rows.append(f"--- {relative} ---\n{content}")

    return "\n\n".join(rows) or "(No candidate files were successfully inspected.)"


def build_static_repair_context(
    tools: "CandidateTools",
    check_result: str,
    *,
    max_files: int = 8,
    max_chars_per_file: int = 5000,
) -> str:
    """Build bounded failure, diff, and changed-file feedback for a repair."""
    rows: list[str] = []
    paths = sorted(
        tools.files_written,
        key=lambda item: item.relative_to(tools.workspace).as_posix(),
    )
    for file_path in paths[:max_files]:
        relative = file_path.relative_to(tools.workspace).as_posix()
        content = tools.read_project_file(relative, max_chars=max_chars_per_file)
        rows.append(f"--- {relative} ---\n{content}")

    changed_files = "\n\n".join(rows) or "(No changed file content is available.)"
    candidate_diff = tools.show_candidate_diff()[-16000:]
    return (
        f"Static-check failure:\n{check_result[:8000]}\n\n"
        f"Candidate diff:\n{candidate_diff}\n\n"
        f"Changed candidate files:\n{changed_files}"
    )


def _check_summary(check_result: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in str(check_result).splitlines() if line.strip()]
    if not lines:
        return "not run", []
    first = lines[0].lower()
    if first.startswith("static_checks=passed"):
        return "passed", []
    if first.startswith("static_checks=failed"):
        return "failed", lines[1:31]
    if "disabled" in first:
        return "disabled", []
    return first[:100], lines[1:31]


def _outcome(text: str, default_lesson: str) -> tuple[str, str]:
    try:
        value = _json_object(text)
    except (ValueError, json.JSONDecodeError):
        return (text.strip() or "Candidate cycle completed.")[:4000], default_lesson
    return (
        str(value.get("summary") or "Candidate cycle completed.")[:4000],
        str(value.get("reusable_lesson") or default_lesson)[:2000],
    )


def _research_handoff(
    text: str,
) -> tuple[list[str], list[str], list[str], str]:
    """Accept only explicit reviewable facts for durable research context."""
    try:
        value = _json_object(text)
    except (ValueError, json.JSONDecodeError):
        return (
            ["Read-only repository research completed; use the recorded inspected paths."],
            [],
            ["The research response was not a valid structured handoff."],
            "Re-inspect the recorded paths before implementation if more detail is needed.",
        )

    def strings(name: str, limit: int = 20) -> list[str]:
        items = value.get(name)
        if not isinstance(items, list):
            return []
        return [str(item)[:1000] for item in items[:limit] if isinstance(item, str) and item.strip()]

    findings = strings("findings") or [
        "Read-only repository research completed; use the recorded inspected paths."
    ]
    return (
        findings,
        strings("decisions"),
        strings("unresolved_questions"),
        str(value.get("next_action") or "Implement the focused task.")[:1000],
    )


def _checkpoint_outcome(text: str, default_lesson: str) -> tuple[str, str]:
    """Never place an unstructured model response in the durable checkpoint."""
    try:
        value = _json_object(text)
    except (ValueError, json.JSONDecodeError):
        return (
            "Implementation stage completed; inspect the verified candidate diff.",
            default_lesson,
        )
    return (
        str(value.get("summary") or "Implementation stage completed.")[:1000],
        str(value.get("reusable_lesson") or default_lesson)[:1000],
    )


def _evaluation_report(text: str, task: dict[str, Any]) -> dict[str, str]:
    plan = normalize_evolution_task(task)["evaluation"]
    report: dict[str, Any] = {}
    try:
        value = _json_object(text)
        candidate = value.get("evaluation_evidence") or value.get("evaluation")
        if isinstance(candidate, dict):
            report = candidate
    except (ValueError, json.JSONDecodeError):
        pass
    result = str(report.get("result") or "unmeasured").strip().lower()
    if result not in {"improved", "no_change", "regressed", "inconclusive", "pending_ci"}:
        result = "unmeasured"
    return {
        "metric": str(report.get("metric") or plan["metric"])[:1000],
        "baseline_evidence": str(report.get("baseline_evidence") or plan["baseline"])[:2000],
        "candidate_evidence": str(report.get("candidate_evidence") or "")[:2000],
        "result": result,
        "measurement_artifact": str(report.get("measurement_artifact") or "")[:1000],
    }


def _candidate_pr_body(
    task: dict[str, Any],
    report: dict[str, str],
    check_result: str,
) -> str:
    fields = evolution_status_fields(task)
    return (
        "## Capability-growth experiment\n\n"
        f"- Evolution class: {fields['evolution_class']}\n"
        f"- Capability target: {fields['capability_target']}\n"
        f"- Research question: {task['question']}\n"
        f"- Hypothesis: {fields['hypothesis']}\n"
        f"- Evaluation plan: {fields['evaluation_plan']}\n"
        f"- Baseline evidence: {report['baseline_evidence']}\n"
        f"- Candidate evidence: {report['candidate_evidence'] or 'Pending GitHub CI evaluation'}\n"
        f"- Current evaluation outcome: {report['result']}\n"
        f"- Measurement artifact: {report['measurement_artifact'] or 'Defined by the candidate/CI plan'}\n\n"
        "## Safety boundary\n\n"
        "This is an isolated candidate for human review. It was not executed locally and cannot merge or "
        "promote itself. Reviewer-controlled tests and all existing safety/resource gates remain authoritative.\n\n"
        f"## Local static validation\n\n````text\n{check_result[:4000]}\n````\n"
    )
