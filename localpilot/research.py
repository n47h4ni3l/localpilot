from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


RESEARCH_NOTEBOOK_TOOL = "update_research_notebook"
_REFERENCE = re.compile(r"\b(?:obs|result)-\d{3,}\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def update_research_notebook(
    verified_fact_pointers: list[str],
    unresolved_questions: list[str],
    inspected_observation_ids: list[str],
    unresolved_fact: str,
    why_current_evidence_is_insufficient: str,
    proposed_tool: str,
    proposed_arguments_json: str,
    result_that_would_change_the_conclusion: str,
    new_hypothesis: str = "",
) -> str:
    """Record a transient evidence-pointer notebook and one information-gain checkpoint.

    This is a planning/index operation, not an evidence source. Every factual notebook entry
    must cite an existing obs-NNN or result-NNN identifier from this owner turn. Supply exactly
    one proposed read-only observation. If it is similar to prior research, state the distinct
    new hypothesis it tests. The agent validates and stores the request only in live turn memory.
    """
    raise RuntimeError("The operator intercepts this transient control tool before execution.")


def canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    observation_id: str
    result_id: str
    tool: str
    arguments: dict[str, Any]
    ok: bool

    @property
    def references(self) -> tuple[str, str]:
        return self.observation_id, self.result_id


@dataclass(frozen=True, slots=True)
class ResearchCheckpoint:
    verified_fact_pointers: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    inspected_observation_ids: tuple[str, ...]
    unresolved_fact: str
    why_current_evidence_is_insufficient: str
    proposed_tool: str
    proposed_arguments: dict[str, Any]
    result_that_would_change_the_conclusion: str
    new_hypothesis: str
    redundant_with: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    accepted: bool
    message: str
    redundant_with: tuple[str, ...] = ()


class TransientResearchNotebook:
    """Turn-local navigation state whose references resolve to live raw tool messages.

    It intentionally stores no tool-result text. Raw results in the conversation remain the
    only evidence and the only input from which a final answer may establish a fact.
    """

    def __init__(self, *, start_at: int = 1) -> None:
        self._start_at = max(1, int(start_at))
        self._observations: list[ObservationRecord] = []
        self._latest: ResearchCheckpoint | None = None
        self._pending: ResearchCheckpoint | None = None

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def has_pending_checkpoint(self) -> bool:
        return self._pending is not None

    def add_observation(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
    ) -> ObservationRecord:
        number = self._start_at + len(self._observations)
        record = ObservationRecord(
            observation_id=f"obs-{number:03d}",
            result_id=f"result-{number:03d}",
            tool=str(tool),
            arguments=dict(arguments),
            ok=bool(ok),
        )
        self._observations.append(record)
        return record

    @staticmethod
    def render_raw_result(record: ObservationRecord, result: Any) -> str:
        return (
            f"[Observation ID: {record.observation_id}]\n"
            f"[Tool result ID: {record.result_id}]\n"
            f"[Tool: {record.tool}]\n"
            "[This raw tool result is evidence; transient notebook text is not.]\n\n"
            f"{result}"
        )

    @staticmethod
    def render_cache_hit(record: ObservationRecord) -> str:
        return (
            "Identical read-only observation already acquired earlier in this turn. "
            f"Reuse the raw result at {record.observation_id}/{record.result_id}; "
            "this duplicate call produced no new evidence."
        )

    def _known_references(self) -> set[str]:
        return {
            reference.lower()
            for observation in self._observations
            for reference in observation.references
        }

    def _validate_references(self, label: str, entries: list[str]) -> str | None:
        known = self._known_references()
        for entry in entries:
            references = {item.lower() for item in _REFERENCE.findall(str(entry))}
            if not references:
                return f"{label} entry lacks an obs-NNN or result-NNN pointer: {entry!r}"
            unknown = sorted(references - known)
            if unknown:
                return f"{label} entry cites unknown current-turn references: {', '.join(unknown)}"
        return None

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        return set(_WORD.findall(str(value).lower().replace("_", " ")))

    @classmethod
    def _queries_similar(cls, left: Any, right: Any) -> bool:
        left_tokens = cls._tokens(left)
        right_tokens = cls._tokens(right)
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        return overlap >= 0.6 or left_tokens <= right_tokens or right_tokens <= left_tokens

    @staticmethod
    def _line_window(arguments: dict[str, Any]) -> tuple[int, int]:
        start = max(1, int(arguments.get("start_line", 1)))
        end = max(start, int(arguments.get("end_line", 200)))
        return start, end

    @classmethod
    def _semantically_similar(
        cls,
        tool: str,
        arguments: dict[str, Any],
        prior: ObservationRecord,
    ) -> bool:
        if tool != prior.tool:
            return False
        old = prior.arguments
        if tool in {"search_repository", "search_public_web"}:
            query_key = "query"
            return cls._queries_similar(arguments.get(query_key, ""), old.get(query_key, ""))
        if tool == "read_repository_file":
            if str(arguments.get("path", "")).lower() != str(old.get("path", "")).lower():
                return False
            new_start, new_end = cls._line_window(arguments)
            old_start, old_end = cls._line_window(old)
            return max(new_start, old_start) <= min(new_end, old_end)
        if tool == "list_repository_tree":
            if str(arguments.get("path", ".")).lower() != str(old.get("path", ".")).lower():
                return False
            return int(arguments.get("depth", 2)) <= int(old.get("depth", 2))
        return canonical_arguments(arguments) == canonical_arguments(old)

    def semantic_redundancies(self, tool: str, arguments: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            observation.observation_id
            for observation in self._observations
            if self._semantically_similar(str(tool), arguments, observation)
        )

    def submit(self, arguments: dict[str, Any]) -> CheckpointDecision:
        """Validate a model-authored notebook without interpreting it as factual evidence."""
        fields = {
            "verified_fact_pointers": list(arguments.get("verified_fact_pointers") or []),
            "unresolved_questions": list(arguments.get("unresolved_questions") or []),
            "inspected_observation_ids": list(arguments.get("inspected_observation_ids") or []),
        }
        for label in ("verified_fact_pointers", "unresolved_questions"):
            error = self._validate_references(label, fields[label])
            if error:
                return CheckpointDecision(False, error)

        known_observations = {item.observation_id for item in self._observations}
        inspected = tuple(str(item).lower() for item in fields["inspected_observation_ids"])
        if not inspected:
            return CheckpointDecision(False, "inspected_observation_ids must cite current-turn observations.")
        unknown_observations = sorted(set(inspected) - known_observations)
        if unknown_observations:
            return CheckpointDecision(
                False,
                "inspected_observation_ids contains unknown IDs: " + ", ".join(unknown_observations),
            )

        referenced_checkpoint_fields = {
            "unresolved_fact": str(arguments.get("unresolved_fact") or "").strip(),
            "why_current_evidence_is_insufficient": str(
                arguments.get("why_current_evidence_is_insufficient") or ""
            ).strip(),
        }
        for label, value in referenced_checkpoint_fields.items():
            if not value:
                return CheckpointDecision(False, f"{label} is required.")
            error = self._validate_references(label, [value])
            if error:
                return CheckpointDecision(False, error)

        proposed_tool = str(arguments.get("proposed_tool") or "").strip()
        proposed_json = str(arguments.get("proposed_arguments_json") or "").strip()
        conclusion_change = str(
            arguments.get("result_that_would_change_the_conclusion") or ""
        ).strip()
        if not proposed_tool or not proposed_json or not conclusion_change:
            return CheckpointDecision(
                False,
                "proposed_tool, proposed_arguments_json, and result_that_would_change_the_conclusion are required.",
            )
        try:
            proposed_arguments = json.loads(proposed_json)
        except (json.JSONDecodeError, TypeError) as exc:
            return CheckpointDecision(False, f"proposed_arguments_json is invalid JSON: {exc}")
        if not isinstance(proposed_arguments, dict):
            return CheckpointDecision(False, "proposed_arguments_json must encode one JSON object.")

        redundant_with = self.semantic_redundancies(proposed_tool, proposed_arguments)
        new_hypothesis = str(arguments.get("new_hypothesis") or "").strip()
        if redundant_with:
            error = self._validate_references("new_hypothesis", [new_hypothesis]) if new_hypothesis else "missing"
            if error:
                return CheckpointDecision(
                    False,
                    "The proposed observation is semantically similar to "
                    + ", ".join(redundant_with)
                    + ". State the distinct new hypothesis it tests and cite the prior observation ID; "
                    "a justified follow-up will be allowed.",
                    redundant_with,
                )

        checkpoint = ResearchCheckpoint(
            verified_fact_pointers=tuple(str(item) for item in fields["verified_fact_pointers"]),
            unresolved_questions=tuple(str(item) for item in fields["unresolved_questions"]),
            inspected_observation_ids=inspected,
            unresolved_fact=referenced_checkpoint_fields["unresolved_fact"],
            why_current_evidence_is_insufficient=referenced_checkpoint_fields[
                "why_current_evidence_is_insufficient"
            ],
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            result_that_would_change_the_conclusion=conclusion_change,
            new_hypothesis=new_hypothesis,
            redundant_with=redundant_with,
        )
        self._latest = checkpoint
        self._pending = checkpoint
        return CheckpointDecision(
            True,
            self.render(),
            redundant_with,
        )

    def authorizes(self, tool: str, arguments: dict[str, Any]) -> bool:
        pending = self._pending
        return bool(
            pending is not None
            and pending.proposed_tool == str(tool)
            and canonical_arguments(pending.proposed_arguments) == canonical_arguments(arguments)
        )

    def consume(self, tool: str, arguments: dict[str, Any]) -> bool:
        if not self.authorizes(tool, arguments):
            return False
        self._pending = None
        return True

    def render(self) -> str:
        checkpoint = self._latest
        if checkpoint is None:
            return "TRANSIENT RESEARCH NOTEBOOK: empty"

        def section(title: str, entries: tuple[str, ...]) -> list[str]:
            return [title, *(f"- {item}" for item in entries)] if entries else [title, "- none"]

        rows = [
            "TRANSIENT RESEARCH NOTEBOOK — NAVIGATION/PLANNING ONLY; NOT EVIDENCE",
            "Resolve every pointer against the full raw tool result still present in this conversation. ",
            "If notebook text conflicts with a raw result, the raw result controls.",
            "",
        ]
        rows.extend(section("Verified fact pointers", checkpoint.verified_fact_pointers))
        rows.append("")
        rows.extend(section("Unresolved questions", checkpoint.unresolved_questions))
        rows.extend(
            [
                "",
                "Inspected observations",
                *(f"- {item}" for item in checkpoint.inspected_observation_ids),
                "",
                "Next highest-value observation / information-gain checkpoint",
                f"- unresolved fact: {checkpoint.unresolved_fact}",
                f"- why raw evidence is insufficient: {checkpoint.why_current_evidence_is_insufficient}",
                f"- proposed observation: {checkpoint.proposed_tool} "
                f"{canonical_arguments(checkpoint.proposed_arguments)}",
                f"- result that would change the conclusion: "
                f"{checkpoint.result_that_would_change_the_conclusion}",
                f"- distinct hypothesis: {checkpoint.new_hypothesis or 'not semantically redundant'}",
            ]
        )
        return "\n".join(rows)

    def clear(self) -> None:
        self._observations.clear()
        self._latest = None
        self._pending = None
