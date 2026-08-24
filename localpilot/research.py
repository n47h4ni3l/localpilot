from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any


RESEARCH_NOTEBOOK_TOOL = "update_research_notebook"
MAX_EVIDENCE_REFS = 8
MAX_CHECKPOINT_TEXT = 320
MAX_TOOL_NAME = 80
MAX_ARGUMENT_DEPTH = 4
MAX_ARGUMENT_CONTAINER_ITEMS = 16
MAX_ARGUMENT_NODES = 64
MAX_ARGUMENT_STRING = 512
MAX_ARGUMENT_SERIALIZED_BYTES = 2048
MAX_CHECKPOINT_SERIALIZED_BYTES = 4096

_REFERENCE = re.compile(r"^(?:obs|result)-\d{3,}$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_CHECKPOINT_FIELDS = {
    "evidence_refs",
    "unresolved_fact",
    "proposed_tool",
    "proposed_arguments",
    "result_that_would_change_the_conclusion",
    "new_hypothesis",
}

_RESEARCH_NOTEBOOK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RESEARCH_NOTEBOOK_TOOL,
        "description": (
            "Authorize exactly one highest-value read-only observation after the research soft budget. "
            "This compact, transient planning delta is not evidence and is removed before final synthesis. "
            "Do not resend histories, factual summaries, raw results, or JSON encoded as a string."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "evidence_refs",
                "unresolved_fact",
                "proposed_tool",
                "proposed_arguments",
                "result_that_would_change_the_conclusion",
            ],
            "properties": {
                "evidence_refs": {
                    "type": "array",
                    "description": "Bare current-turn obs-NNN or result-NNN IDs relevant to this decision.",
                    "minItems": 1,
                    "maxItems": MAX_EVIDENCE_REFS,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": r"^(?:obs|result)-\d{3,}$",
                        "maxLength": 32,
                    },
                },
                "unresolved_fact": {
                    "type": "string",
                    "description": "The single material uncertainty blocking an answer.",
                    "minLength": 1,
                    "maxLength": MAX_CHECKPOINT_TEXT,
                },
                "proposed_tool": {
                    "type": "string",
                    "description": "One read-only tool for the next observation.",
                    "minLength": 1,
                    "maxLength": MAX_TOOL_NAME,
                },
                "proposed_arguments": {
                    "type": "object",
                    "description": "The proposed tool arguments as an object, never JSON text.",
                    "maxProperties": MAX_ARGUMENT_CONTAINER_ITEMS,
                },
                "result_that_would_change_the_conclusion": {
                    "type": "string",
                    "description": "The concrete result that would materially change the current decision.",
                    "minLength": 1,
                    "maxLength": MAX_CHECKPOINT_TEXT,
                },
                "new_hypothesis": {
                    "type": "string",
                    "description": "For redundant research only: the distinct hypothesis tested.",
                    "maxLength": MAX_CHECKPOINT_TEXT,
                },
            },
        },
    },
}


def research_notebook_tool_schema() -> dict[str, Any]:
    """Return the bounded manual schema used instead of Ollama's lossy callable conversion."""
    return copy.deepcopy(_RESEARCH_NOTEBOOK_TOOL_SCHEMA)


def update_research_notebook(
    evidence_refs: list[str],
    unresolved_fact: str,
    proposed_tool: str,
    proposed_arguments: dict[str, Any],
    result_that_would_change_the_conclusion: str,
    new_hypothesis: str = "",
) -> str:
    """Record one compact, transient next-observation decision.

    Args:
        evidence_refs: Bare current-turn obs-NNN or result-NNN IDs relevant to this decision.
        unresolved_fact: The single material uncertainty blocking an answer.
        proposed_tool: One read-only tool for the next observation.
        proposed_arguments: Tool arguments as an object, never JSON text.
        result_that_would_change_the_conclusion: Result that would materially change the decision.
        new_hypothesis: Distinct hypothesis when the observation resembles prior research.
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
    evidence_refs: tuple[str, ...]
    unresolved_fact: str
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
    """Turn-local navigation state that never stores tool-result text or factual summaries."""

    def __init__(self, *, start_at: int = 1, allowed_tools: set[str] | None = None) -> None:
        self._start_at = max(1, int(start_at))
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self._observations: list[ObservationRecord] = []
        self._latest: ResearchCheckpoint | None = None
        self._pending: ResearchCheckpoint | None = None

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def has_pending_checkpoint(self) -> bool:
        return self._pending is not None

    def add_observation(self, *, tool: str, arguments: dict[str, Any], ok: bool) -> ObservationRecord:
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
            "[This raw tool result is evidence; transient checkpoint text is not.]\n\n"
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

    @staticmethod
    def _reject(code: str, detail: str) -> CheckpointDecision:
        return CheckpointDecision(False, f"Checkpoint rejected ({code}): {detail}")

    @staticmethod
    def _bounded_text(arguments: dict[str, Any], field: str, *, required: bool) -> tuple[str, str | None]:
        value = arguments.get(field, "")
        if not isinstance(value, str):
            return "", f"{field} must be a string."
        value = value.strip()
        if required and not value:
            return "", f"{field} is required."
        if len(value) > MAX_CHECKPOINT_TEXT:
            return "", f"{field} exceeds {MAX_CHECKPOINT_TEXT} characters."
        return value, None

    @staticmethod
    def _validate_argument_value(value: Any, *, depth: int = 0) -> tuple[int, str | None]:
        if depth > MAX_ARGUMENT_DEPTH:
            return 0, f"proposed_arguments exceeds depth {MAX_ARGUMENT_DEPTH}."
        if value is None or isinstance(value, (bool, int)):
            return 1, None
        if isinstance(value, float):
            return (1, None) if math.isfinite(value) else (0, "proposed_arguments contains a non-finite number.")
        if isinstance(value, str):
            if len(value) > MAX_ARGUMENT_STRING:
                return 0, f"proposed_arguments contains a string over {MAX_ARGUMENT_STRING} characters."
            return 1, None
        if isinstance(value, list):
            if len(value) > MAX_ARGUMENT_CONTAINER_ITEMS:
                return 0, f"proposed_arguments list exceeds {MAX_ARGUMENT_CONTAINER_ITEMS} items."
            nodes = 1
            for item in value:
                child_nodes, error = TransientResearchNotebook._validate_argument_value(item, depth=depth + 1)
                if error:
                    return 0, error
                nodes += child_nodes
                if nodes > MAX_ARGUMENT_NODES:
                    return 0, f"proposed_arguments exceeds {MAX_ARGUMENT_NODES} values."
            return nodes, None
        if isinstance(value, dict):
            if len(value) > MAX_ARGUMENT_CONTAINER_ITEMS:
                return 0, f"proposed_arguments object exceeds {MAX_ARGUMENT_CONTAINER_ITEMS} fields."
            nodes = 1
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > MAX_TOOL_NAME:
                    return 0, "proposed_arguments keys must be short non-empty strings."
                child_nodes, error = TransientResearchNotebook._validate_argument_value(item, depth=depth + 1)
                if error:
                    return 0, error
                nodes += child_nodes
                if nodes > MAX_ARGUMENT_NODES:
                    return 0, f"proposed_arguments exceeds {MAX_ARGUMENT_NODES} values."
            return nodes, None
        return 0, "proposed_arguments contains a non-JSON value."

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
    def _semantically_similar(cls, tool: str, arguments: dict[str, Any], prior: ObservationRecord) -> bool:
        if tool != prior.tool:
            return False
        old = prior.arguments
        if tool in {"search_repository", "search_public_web"}:
            return cls._queries_similar(arguments.get("query", ""), old.get("query", ""))
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
        """Validate one model-authored planning delta without treating it as evidence."""
        if not isinstance(arguments, dict):
            return self._reject("shape", "arguments must be one object.")
        unexpected = sorted(set(arguments) - _CHECKPOINT_FIELDS)
        if unexpected:
            return self._reject("fields", "use only the six compact delta fields.")
        try:
            checkpoint_bytes = len(json.dumps(arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError):
            return self._reject("json", "arguments must contain only JSON values.")
        if checkpoint_bytes > MAX_CHECKPOINT_SERIALIZED_BYTES:
            return self._reject("size", f"payload exceeds {MAX_CHECKPOINT_SERIALIZED_BYTES} bytes.")

        raw_refs = arguments.get("evidence_refs")
        if not isinstance(raw_refs, list) or not raw_refs:
            return self._reject("refs", "evidence_refs must be a non-empty list of bare current-turn IDs.")
        if len(raw_refs) > MAX_EVIDENCE_REFS:
            return self._reject("refs", f"evidence_refs is limited to {MAX_EVIDENCE_REFS} IDs.")
        if any(not isinstance(item, str) or _REFERENCE.fullmatch(item) is None for item in raw_refs):
            return self._reject("refs", "only bare obs-NNN or result-NNN IDs are accepted.")
        evidence_refs = tuple(item.lower() for item in raw_refs)
        if len(set(evidence_refs)) != len(evidence_refs):
            return self._reject("refs", "evidence_refs must not repeat IDs.")
        unknown = sorted(set(evidence_refs) - self._known_references())
        if unknown:
            return self._reject("refs", "evidence_refs contains an unknown current-turn ID.")

        unresolved_fact, error = self._bounded_text(arguments, "unresolved_fact", required=True)
        if error:
            return self._reject("text", error)
        conclusion_change, error = self._bounded_text(arguments, "result_that_would_change_the_conclusion", required=True)
        if error:
            return self._reject("text", error)
        new_hypothesis, error = self._bounded_text(arguments, "new_hypothesis", required=False)
        if error:
            return self._reject("text", error)

        proposed_tool = arguments.get("proposed_tool")
        if not isinstance(proposed_tool, str) or not proposed_tool.strip():
            return self._reject("tool", "proposed_tool is required.")
        proposed_tool = proposed_tool.strip()
        if len(proposed_tool) > MAX_TOOL_NAME:
            return self._reject("tool", f"proposed_tool exceeds {MAX_TOOL_NAME} characters.")
        if self._allowed_tools is not None and proposed_tool not in self._allowed_tools:
            return self._reject("tool", "proposed_tool is not an allowed read-only observation tool.")

        proposed_arguments = arguments.get("proposed_arguments")
        if not isinstance(proposed_arguments, dict):
            return self._reject("arguments", "proposed_arguments must be an object, not JSON text.")
        _, error = self._validate_argument_value(proposed_arguments)
        if error:
            return self._reject("arguments", error)
        serialized_arguments = json.dumps(
            proposed_arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(serialized_arguments) > MAX_ARGUMENT_SERIALIZED_BYTES:
            return self._reject("arguments", f"proposed_arguments exceeds {MAX_ARGUMENT_SERIALIZED_BYTES} bytes.")

        redundant_with = self.semantic_redundancies(proposed_tool, proposed_arguments)
        if redundant_with:
            if not new_hypothesis:
                return CheckpointDecision(
                    False,
                    "Checkpoint rejected (redundancy): state one distinct new_hypothesis.",
                    redundant_with,
                )
            redundant_refs = {
                reference
                for observation in self._observations
                if observation.observation_id in redundant_with
                for reference in observation.references
            }
            if not (set(evidence_refs) & redundant_refs):
                return CheckpointDecision(
                    False,
                    "Checkpoint rejected (redundancy): evidence_refs must include the similar prior observation.",
                    redundant_with,
                )

        checkpoint = ResearchCheckpoint(
            evidence_refs=evidence_refs,
            unresolved_fact=unresolved_fact,
            proposed_tool=proposed_tool,
            proposed_arguments=dict(proposed_arguments),
            result_that_would_change_the_conclusion=conclusion_change,
            new_hypothesis=new_hypothesis,
            redundant_with=redundant_with,
        )
        self._latest = checkpoint
        self._pending = checkpoint
        return CheckpointDecision(True, self.render(), redundant_with)

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
            return "TRANSIENT RESEARCH CHECKPOINT: empty"
        rows = [
            "TRANSIENT RESEARCH CHECKPOINT — PLANNING ONLY; NOT EVIDENCE",
            "Accepted for exactly one next observation and removed before final synthesis.",
            f"evidence_refs: {', '.join(checkpoint.evidence_refs)}",
            f"unresolved_fact: {checkpoint.unresolved_fact}",
            f"proposed_observation: {checkpoint.proposed_tool} {canonical_arguments(checkpoint.proposed_arguments)}",
            "result_that_would_change_the_conclusion: "
            f"{checkpoint.result_that_would_change_the_conclusion}",
        ]
        if checkpoint.new_hypothesis:
            rows.append(f"new_hypothesis: {checkpoint.new_hypothesis}")
        return "\n".join(rows)

    def clear(self) -> None:
        self._observations.clear()
        self._latest = None
        self._pending = None
