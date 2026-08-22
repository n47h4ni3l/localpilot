from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


CORE_CAPABILITY_QUESTION = (
    "What is the highest-leverage change I could make to increase my own future capability?"
)


class EvolutionClass(str, Enum):
    REPAIR = "repair"
    EXTEND = "extend"
    IMPROVE_COGNITION = "improve_cognition"
    EXPLORE = "explore"

    @property
    def label(self) -> str:
        return {
            self.REPAIR: "Repair",
            self.EXTEND: "Extend",
            self.IMPROVE_COGNITION: "Improve Cognition",
            self.EXPLORE: "Explore",
        }[self]


_CLASS_ALIASES = {
    "repair": EvolutionClass.REPAIR,
    "extend": EvolutionClass.EXTEND,
    "improve cognition": EvolutionClass.IMPROVE_COGNITION,
    "improve_cognition": EvolutionClass.IMPROVE_COGNITION,
    "cognition": EvolutionClass.IMPROVE_COGNITION,
    "explore": EvolutionClass.EXPLORE,
}
_SENSITIVE = re.compile(
    r"(?i)(password|passwd|token|secret|api[_-]?key|credential|authorization|bearer)"
)
_TOKEN_SHAPES = re.compile(r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})")
_OPAQUE_VALUES = re.compile(
    r"\b(?=[A-Za-z0-9+/=_-]{32,}\b)(?=[A-Za-z0-9+/=_-]*[A-Za-z])"
    r"(?=[A-Za-z0-9+/=_-]*\d)[A-Za-z0-9+/=_-]+\b"
)


@dataclass(frozen=True, slots=True)
class CapabilityProposal:
    evolution_class: EvolutionClass
    title: str
    capability_target: str
    question: str
    observed_limitation: str
    evidence: tuple[str, ...]
    alternatives: tuple[str, ...]
    hypothesis: str
    metric: str
    baseline: str
    success_criterion: str
    measurement_method: str
    mission_alignment: str
    current_frontier: str
    why_high_leverage: str
    capability_unlocked: str
    next_frontier: str
    expected_complexity: str

    def task(self, task_id: str) -> dict[str, Any]:
        return {
            "id": task_id,
            "title": self.title,
            "status": "todo",
            "source": "capability_discovery",
            "evolution_class": self.evolution_class.value,
            "capability_target": self.capability_target,
            "mission_alignment": self.mission_alignment,
            "current_frontier": self.current_frontier,
            "why_high_leverage": self.why_high_leverage,
            "capability_unlocked": self.capability_unlocked,
            "next_frontier": self.next_frontier,
            "question": self.question,
            "observed_limitation": self.observed_limitation,
            "evidence": list(self.evidence),
            "alternatives": list(self.alternatives),
            "hypothesis": self.hypothesis,
            "expected_complexity": self.expected_complexity,
            "evaluation": {
                "metric": self.metric,
                "baseline": self.baseline,
                "success_criterion": self.success_criterion,
                "measurement_method": self.measurement_method,
            },
            "acceptance": experiment_acceptance(
                observed_limitation=self.observed_limitation,
                hypothesis=self.hypothesis,
                metric=self.metric,
                baseline=self.baseline,
                success_criterion=self.success_criterion,
                measurement_method=self.measurement_method,
            ),
        }


def _clean(value: Any, limit: int = 1000) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    if _SENSITIVE.search(text) or _TOKEN_SHAPES.search(text) or _OPAQUE_VALUES.search(text):
        return "<redacted>"
    return text[:limit]


def _strings(value: Any, *, limit: int = 12) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_clean(item) for item in value[:limit] if _clean(item))


def _class(value: Any) -> EvolutionClass | None:
    return _CLASS_ALIASES.get(_clean(value, 80).lower().replace("-", "_"))


def classify_evolution_task(task: dict[str, Any]) -> EvolutionClass:
    explicit = _class(task.get("evolution_class") or task.get("class"))
    if explicit is not None:
        return explicit

    text = " ".join(
        [
            _clean(task.get("title"), 1000),
            _clean(task.get("observed_limitation"), 1000),
            " ".join(_strings(task.get("acceptance"), limit=30)),
        ]
    ).lower()
    scores = {
        EvolutionClass.REPAIR: sum(
            word in text for word in ("bug", "failure", "fix", "repair", "reliab", "resource")
        ),
        EvolutionClass.EXTEND: sum(
            word in text for word in ("new tool", "add tool", "capability", "operator", "observer", "integration")
        ),
        EvolutionClass.IMPROVE_COGNITION: sum(
            word in text
            for word in (
                "planning",
                "memory",
                "retrieval",
                "evaluation",
                "model routing",
                "context",
                "learning",
                "self-review",
            )
        ),
        EvolutionClass.EXPLORE: sum(
            word in text
            for word in ("experiment", "hypothesis", "architecture", "adaptation", "self-teaching", "uncertain")
        ),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else EvolutionClass.REPAIR


def experiment_acceptance(
    *,
    observed_limitation: str,
    hypothesis: str,
    metric: str,
    baseline: str,
    success_criterion: str,
    measurement_method: str,
) -> list[str]:
    return [
        f"Address the observed limitation: {observed_limitation}",
        f"Test the falsifiable hypothesis: {hypothesis}",
        f"Measure {metric} against baseline {baseline}",
        f"Use {measurement_method} and require {success_criterion}",
        "Report before/after evidence when feasible; otherwise identify the CI evaluation still required",
        "Do not retain added complexity when the evaluation is regressed or unmeasured",
    ]


def normalize_evolution_task(task: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task)
    kind = classify_evolution_task(normalized)
    title = _clean(normalized.get("title") or normalized.get("id") or "Capability experiment")
    target = _clean(normalized.get("capability_target") or title)
    limitation = _clean(
        normalized.get("observed_limitation")
        or f"The acceptance criteria for {title} are not yet satisfied."
    )
    evaluation = normalized.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    acceptance = list(normalized.get("acceptance") or [])
    normalized.update(
        {
            "evolution_class": kind.value,
            "capability_target": target,
            "mission_alignment": _clean(
                normalized.get("mission_alignment")
                or f"Increase transferable future capability in {target} while preserving human control."
            ),
            "current_frontier": _clean(
                normalized.get("current_frontier") or limitation
            ),
            "why_high_leverage": _clean(
                normalized.get("why_high_leverage")
                or f"Improving {target} should increase capability across future tasks or improve learning."
            ),
            "capability_unlocked": _clean(
                normalized.get("capability_unlocked")
                or f"Stronger and more transferable {target}"
            ),
            "next_frontier": _clean(
                normalized.get("next_frontier")
                or f"Reassess the capability frontier after measuring {target}."
            ),
            "question": _clean(
                normalized.get("question")
                or f"How can LocalPilot improve {target} with the highest verified leverage?"
            ),
            "observed_limitation": limitation,
            "evidence": list(_strings(normalized.get("evidence"))),
            "alternatives": list(_strings(normalized.get("alternatives"))),
            "hypothesis": _clean(
                normalized.get("hypothesis")
                or f"Implementing {title} will satisfy its acceptance contract without weakening safety."
            ),
            "expected_complexity": _clean(
                normalized.get("expected_complexity") or "medium", 20
            ).lower(),
            "evaluation": {
                "metric": _clean(
                    evaluation.get("metric") or "acceptance criteria and regression checks passed"
                ),
                "baseline": _clean(
                    evaluation.get("baseline") or "one or more acceptance criteria are currently unmet"
                ),
                "success_criterion": _clean(
                    evaluation.get("success_criterion")
                    or "all acceptance criteria and GitHub CI checks pass"
                ),
                "measurement_method": _clean(
                    evaluation.get("measurement_method")
                    or "non-executing local static checks followed by GitHub CI"
                ),
            },
        }
    )
    if not acceptance:
        normalized["acceptance"] = experiment_acceptance(
            observed_limitation=limitation,
            hypothesis=normalized["hypothesis"],
            metric=normalized["evaluation"]["metric"],
            baseline=normalized["evaluation"]["baseline"],
            success_criterion=normalized["evaluation"]["success_criterion"],
            measurement_method=normalized["evaluation"]["measurement_method"],
        )
    return normalized


def _proposal(value: dict[str, Any]) -> CapabilityProposal:
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    kind = _class(value.get("evolution_class"))
    proposal = CapabilityProposal(
        evolution_class=kind or EvolutionClass.REPAIR,
        title=_clean(value.get("title")),
        capability_target=_clean(value.get("capability_target")),
        mission_alignment=_clean(value.get("mission_alignment")),
        current_frontier=_clean(value.get("current_frontier")),
        why_high_leverage=_clean(value.get("why_high_leverage")),
        capability_unlocked=_clean(value.get("capability_unlocked")),
        next_frontier=_clean(value.get("next_frontier")),
        question=_clean(value.get("question")),
        observed_limitation=_clean(value.get("observed_limitation")),
        evidence=_strings(value.get("evidence")),
        alternatives=_strings(value.get("alternatives")),
        hypothesis=_clean(value.get("hypothesis")),
        metric=_clean(evaluation.get("metric")),
        baseline=_clean(evaluation.get("baseline")),
        success_criterion=_clean(evaluation.get("success_criterion")),
        measurement_method=_clean(evaluation.get("measurement_method")),
        expected_complexity=_clean(value.get("expected_complexity") or "medium", 20).lower(),
    )
    missing = [
        name
        for name, item in (
            ("title", proposal.title),
            ("capability_target", proposal.capability_target),
            ("mission_alignment", proposal.mission_alignment),
            ("current_frontier", proposal.current_frontier),
            ("why_high_leverage", proposal.why_high_leverage),
            ("capability_unlocked", proposal.capability_unlocked),
            ("next_frontier", proposal.next_frontier),
            ("question", proposal.question),
            ("observed_limitation", proposal.observed_limitation),
            ("evidence", proposal.evidence),
            ("alternatives", proposal.alternatives),
            ("hypothesis", proposal.hypothesis),
            ("evaluation.metric", proposal.metric),
            ("evaluation.baseline", proposal.baseline),
            ("evaluation.success_criterion", proposal.success_criterion),
            ("evaluation.measurement_method", proposal.measurement_method),
        )
        if not item
    ]
    if kind is None:
        missing.append("evolution_class")
    if missing:
        raise ValueError(f"Capability proposal is unmeasured or incomplete: {', '.join(missing)}")
    if proposal.expected_complexity not in {"low", "medium", "high"}:
        raise ValueError("expected_complexity must be low, medium, or high")
    return proposal


def parse_capability_proposals(text: str) -> list[CapabilityProposal]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Capability discovery did not return a JSON object.")
    root = json.loads(candidate[start : end + 1])
    if not isinstance(root, dict):
        raise ValueError("Capability discovery root must be a JSON object.")
    values = root.get("proposals")
    if not isinstance(values, list) or not values:
        values = [root]
    proposals: list[CapabilityProposal] = []
    errors: list[str] = []
    for value in values[:6]:
        if not isinstance(value, dict):
            errors.append("proposal is not an object")
            continue
        try:
            proposals.append(_proposal(value))
        except ValueError as exc:
            errors.append(str(exc))
    if not proposals:
        raise ValueError("No measurable capability proposal was produced: " + "; ".join(errors))
    return proposals


def capability_proposal_score(proposal: CapabilityProposal) -> int:
    """Prefer evidence, alternatives and measurable leverage; penalize complexity."""
    if not all(
        (
            proposal.metric,
            proposal.baseline,
            proposal.success_criterion,
            proposal.measurement_method,
        )
    ):
        return -10_000
    score = min(len(proposal.evidence), 4) * 3 + min(len(proposal.alternatives), 3) * 2
    score += {
        EvolutionClass.REPAIR: 0,
        EvolutionClass.EXTEND: 3,
        EvolutionClass.IMPROVE_COGNITION: 4,
        EvolutionClass.EXPLORE: 2,
    }[proposal.evolution_class]
    score -= {"low": 0, "medium": 2, "high": 5}.get(proposal.expected_complexity, 8)

    leverage_text = " ".join(
        (
            proposal.mission_alignment,
            proposal.why_high_leverage,
            proposal.capability_unlocked,
            proposal.next_frontier,
        )
    ).lower()
    transfer_signals = (
        "general",
        "transfer",
        "across",
        "many future",
        "learn",
        "new capability",
        "multiple",
    )
    score += min(sum(signal in leverage_text for signal in transfer_signals), 5)
    return score


def select_capability_proposal(proposals: Iterable[CapabilityProposal]) -> CapabilityProposal:
    ranked = sorted(proposals, key=capability_proposal_score, reverse=True)
    if not ranked or capability_proposal_score(ranked[0]) < -100:
        raise ValueError("No measured capability proposal is eligible.")
    return ranked[0]


def capability_task_id(proposal: CapabilityProposal) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", proposal.capability_target.lower()).strip("-")[:28]
    digest = hashlib.sha256(
        json.dumps(
            {
                "class": proposal.evolution_class.value,
                "target": proposal.capability_target,
                "hypothesis": proposal.hypothesis,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"capability-{slug or 'growth'}-{digest}"


def evolution_status_fields(task: dict[str, Any]) -> dict[str, str]:
    normalized = normalize_evolution_task(task)
    evaluation = normalized["evaluation"]
    return {
        "evolution_class": EvolutionClass(normalized["evolution_class"]).label,
        "capability_target": normalized["capability_target"],
        "mission_alignment": normalized["mission_alignment"],
        "current_frontier": normalized["current_frontier"],
        "why_high_leverage": normalized["why_high_leverage"],
        "capability_unlocked": normalized["capability_unlocked"],
        "next_frontier": normalized["next_frontier"],
        "hypothesis": normalized["hypothesis"],
        "evaluation_plan": (
            f"{evaluation['metric']}; baseline: {evaluation['baseline']}; "
            f"success: {evaluation['success_criterion']}; method: {evaluation['measurement_method']}"
        ),
    }
