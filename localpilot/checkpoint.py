from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_VERSION = 2
_SENSITIVE = re.compile(
    r"(?i)(password|passwd|token|secret|api[_-]?key|credential|authorization|bearer)"
)
_TOKEN_SHAPES = re.compile(r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})")
_OPAQUE_VALUES = re.compile(
    r"\b(?=[A-Za-z0-9+/=_-]{32,}\b)(?=[A-Za-z0-9+/=_-]*[A-Za-z])"
    r"(?=[A-Za-z0-9+/=_-]*\d)[A-Za-z0-9+/=_-]+\b"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    if _SENSITIVE.search(text) or _TOKEN_SHAPES.search(text) or _OPAQUE_VALUES.search(text):
        return "<redacted>"
    return text[:limit]


def _safe_items(values: Iterable[Any], *, count: int = 50, limit: int = 1000) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        safe = _safe_text(value, limit)
        if safe and safe not in items:
            items.append(safe)
        if len(items) >= count:
            break
    return tuple(items)


def task_fingerprint(task: dict[str, Any]) -> str:
    """Fingerprint the reviewable task contract without storing source content."""
    contract = {
        "id": str(task.get("id") or ""),
        "title": str(task.get("title") or ""),
        "acceptance": list(task.get("acceptance") or []),
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvolutionCheckpoint:
    """Compact engineering handoff; deliberately excludes messages and file content."""

    version: int
    updated_at: str
    cycle_id: int
    task_id: str
    branch: str
    workspace: str
    objective: str
    acceptance_criteria: tuple[str, ...]
    task_fingerprint: str
    evolution_class: str
    capability_target: str
    hypothesis: str
    evaluation_plan: str
    milestone: str
    files_inspected: tuple[str, ...]
    files_changed: tuple[str, ...]
    research_findings: tuple[str, ...]
    decisions: tuple[str, ...]
    git_head: str
    git_state_digest: str
    diff_status: str
    static_check_status: str
    static_check_failures: tuple[str, ...]
    test_status: str
    test_failures: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    next_action: str
    reusable_lessons: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        cycle_id: int,
        task: dict[str, Any],
        branch: str,
        workspace: str | Path,
        milestone: str,
        files_inspected: Iterable[str] = (),
        files_changed: Iterable[str] = (),
        research_findings: Iterable[str] = (),
        decisions: Iterable[str] = (),
        git_head: str = "",
        git_state_digest: str = "",
        diff_status: str = "not inspected",
        static_check_status: str = "not run",
        static_check_failures: Iterable[str] = (),
        test_status: str = "not run locally; GitHub CI required",
        test_failures: Iterable[str] = (),
        unresolved_questions: Iterable[str] = (),
        next_action: str = "Validate the candidate state and continue the current milestone.",
        reusable_lessons: Iterable[str] = (),
    ) -> "EvolutionCheckpoint":
        return cls(
            version=CHECKPOINT_VERSION,
            updated_at=_now(),
            cycle_id=int(cycle_id),
            task_id=_safe_text(task.get("id"), 200),
            branch=_safe_text(branch, 300),
            workspace=str(Path(workspace).resolve()),
            objective=_safe_text(task.get("title"), 1000),
            acceptance_criteria=_safe_items(task.get("acceptance") or (), count=30, limit=1000),
            task_fingerprint=task_fingerprint(task),
            evolution_class=_safe_text(task.get("evolution_class") or "repair", 80),
            capability_target=_safe_text(task.get("capability_target") or task.get("title"), 1000),
            hypothesis=_safe_text(task.get("hypothesis"), 2000),
            evaluation_plan=_safe_text(
                json.dumps(task.get("evaluation") or {}, ensure_ascii=False, sort_keys=True),
                3000,
            ),
            milestone=_safe_text(milestone, 100),
            files_inspected=_safe_items(files_inspected, count=100, limit=500),
            files_changed=_safe_items(files_changed, count=100, limit=500),
            research_findings=_safe_items(research_findings, count=20, limit=2000),
            decisions=_safe_items(decisions, count=20, limit=1000),
            git_head=str(git_head or "")[:100],
            git_state_digest=str(git_state_digest or "")[:100],
            diff_status=_safe_text(diff_status, 2000),
            static_check_status=_safe_text(static_check_status, 100),
            static_check_failures=_safe_items(static_check_failures, count=30, limit=1000),
            test_status=_safe_text(test_status, 500),
            test_failures=_safe_items(test_failures, count=20, limit=1000),
            unresolved_questions=_safe_items(unresolved_questions, count=20, limit=1000),
            next_action=_safe_text(next_action, 1000),
            reusable_lessons=_safe_items(reusable_lessons, count=20, limit=1000),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvolutionCheckpoint":
        if value.get("version") == 1:
            value = dict(value)
            value.update(
                {
                    "version": CHECKPOINT_VERSION,
                    "evolution_class": "repair",
                    "capability_target": str(value.get("objective") or ""),
                    "hypothesis": "",
                    "evaluation_plan": "{}",
                }
            )
        required = {field.name for field in fields(cls)}
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise ValueError(f"Checkpoint schema mismatch; missing={missing}, extra={extra}")
        if value.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {value.get('version')!r}")
        if isinstance(value.get("cycle_id"), bool) or not isinstance(value.get("cycle_id"), int):
            raise ValueError("Checkpoint cycle_id must be an integer.")
        string_fields = required - {
            "version",
            "cycle_id",
            "acceptance_criteria",
            "files_inspected",
            "files_changed",
            "research_findings",
            "decisions",
            "static_check_failures",
            "test_failures",
            "unresolved_questions",
            "reusable_lessons",
        }
        for name in string_fields:
            if not isinstance(value.get(name), str):
                raise ValueError(f"Checkpoint field {name!r} must be a string.")
        try:
            datetime.fromisoformat(value["updated_at"])
        except ValueError as exc:
            raise ValueError("Checkpoint updated_at must be an ISO timestamp.") from exc
        for name in (
            "acceptance_criteria",
            "files_inspected",
            "files_changed",
            "research_findings",
            "decisions",
            "static_check_failures",
            "test_failures",
            "unresolved_questions",
            "reusable_lessons",
        ):
            item = value.get(name)
            if not isinstance(item, list) or not all(isinstance(part, str) for part in item):
                raise ValueError(f"Checkpoint field {name!r} must be a string list.")
            value[name] = tuple(item)
        return cls(**value)


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, checkpoint: EvolutionCheckpoint) -> None:
        payload = json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2) + "\n"
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> EvolutionCheckpoint | None:
        if not self.path.exists():
            return None
        if self.path.stat().st_size > 128_000:
            raise ValueError("Checkpoint exceeds the 128 KB durability limit.")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Checkpoint root must be a JSON object.")
        return EvolutionCheckpoint.from_dict(value)

    def clear(self) -> bool:
        if not self.path.exists():
            return False
        self.path.unlink()
        return True
