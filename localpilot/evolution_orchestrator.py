from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import psutil


_WORD = re.compile(r"[a-z0-9]+")
_LOW_SIGNAL_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    "add", "build", "create", "implement", "improve", "localpilot", "system",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_text(value: Any, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit]


def _terms(value: Any) -> frozenset[str]:
    return frozenset(
        word
        for word in _WORD.findall(_bounded_text(value).lower())
        if len(word) > 2 and word not in _LOW_SIGNAL_WORDS
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def opportunity_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Compare capability intent while ignoring generic implementation wording."""

    left_target = _terms(left.get("capability_target"))
    right_target = _terms(right.get("capability_target"))
    left_intent = _terms(
        " ".join(
            _bounded_text(left.get(key))
            for key in ("question", "observed_limitation", "hypothesis")
        )
    )
    right_intent = _terms(
        " ".join(
            _bounded_text(right.get(key))
            for key in ("question", "observed_limitation", "hypothesis")
        )
    )
    return round(0.45 * _jaccard(left_target, right_target) + 0.55 * _jaccard(left_intent, right_intent), 6)


class EvolutionBudgetExceeded(RuntimeError):
    pass


class EvolutionRunAlreadyActive(RuntimeError):
    """Raised when another live process already owns the evolution lease."""


class EvolutionRunLease:
    """Crash-safe inter-process lease for one mutable evolution invocation.

    The lease uses atomic create-once semantics rather than a best-effort state
    file. PID reuse is guarded by the process creation time. A stale/corrupt
    lease is removed only after its contents are rechecked, so two contenders
    cannot both decide the same live lease is stale and silently proceed.
    """

    def __init__(self, path: str | Path, invocation_id: str) -> None:
        self.path = Path(path).resolve()
        self.invocation_id = str(invocation_id)
        self.pid = os.getpid()
        try:
            self.process_create_time = float(psutil.Process(self.pid).create_time())
        except (psutil.Error, OSError):
            self.process_create_time = 0.0
        self.acquired = False

    @staticmethod
    def _decode(raw: str) -> dict[str, Any] | None:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _owner_live(owner: dict[str, Any] | None) -> bool:
        if not owner:
            return False
        try:
            pid = int(owner.get("pid"))
        except (TypeError, ValueError):
            return False
        if pid <= 0 or not psutil.pid_exists(pid):
            return False
        expected = owner.get("process_create_time")
        try:
            process = psutil.Process(pid)
            actual = float(process.create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except (psutil.AccessDenied, OSError):
            # If a process exists but its metadata cannot be inspected, fail
            # closed and treat the lease as live rather than risking overlap.
            return True
        try:
            expected_time = float(expected)
        except (TypeError, ValueError):
            return True
        return abs(actual - expected_time) < 1.0

    def _payload(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "invocation_id": self.invocation_id,
                "pid": self.pid,
                "process_create_time": self.process_create_time,
                "started_at": _utc_now(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    observed = self.path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                owner = self._decode(observed)
                if self._owner_live(owner):
                    owner_pid = owner.get("pid") if owner else "unknown"
                    owner_id = owner.get("invocation_id") if owner else "unknown"
                    raise EvolutionRunAlreadyActive(
                        "another evolution invocation is already active "
                        f"(pid={owner_pid}, invocation={owner_id})"
                    )
                try:
                    current = self.path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                if current != observed:
                    continue
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            else:
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(self._payload())
                        handle.flush()
                        os.fsync(handle.fileno())
                except Exception:
                    self.path.unlink(missing_ok=True)
                    raise
                self.acquired = True
                return
        raise EvolutionRunAlreadyActive(
            "could not acquire the evolution lease because its owner changed concurrently"
        )

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.acquired = False
            return
        owner = self._decode(raw)
        if owner and str(owner.get("invocation_id") or "") == self.invocation_id:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.acquired = False

    def __enter__(self) -> "EvolutionRunLease":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


@dataclass
class EvolutionRunBudget:
    invocation_id: str
    state_path: Path
    wall_clock_seconds: float
    max_tool_calls: int
    max_web_calls: int
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path).resolve()
        self.started_monotonic = self.monotonic()
        self.started_at = _utc_now()
        self.tool_calls = 0
        self.web_calls = 0
        self.last_stage = "starting"
        self.status = "running"
        self._lock = threading.Lock()
        self._persist()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.monotonic() - self.started_monotonic)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "invocation_id": self.invocation_id,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "status": self.status,
            "stage": self.last_stage,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "limits": {
                "wall_clock_seconds": self.wall_clock_seconds,
                "tool_calls": self.max_tool_calls,
                "web_calls": self.max_web_calls,
            },
            "usage": {
                "tool_calls": self.tool_calls,
                "web_calls": self.web_calls,
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._snapshot(), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def check(self, stage: str) -> None:
        with self._lock:
            self.last_stage = _bounded_text(stage, 120) or "unknown"
            if self.elapsed_seconds > self.wall_clock_seconds:
                self.status = "budget_exhausted"
                self._persist()
                raise EvolutionBudgetExceeded(
                    f"cycle wall-clock budget exhausted after {self.elapsed_seconds:.1f}s "
                    f"(limit {self.wall_clock_seconds:.1f}s)"
                )

    def consume_tool(self, name: str, stage: str) -> None:
        with self._lock:
            self.last_stage = _bounded_text(stage, 120) or "unknown"
            if self.elapsed_seconds > self.wall_clock_seconds:
                self.status = "budget_exhausted"
                self._persist()
                raise EvolutionBudgetExceeded(
                    f"cycle wall-clock budget exhausted after {self.elapsed_seconds:.1f}s "
                    f"(limit {self.wall_clock_seconds:.1f}s)"
                )
            is_web_call = name in {"search_public_web", "fetch_public_https"}
            if self.tool_calls >= self.max_tool_calls:
                self.status = "budget_exhausted"
                self._persist()
                raise EvolutionBudgetExceeded(
                    f"cycle tool-call budget exhausted at {self.tool_calls} calls "
                    f"(limit {self.max_tool_calls})"
                )
            if is_web_call and self.web_calls >= self.max_web_calls:
                self.status = "budget_exhausted"
                self._persist()
                raise EvolutionBudgetExceeded(
                    f"cycle web-call budget exhausted at {self.web_calls} calls "
                    f"(limit {self.max_web_calls})"
                )
            self.tool_calls += 1
            if is_web_call:
                self.web_calls += 1
            self._persist()

    def finish(self, status: str) -> dict[str, Any]:
        with self._lock:
            if self.status == "running":
                self.status = _bounded_text(status, 80) or "completed"
            self._persist()
            return self._snapshot()


class OpportunityLedger:
    """Small durable queue for measured capability proposals and their outcomes."""

    def __init__(
        self,
        path: str | Path,
        *,
        similarity_threshold: float = 0.82,
        max_entries: int = 48,
    ) -> None:
        self.path = Path(path).resolve()
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = max(1, int(max_entries))

    def _read(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return []
        entries = payload.get("opportunities") if isinstance(payload, dict) else None
        return [dict(item) for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []

    def _write(self, entries: Iterable[dict[str, Any]]) -> None:
        retained = list(entries)[-self.max_entries :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {"version": 1, "updated_at": _utc_now(), "opportunities": retained},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    @staticmethod
    def _task_view(task: dict[str, Any]) -> dict[str, Any]:
        evaluation = task.get("evaluation") if isinstance(task.get("evaluation"), dict) else {}
        return {
            "id": _bounded_text(task.get("id"), 200),
            "title": _bounded_text(task.get("title"), 1000),
            "status": "todo",
            "source": "capability_discovery",
            "evolution_class": _bounded_text(task.get("evolution_class"), 40),
            "capability_target": _bounded_text(task.get("capability_target"), 1000),
            "mission_alignment": _bounded_text(task.get("mission_alignment"), 2000),
            "current_frontier": _bounded_text(task.get("current_frontier"), 2000),
            "why_high_leverage": _bounded_text(task.get("why_high_leverage"), 2000),
            "capability_unlocked": _bounded_text(task.get("capability_unlocked"), 2000),
            "next_frontier": _bounded_text(task.get("next_frontier"), 2000),
            "question": _bounded_text(task.get("question"), 1000),
            "observed_limitation": _bounded_text(task.get("observed_limitation"), 2000),
            "evidence": [
                _bounded_text(item, 1000) for item in list(task.get("evidence") or [])[:12]
            ],
            "alternatives": [
                _bounded_text(item, 1000) for item in list(task.get("alternatives") or [])[:12]
            ],
            "hypothesis": _bounded_text(task.get("hypothesis"), 2000),
            "expected_complexity": _bounded_text(task.get("expected_complexity"), 20),
            "evaluation": {
                "metric": _bounded_text(evaluation.get("metric"), 1000),
                "baseline": _bounded_text(evaluation.get("baseline"), 1000),
                "success_criterion": _bounded_text(evaluation.get("success_criterion"), 1000),
                "measurement_method": _bounded_text(evaluation.get("measurement_method"), 1000),
            },
            "acceptance": [
                _bounded_text(item, 1000) for item in list(task.get("acceptance") or [])[:12]
            ],
        }

    def duplicate(self, task: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
        candidate = self._task_view(task)
        best: dict[str, Any] | None = None
        best_score = 0.0
        for entry in self._read():
            if entry.get("status") in {"abandoned", "superseded"}:
                continue
            existing = entry.get("task")
            if not isinstance(existing, dict):
                continue
            if candidate["id"] and candidate["id"] == existing.get("id"):
                return entry, 1.0
            score = opportunity_similarity(candidate, existing)
            if score > best_score:
                best, best_score = entry, score
        if best_score >= self.similarity_threshold:
            return best, best_score
        return None, best_score

    def enqueue(self, task: dict[str, Any], *, score: int) -> bool:
        duplicate, _similarity = self.duplicate(task)
        if duplicate is not None:
            return False
        entries = self._read()
        now = _utc_now()
        entries.append(
            {
                "task": self._task_view(task),
                "score": int(score),
                "status": "proposed",
                "created_at": now,
                "updated_at": now,
                "outcome": "",
            }
        )
        self._write(entries)
        return True

    def next_task(self) -> dict[str, Any] | None:
        entries = self._read()
        eligible = [entry for entry in entries if entry.get("status") == "proposed"]
        if not eligible:
            return None
        selected = max(eligible, key=lambda item: (int(item.get("score") or 0), item.get("created_at", "")))
        task = selected.get("task")
        return dict(task) if isinstance(task, dict) else None

    def update(self, task_id: str, status: str, outcome: str = "") -> None:
        entries = self._read()
        changed = False
        now = _utc_now()
        for entry in entries:
            task = entry.get("task")
            if isinstance(task, dict) and str(task.get("id")) == str(task_id):
                entry["status"] = _bounded_text(status, 80)
                entry["updated_at"] = now
                if outcome:
                    entry["outcome"] = _bounded_text(outcome, 2000)
                changed = True
        if changed:
            self._write(entries)

    def context(self, limit: int = 12) -> list[dict[str, Any]]:
        return [
            {
                "task_id": entry.get("task", {}).get("id"),
                "capability_target": entry.get("task", {}).get("capability_target"),
                "hypothesis": entry.get("task", {}).get("hypothesis"),
                "metric": entry.get("task", {}).get("evaluation", {}).get("metric"),
                "status": entry.get("status"),
                "outcome": entry.get("outcome"),
            }
            for entry in self._read()[-max(1, min(int(limit), 24)) :]
            if isinstance(entry.get("task"), dict)
        ]
