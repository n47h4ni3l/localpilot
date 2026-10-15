"""Durable, bounded autonomous continuation receipts for existing Evolution PRs.

A passing GitHub check is not evidence the objective is finished. Persist the
exact one-step acceptance brief during execution so recovery can reconstruct
the same task fingerprint. A PR HEAD is attempted at most once; a crash, an
empty diff, or an unchanged successful CI run cannot trigger endless retries.
Owner-authenticated PR revisions remain an independent override path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MAX_AUTONOMOUS_STEPS_PER_PR = 6
MAX_TRACKED_BRANCHES = 64
MAX_STEP_CONTEXT_CHARS = 7000


class AutonomousPRLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Invalid autonomous PR iteration ledger.")
        branches = value.get("branches")
        if not isinstance(branches, dict) or len(branches) > MAX_TRACKED_BRANCHES:
            raise ValueError("Invalid autonomous PR iteration entries.")
        for branch, record in branches.items():
            if (
                not isinstance(branch, str)
                or not isinstance(record, dict)
                or type(record.get("attempts")) is not int
                or not 0 <= record["attempts"] <= MAX_AUTONOMOUS_STEPS_PER_PR
                or not isinstance(record.get("head"), str)
                or record.get("state") not in {"active", "completed", "needs_work"}
                or (record.get("cycle_id") is not None and type(record["cycle_id"]) is not int)
                or not isinstance(record.get("task_id"), str)
                or not isinstance(record.get("acceptance"), list)
                or any(not isinstance(item, str) for item in record["acceptance"])
                or len(json.dumps(record["acceptance"], ensure_ascii=False)) > MAX_STEP_CONTEXT_CHARS
            ):
                raise ValueError("Invalid autonomous PR iteration record.")
        return branches

    def active(self, branch: str) -> dict[str, object] | None:
        row = self._read().get(branch)
        return dict(row) if row and row["state"] == "active" else None

    def active_for_task(self, task_id: str) -> dict[str, object] | None:
        for branch, row in self._read().items():
            if row["state"] == "active" and row["task_id"] == task_id:
                return {"branch": branch, **row}
        return None

    def can_start(self, branch: str, head: str) -> bool:
        if not branch or not head:
            return False
        row = self._read().get(branch)
        return row is None or (
            row["state"] != "active"
            and row["head"] != head
            and int(row["attempts"]) < MAX_AUTONOMOUS_STEPS_PER_PR
        )

    def claim(
        self, branch: str, head: str, *,
        cycle_id: int | None = None,
        task_id: str = "",
        acceptance: list[str] | None = None,
    ) -> bool:
        """Claim this exact PR HEAD before inference and persist its step brief."""
        records = self._read()
        if not self.can_start(branch, head):
            return False
        if branch not in records and len(records) >= MAX_TRACKED_BRANCHES:
            raise RuntimeError("Autonomous PR receipts full; owner review required.")
        acceptance = acceptance or []
        if (
            (cycle_id is not None and (type(cycle_id) is not int or cycle_id <= 0 or not task_id))
            or any(not isinstance(item, str) for item in acceptance)
            or len(json.dumps(acceptance, ensure_ascii=False)) > MAX_STEP_CONTEXT_CHARS
        ):
            raise ValueError("Invalid autonomous PR step identity or acceptance brief.")
        prior = records.get(branch)
        records[branch] = {
            "head": head,
            "attempts": int(prior["attempts"]) + 1 if prior else 1,
            "state": "active" if cycle_id is not None else "completed",
            "cycle_id": cycle_id,
            "task_id": task_id,
            "acceptance": acceptance,
        }
        self._save(records)
        return True

    def finish(self, branch: str, cycle_id: int, status: str) -> None:
        if status not in {"completed", "needs_work"}:
            raise ValueError("Unknown autonomous PR iteration outcome.")
        records = self._read()
        row = records.get(branch)
        if row is None or row["cycle_id"] != cycle_id:
            raise ValueError("Autonomous PR cycle identity mismatch.")
        if row["state"] == status:
            return
        if row["state"] != "active":
            raise ValueError("Autonomous PR receipt already finalized.")
        row["state"] = status
        self._save(records)

    def _save(self, records: dict[str, dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.path.with_name(self.path.name + ".tmp")
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump({"version": 1, "branches": records}, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(target, self.path)
        finally:
            if target.exists():
                target.unlink()
