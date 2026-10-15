"""Durable, bounded autonomous continuation receipts for existing Evolution PRs.

A successful GitHub check is not proof an objective is finished. Track which
PR head has already been offered one incremental iteration, so a crash,
failed/no-op cycle, or unchanged CI run cannot trigger unlimited retries.
Owner-authenticated PR feedback remains the independent override path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MAX_AUTONOMOUS_STEPS_PER_PR = 6
MAX_TRACKED_BRANCHES = 64


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
            ):
                raise ValueError("Invalid autonomous PR iteration record.")
        return branches

    def can_start(self, branch: str, head: str) -> bool:
        if not branch or not head:
            return False
        current = self._read().get(branch)
        return current is None or (
            current["head"] != head
            and int(current["attempts"]) < MAX_AUTONOMOUS_STEPS_PER_PR
        )

    def claim(self, branch: str, head: str) -> bool:
        """Claim the current PR head *before* running inference.

        Claims never repeat the same HEAD, even if a run crashes or changes
        nothing. Each later verified commit permits one more bounded step.
        """
        records = self._read()
        if not self.can_start(branch, head):
            return False
        if branch not in records and len(records) >= MAX_TRACKED_BRANCHES:
            raise RuntimeError("Autonomous PR receipts full; owner review required.")
        prior = records.get(branch)
        records[branch] = {
            "head": head,
            "attempts": int(prior["attempts"]) + 1 if prior else 1,
        }
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
        return True
