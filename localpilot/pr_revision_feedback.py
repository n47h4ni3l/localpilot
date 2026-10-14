"""Owner-authenticated PR review commands and a private, bounded receipt ledger.

The marker is a routing command, not an authentication secret. GitHub's verified
comment author and the LocalPilot-managed PR branch are independent gates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER = "NESTRA-REVISE:"
MAX_FEEDBACK_CHARS = 4000
MAX_RECEIPTS = 100


@dataclass(frozen=True, slots=True)
class OwnerRevision:
    comment_id: int
    pr_number: int
    branch: str
    feedback: str


def parse_owner_revision(
    comment: dict[str, Any], *, owner_login: str, pr_number: int, branch: str,
) -> OwnerRevision | None:
    """Only a real repository-owner issue comment beginning with the marker counts."""
    if not isinstance(comment, dict) or not owner_login:
        return None
    user = comment.get("user")
    if not isinstance(user, dict) or str(user.get("login") or "").casefold() != owner_login.casefold():
        return None
    identifier = comment.get("id")
    if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier <= 0:
        return None
    body = comment.get("body")
    if not isinstance(body, str) or not body.startswith(MARKER):
        return None
    feedback = body[len(MARKER):].strip()
    if not feedback or len(feedback) > MAX_FEEDBACK_CHARS:
        return None
    return OwnerRevision(identifier, pr_number, branch, feedback)


class OwnerRevisionLedger:
    """Durable command receipts; a single Evolution lease serializes modifications."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("receipts"), list):
            raise ValueError("Invalid owner PR revision ledger; refusing revision commands.")
        records = value["receipts"]
        if len(records) > MAX_RECEIPTS or any(
            not isinstance(item, dict)
            or type(item.get("comment_id")) is not int
            or not isinstance(item.get("branch"), str)
            or item.get("state") not in {"active", "completed", "needs_work"}
            for item in records
        ):
            raise ValueError("Invalid owner PR revision receipts.")
        return records

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.path.with_name(self.path.name + ".tmp")
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump({"version": 1, "receipts": records[-MAX_RECEIPTS:]}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(target, self.path)
        finally:
            if target.exists():
                target.unlink()

    def seen(self, branch: str) -> set[int]:
        return {int(r["comment_id"]) for r in self._load() if r["branch"] == branch}

    def active(self, branch: str) -> dict[str, Any] | None:
        records = self._load()
        return next((dict(r) for r in reversed(records) if r["branch"] == branch and r["state"] == "active"), None)

    def active_for_task(self, task_id: str) -> dict[str, Any] | None:
        records = self._load()
        return next(
            (dict(r) for r in reversed(records)
             if r.get("task_id") == task_id and r["state"] == "active"),
            None,
        )

    def begin(self, request: OwnerRevision, cycle_id: int, task_id: str) -> bool:
        records = self._load()
        if any(r["comment_id"] == request.comment_id for r in records):
            return False
        if any(r["branch"] == request.branch and r["state"] == "active" for r in records):
            return False
        if len(records) >= MAX_RECEIPTS:
            completed = next(
                (index for index, row in enumerate(records) if row["state"] != "active"),
                None,
            )
            if completed is None:
                raise RuntimeError("PR revision receipts are full; refusing to discard active work.")
            records.pop(completed)
        records.append({
            "comment_id": request.comment_id, "pr_number": request.pr_number,
            "branch": request.branch, "cycle_id": cycle_id, "task_id": task_id,
            "feedback": request.feedback, "state": "active",
        })
        self._save(records)
        return True

    def finish(self, comment_id: int, state: str) -> None:
        if state not in {"completed", "needs_work"}:
            raise ValueError("Invalid PR revision outcome.")
        records = self._load()
        found = False
        for row in records:
            if row["comment_id"] == comment_id and row["state"] == "active":
                row["state"] = state
                found = True
        if not found:
            raise ValueError("Active PR revision receipt not found.")
        self._save(records)
