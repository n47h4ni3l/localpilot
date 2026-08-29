from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from localpilot.audit import AuditLog
from localpilot.process import hidden_process_creation_flags


class RuntimeEvidence:
    """Bounded read-only evidence about this checkout and runtime lifecycle."""

    def __init__(
        self,
        project_root: str | Path,
        audit_path: str | Path,
        *,
        main_branch: str = "main",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.audit = AuditLog(audit_path)
        self.main_branch = str(main_branch).strip() or "main"

    def _git(self, *args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.project_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def repository(self) -> dict[str, Any]:
        if not (self.project_root / ".git").exists():
            return {
                "branch": None,
                "commit": None,
                "working_tree_clean": None,
                "upstream": None,
                "upstream_commit": None,
                "ahead_of_upstream": None,
                "behind_upstream": None,
                "matches_upstream": None,
                "main_branch": self.main_branch,
                "main_commit": None,
                "origin_main_commit": None,
                "head_matches_main": None,
                "main_matches_origin": None,
                "freshness_scope": "not a Git checkout",
            }
        branch = self._git("branch", "--show-current")
        commit = self._git("rev-parse", "HEAD")
        changes = self._git("status", "--short")
        upstream = self._git("rev-parse", "--abbrev-ref", "@{upstream}")
        upstream_commit = self._git("rev-parse", "@{upstream}") if upstream else None
        ahead = behind = None
        if upstream:
            counts = self._git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
            if counts:
                parts = counts.split()
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    behind, ahead = (int(parts[0]), int(parts[1]))
        main_commit = self._git("rev-parse", self.main_branch)
        origin_main = self._git("rev-parse", f"origin/{self.main_branch}")
        return {
            "branch": branch,
            "commit": commit,
            "working_tree_clean": changes == "" if changes is not None else None,
            "upstream": upstream,
            "upstream_commit": upstream_commit,
            "ahead_of_upstream": ahead,
            "behind_upstream": behind,
            "matches_upstream": commit == upstream_commit if commit and upstream_commit else None,
            "main_branch": self.main_branch,
            "main_commit": main_commit,
            "origin_main_commit": origin_main,
            "head_matches_main": commit == main_commit if commit and main_commit else None,
            "main_matches_origin": main_commit == origin_main if main_commit and origin_main else None,
            "freshness_scope": "local refs only; fetch GitHub before claiming remote freshness",
        }

    def lifecycle(self, *, limit: int = 8) -> list[dict[str, Any]]:
        fields = (
            "timestamp",
            "transition",
            "old_pid",
            "new_pid",
            "process_started_at",
            "reason",
            "return_code",
            "signal",
            "request_id",
            "session_id",
            "message_id",
            "affected_requests",
            "source",
        )
        return [
            {key: row.get(key) for key in fields if key in row}
            for row in self.audit.recent("runtime_lifecycle", limit=limit)
        ]

    def snapshot(self, *, limit: int = 8) -> dict[str, Any]:
        lifecycle = self.lifecycle(limit=limit)
        current_process: dict[str, Any] | None = None
        for row in lifecycle:
            transition = row.get("transition")
            if transition in {"started", "ready"} and row.get("new_pid") is not None:
                current_process = {
                    "pid": row.get("new_pid"),
                    "started_at": row.get("process_started_at"),
                    "source": row.get("source"),
                    "reason": row.get("reason"),
                }
                break
            if transition in {"exited", "stopped"}:
                break
        return {
            "current_process": current_process,
            "recent_lifecycle": lifecycle,
            "repository": self.repository(),
        }
