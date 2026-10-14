from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from localpilot.desktop_state import DesktopUIState
from localpilot.process import hidden_process_creation_flags

_SAFE_REMOTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    enabled: bool
    current_version: str
    last_checked: str | None
    update_available: bool | None
    target_version: str | None
    check_error: str | None
    target_sha: str | None = None
    dismissed_target_sha: str | None = None
    apply_requested_sha: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "enabled": self.enabled,
            "currentVersion": self.current_version,
            "lastChecked": self.last_checked,
            "updateAvailable": self.update_available,
            "targetVersion": self.target_version,
            "checkError": self.check_error,
            "targetSha": self.target_sha,
            "noticeAvailable": (
                not self.enabled and self.update_available is True and not self.check_error
                and self.target_sha is not None and self.target_sha != self.dismissed_target_sha
            ),
            "applyRequested": bool(self.target_sha and self.target_sha == self.apply_requested_sha),
        }


def _run(root: Path, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=timeout,
        creationflags=hidden_process_creation_flags(),
    )


def _short_sha(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text[:7] if text else None


def _current_version(root: Path) -> str:
    result = _run(root, ["git", "rev-parse", "--short=7", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def current_update_status(root: str | Path, state: DesktopUIState) -> UpdateStatus:
    root = Path(root).resolve()
    values = state.read()
    return UpdateStatus(
        enabled=bool(values.get("automatic_updates")),
        current_version=_current_version(root),
        last_checked=values.get("update_last_checked") if isinstance(values.get("update_last_checked"), str) else None,
        update_available=(
            bool(values.get("update_available"))
            if isinstance(values.get("update_available"), bool)
            else None
        ),
        target_version=(
            str(values.get("update_target_version"))
            if isinstance(values.get("update_target_version"), str)
            else None
        ),
        check_error=(
            str(values.get("update_check_error"))
            if isinstance(values.get("update_check_error"), str)
            else None
        ),
        target_sha=values.get("update_target_sha"),
        dismissed_target_sha=values.get("update_dismissed_target_sha"),
        apply_requested_sha=values.get("update_apply_requested_sha"),
    )


def set_automatic_updates(root: str | Path, state: DesktopUIState, enabled: bool) -> UpdateStatus:
    state.update(automatic_updates=bool(enabled))
    return current_update_status(root, state)


def request_manual_update(state: DesktopUIState, target_sha: str) -> dict[str, Any]:
    """Queue the owner's exact selection; the companion applies it after chat closes."""
    values = state.read()
    if (not re.fullmatch(r"[0-9a-f]{40}", str(target_sha))
            or target_sha != values.get("update_target_sha")
            or values.get("update_available") is not True or values.get("update_check_error")):
        return {"ok": False, "reason": "The update changed or is unavailable. Check again before choosing it."}
    state.update(update_apply_requested_sha=target_sha)
    return {"ok": True}


def dismiss_update_notice(state: DesktopUIState, target_sha: str) -> dict[str, Any]:
    if target_sha != state.read().get("update_target_sha"):
        return {"ok": False, "reason": "The available update changed."}
    state.update(update_dismissed_target_sha=target_sha)
    return {"ok": True}


def check_for_updates(
    root: str | Path,
    state: DesktopUIState,
    *,
    remote: str = "origin",
    main_branch: str = "main",
) -> UpdateStatus:
    """Refresh trusted-main metadata without changing the checkout.

    The settings surface is deliberately read-only with respect to repository
    contents. Applying an update is owned by the external updater/restart
    handoff, not by the WebView process.
    """

    root = Path(root).resolve()
    checked_at = datetime.now(timezone.utc).isoformat()
    update_available: bool | None = None
    target_version: str | None = None
    target_sha: str | None = None
    error: str | None = None

    try:
        if not _SAFE_REMOTE.fullmatch(str(remote)):
            raise RuntimeError("Configured Git remote name is unsafe.")
        if not _SAFE_BRANCH.fullmatch(str(main_branch)):
            raise RuntimeError("Configured main branch name is unsafe.")

        top = _run(root, ["git", "rev-parse", "--show-toplevel"])
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            raise RuntimeError("Project root is not the Git checkout root.")

        branch = _run(root, ["git", "branch", "--show-current"])
        if branch.returncode != 0 or branch.stdout.strip() != main_branch:
            current = branch.stdout.strip() or "detached HEAD"
            raise RuntimeError(
                f"Update check requires trusted {main_branch}; current checkout is {current}."
            )

        status = _run(root, ["git", "status", "--porcelain", "--untracked-files=all"])
        if status.returncode != 0:
            raise RuntimeError(status.stderr.strip() or "Could not inspect the working tree.")
        if status.stdout.strip():
            raise RuntimeError("Main checkout has uncommitted work; automatic update check was deferred.")

        remote_ref = f"refs/remotes/{remote}/{main_branch}"
        fetched = _run(
            root,
            [
                "git",
                "fetch",
                "--no-tags",
                "--prune",
                remote,
                f"+refs/heads/{main_branch}:{remote_ref}",
            ],
            timeout=120,
        )
        if fetched.returncode != 0:
            raise RuntimeError(fetched.stderr.strip() or fetched.stdout.strip() or "Git fetch failed.")

        head = _run(root, ["git", "rev-parse", "--verify", "HEAD^{commit}"])
        target = _run(root, ["git", "rev-parse", "--verify", f"{remote_ref}^{{commit}}"])
        if head.returncode != 0 or target.returncode != 0:
            raise RuntimeError("Could not resolve local and remote main commits.")

        head_sha = head.stdout.strip()
        target_sha = target.stdout.strip()
        target_version = _short_sha(target_sha)
        if head_sha == target_sha:
            update_available = False
        else:
            ancestor = _run(root, ["git", "merge-base", "--is-ancestor", head_sha, target_sha])
            if ancestor.returncode != 0:
                raise RuntimeError("Local main is ahead of or diverged from the remote; update was refused.")
            update_available = True
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        error = str(exc)[:1000]

    state.update(
        update_last_checked=checked_at,
        update_available=update_available,
        update_target_version=target_version,
        update_target_sha=target_sha if error is None else None,
        update_check_error=error,
    )
    return current_update_status(root, state)
