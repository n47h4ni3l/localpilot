from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from localpilot.desktop_state import DesktopUIState
from localpilot.desktop_update_status import check_for_updates, current_update_status
from localpilot.evolution_orchestrator import EvolutionRunLease
from localpilot.foreground import active_foreground_turns

CHECK_INTERVAL_SECONDS = 30 * 60
HANDOFF_FILENAME = "desktop-update-handoff.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def update_check_due(state: DesktopUIState, *, now: datetime | None = None) -> bool:
    values = state.read()
    checked = _parse_timestamp(values.get("update_last_checked"))
    if checked is None:
        return True
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (current - checked).total_seconds() >= CHECK_INTERVAL_SECONDS


def evolution_active(data_dir: str | Path) -> bool:
    path = Path(data_dir).resolve() / "evolution-run.lock"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        return True
    owner = EvolutionRunLease._decode(raw)
    return EvolutionRunLease._owner_live(owner)


def update_safe_boundary(data_dir: str | Path) -> bool:
    data_path = Path(data_dir).resolve()
    if active_foreground_turns(data_path):
        return False
    return not evolution_active(data_path)


def _run(root: Path, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=timeout,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0,
    )


def _pythonw_executable() -> Path:
    executable = Path(sys.executable).resolve()
    if os.name != "nt" or executable.name.lower() == "pythonw.exe":
        return executable
    pythonw = executable.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else executable


def prepare_update_handoff(
    root: str | Path,
    state: DesktopUIState,
    *,
    config_path: str | Path | None,
    remote: str,
    main_branch: str,
) -> Path | None:
    root_path = Path(root).resolve()
    values = state.read()
    if not bool(values.get("automatic_updates")):
        return None

    status = current_update_status(root_path, state)
    if update_check_due(state):
        status = check_for_updates(
            root_path,
            state,
            remote=remote,
            main_branch=main_branch,
        )
    if status.check_error or status.update_available is not True:
        return None

    data_dir = state.path.parent
    if not update_safe_boundary(data_dir):
        return None

    branch = _run(root_path, ["git", "branch", "--show-current"])
    dirty = _run(root_path, ["git", "status", "--porcelain", "--untracked-files=all"])
    head = _run(root_path, ["git", "rev-parse", "--verify", "HEAD^{commit}"])
    remote_ref = f"refs/remotes/{remote}/{main_branch}"
    target = _run(root_path, ["git", "rev-parse", "--verify", f"{remote_ref}^{{commit}}"])
    if (
        branch.returncode != 0
        or branch.stdout.strip() != main_branch
        or dirty.returncode != 0
        or dirty.stdout.strip()
        or head.returncode != 0
        or target.returncode != 0
    ):
        state.update(update_check_error="Automatic update was deferred because trusted main changed during handoff preparation.")
        return None

    old_sha = head.stdout.strip()
    target_sha = target.stdout.strip()
    if old_sha == target_sha:
        state.update(update_available=False, update_target_version=target_sha[:7], update_check_error=None)
        return None
    ancestor = _run(root_path, ["git", "merge-base", "--is-ancestor", old_sha, target_sha])
    if ancestor.returncode != 0:
        state.update(update_check_error="Automatic update was refused because local main is ahead of or diverged from the remote.")
        return None

    payload = {
        "version": 1,
        "attempt_id": uuid.uuid4().hex,
        "created_at": _utc_now(),
        "root": str(root_path),
        "data_dir": str(data_dir),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "remote": remote,
        "main_branch": main_branch,
        "old_sha": old_sha,
        "target_sha": target_sha,
    }
    handoff = data_dir / HANDOFF_FILENAME
    temporary = handoff.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, handoff)
    return handoff


def launch_update_if_ready(
    root: str | Path,
    state: DesktopUIState,
    *,
    config_path: str | Path | None,
    remote: str,
    main_branch: str,
    parent_pid: int | None = None,
) -> bool:
    handoff = prepare_update_handoff(
        root,
        state,
        config_path=config_path,
        remote=remote,
        main_branch=main_branch,
    )
    if handoff is None:
        return False

    root_path = Path(root).resolve()
    argv = [
        str(_pythonw_executable()),
        "-m",
        "localpilot.desktop_updater",
        "--root",
        str(root_path),
        "--handoff",
        str(handoff),
        "--parent-pid",
        str(int(parent_pid or os.getpid())),
    ]
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        ) | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        start_new_session = True
    try:
        subprocess.Popen(
            argv,
            cwd=str(root_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
            start_new_session=start_new_session,
            close_fds=True,
        )
    except OSError as exc:
        state.update(update_check_error=f"Could not start the external updater: {type(exc).__name__}: {exc}"[:1000])
        return False
    return True
