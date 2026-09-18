from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from localpilot.desktop_state import DesktopUIState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(root: Path, args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
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


def _powershell_executable() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable:
        return executable
    if os.name == "nt":
        return "powershell.exe"
    raise RuntimeError("PowerShell is required for LocalPilot updates.")


def _run_shared_update_script(
    root: Path,
    *,
    branch: str,
    remote: str,
    old_sha: str,
    target_sha: str,
    config_path: str | None,
) -> subprocess.CompletedProcess[str]:
    script = root / "scripts" / "update-and-restart.ps1"
    if not script.is_file():
        raise RuntimeError(f"Shared LocalPilot updater script is missing: {script}")

    argv = [
        _powershell_executable(),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Branch",
        branch,
        "-Remote",
        remote,
        "-SkipFetch",
        "-ExpectedOldSha",
        old_sha,
        "-ExpectedTargetSha",
        target_sha,
    ]
    if config_path:
        argv.extend(["-ConfigPath", str(Path(config_path).resolve())])
    return _run(root, argv, timeout=1800)


def _localpilot_process_running(root: Path) -> bool:
    for process in psutil.process_iter():
        try:
            if _belongs_to_localpilot(process, root):
                return True
        except (psutil.Error, OSError):
            continue
    return False


def _console_python() -> Path:
    executable = Path(sys.executable).resolve()
    if os.name == "nt" and executable.name.lower() == "pythonw.exe":
        python = executable.with_name("python.exe")
        if python.exists():
            return python
    return executable


def _gui_python() -> Path:
    executable = Path(sys.executable).resolve()
    if os.name != "nt" or executable.name.lower() == "pythonw.exe":
        return executable
    pythonw = executable.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else executable


def _load_handoff(path: Path, root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("version") or 0) != 1:
        raise RuntimeError("Unsupported automatic-update handoff format.")
    if Path(str(value.get("root") or "")).resolve() != root:
        raise RuntimeError("Automatic-update handoff root does not match this checkout.")
    old_sha = str(value.get("old_sha") or "").strip()
    target_sha = str(value.get("target_sha") or "").strip()
    if len(old_sha) != 40 or len(target_sha) != 40:
        raise RuntimeError("Automatic-update handoff contains invalid commit identities.")
    return value


def _wait_for_parent_exit(parent_pid: int, timeout: float = 20.0) -> None:
    if parent_pid <= 0:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(parent_pid):
            return
        time.sleep(0.1)
    raise RuntimeError("Desktop process did not exit for automatic update handoff.")


def _belongs_to_localpilot(process: psutil.Process, root: Path) -> bool:
    if process.pid == os.getpid():
        return False
    try:
        cmdline = [str(item) for item in process.cmdline()]
    except (psutil.Error, OSError):
        return False
    if not cmdline:
        return False
    joined = " ".join(cmdline).lower()
    markers = (
        "localpilot.broker",
        "localpilot.runtime_worker",
        "localpilot.native_avatar",
        "localpilot.native_avatar_companion",
        "localpilot.webview_app",
        "localpilot.background_worker",
        "localpilot.exe",
    )
    if not any(marker in joined for marker in markers):
        return False
    root_text = str(root).lower()
    if root_text in joined:
        return True
    try:
        cwd = Path(process.cwd()).resolve()
    except (psutil.Error, OSError, RuntimeError):
        return False
    return cwd == root


def _stop_localpilot_processes(root: Path) -> None:
    matches: list[psutil.Process] = []
    for process in psutil.process_iter():
        try:
            if _belongs_to_localpilot(process, root):
                matches.append(process)
        except (psutil.Error, OSError):
            continue
    for process in matches:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _gone, alive = psutil.wait_procs(matches, timeout=5)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if alive:
        psutil.wait_procs(alive, timeout=5)


def _verify_clean_trusted_main(root: Path, branch: str, old_sha: str, target_sha: str) -> None:
    current_branch = _run(root, ["git", "branch", "--show-current"])
    status = _run(root, ["git", "status", "--porcelain", "--untracked-files=all"])
    head = _run(root, ["git", "rev-parse", "--verify", "HEAD^{commit}"])
    if current_branch.returncode != 0 or current_branch.stdout.strip() != branch:
        raise RuntimeError(f"Automatic update requires trusted {branch}.")
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("Automatic update refused because the main checkout is not clean.")
    if head.returncode != 0 or head.stdout.strip() != old_sha:
        raise RuntimeError("Automatic update refused because local HEAD changed after the update check.")
    ancestor = _run(root, ["git", "merge-base", "--is-ancestor", old_sha, target_sha])
    if ancestor.returncode != 0:
        raise RuntimeError("Automatic update refused because the target is not a fast-forward descendant.")


def _refresh_environment(root: Path) -> None:
    python = _console_python()
    installed = _run(root, [str(python), "-m", "pip", "install", "-e", "."], timeout=600)
    if installed.returncode != 0:
        raise RuntimeError(installed.stderr.strip() or installed.stdout.strip() or "Editable install refresh failed.")
    smoke = _run(root, [str(python), "-c", "import localpilot, localpilot.native_avatar_companion"], timeout=60)
    if smoke.returncode != 0:
        raise RuntimeError(smoke.stderr.strip() or smoke.stdout.strip() or "Updated LocalPilot import check failed.")


def _launch_companion(root: Path, config_path: str | None) -> subprocess.Popen[Any]:
    argv = [str(_gui_python()), "-m", "localpilot.native_avatar_companion", "--root", str(root)]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        ) | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        start_new_session = True
    return subprocess.Popen(
        argv,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        creationflags=creationflags,
        start_new_session=start_new_session,
        close_fds=True,
    )


def _rollback(root: Path, old_sha: str) -> None:
    status = _run(root, ["git", "status", "--porcelain", "--untracked-files=all"])
    if status.returncode != 0 or status.stdout.strip():
        return
    reset = _run(root, ["git", "reset", "--hard", old_sha], timeout=120)
    if reset.returncode == 0:
        try:
            _refresh_environment(root)
        except Exception:
            pass


def apply_handoff(root: str | Path, handoff_path: str | Path, parent_pid: int) -> bool:
    root_path = Path(root).resolve()
    handoff_file = Path(handoff_path).resolve()
    handoff = _load_handoff(handoff_file, root_path)
    data_dir = Path(str(handoff.get("data_dir") or "")).resolve()
    state = DesktopUIState(data_dir)
    config_path = handoff.get("config_path") if isinstance(handoff.get("config_path"), str) else None
    old_sha = str(handoff["old_sha"])
    target_sha = str(handoff["target_sha"])
    branch = str(handoff.get("main_branch") or "main")

    try:
        if not state.read().get("automatic_updates"):
            raise RuntimeError("Automatic updates were disabled before the handoff completed.")
        _wait_for_parent_exit(int(parent_pid))
        _verify_clean_trusted_main(root_path, branch, old_sha, target_sha)

        # The automatic updater uses exactly the same lifecycle as the manual
        # updater. The target commit was fetched and pinned before the desktop
        # handed off, so the shared script performs no network access after the
        # owner-facing process has exited.
        applied = _run_shared_update_script(
            root_path,
            branch=branch,
            remote=str(handoff.get("remote") or "origin"),
            old_sha=old_sha,
            target_sha=target_sha,
            config_path=config_path,
        )
        if applied.returncode != 0:
            detail = applied.stderr.strip() or applied.stdout.strip()
            raise RuntimeError(detail or "Shared LocalPilot update script failed.")

        verified = _run(root_path, ["git", "rev-parse", "--verify", "HEAD^{commit}"])
        if verified.returncode != 0 or verified.stdout.strip() != target_sha:
            raise RuntimeError("Shared updater completed without reaching the pinned target commit.")

        time.sleep(3.0)
        if not _localpilot_process_running(root_path):
            raise RuntimeError("Shared updater completed but the LocalPilot desktop did not remain running.")

        state.update(
            update_last_checked=_utc_now(),
            update_available=False,
            update_target_version=target_sha[:7],
            update_check_error=None,
        )
        handoff["status"] = "completed"
        handoff["completed_at"] = _utc_now()
        handoff_file.write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        try:
            current = _run(root_path, ["git", "rev-parse", "--verify", "HEAD^{commit}"])
            if current.returncode == 0 and current.stdout.strip() == target_sha:
                _stop_localpilot_processes(root_path)
                _rollback(root_path, old_sha)
        except Exception:
            pass
        state.update(
            update_last_checked=_utc_now(),
            update_available=False,
            update_check_error=f"Automatic update failed: {type(exc).__name__}: {exc}"[:1000],
        )
        handoff["status"] = "failed"
        handoff["failed_at"] = _utc_now()
        handoff["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        try:
            handoff_file.write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        try:
            if not _localpilot_process_running(root_path):
                _launch_companion(root_path, config_path)
        except OSError:
            pass
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-desktop-updater")
    parser.add_argument("--root", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(0 if apply_handoff(args.root, args.handoff, args.parent_pid) else 1)


if __name__ == "__main__":
    main()
