from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from localpilot.config import load_config
from localpilot.desktop import BrokerClient
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
        "-PythonExecutable",
        str(_console_python()),
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
    """Require a desktop process; a restarted background worker is insufficient."""
    for process in psutil.process_iter():
        try:
            if _belongs_to_localpilot(process, root) and _process_role(list(process.cmdline())) in {"chat", "avatar"}:
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


_PROCESS_ROLES = {
    "localpilot.broker": "broker",
    "localpilot.runtime_worker": "runtime",
    "localpilot.background_worker": "worker",
    "localpilot.webview_app": "chat",
    "localpilot.native_avatar": "avatar",
    "localpilot.native_avatar_companion": "avatar",
    "localpilot.native_avatar_legacy": "avatar",
    "localpilot.desktop": "chat",
}


def _argument(argv: list[str], name: str) -> str | None:
    values = []
    for index, value in enumerate(argv):
        if value == name:
            if index + 1 == len(argv):
                raise ValueError(f"Missing {name} value")
            values.append(argv[index + 1])
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1:])
    if len(values) > 1:
        raise ValueError(f"Ambiguous {name} value")
    return values[0] if values else None


def _process_role(argv: list[str]) -> str | None:
    try:
        module_index = argv.index("-m") + 1
        module = argv[module_index]
    except (ValueError, IndexError):
        module_index = 0
        module = "localpilot.cli" if argv and Path(argv[0]).name.casefold() == "localpilot.exe" else ""
    if module == "localpilot.cli":
        arguments = argv[module_index + 1:]
        index = 0
        while index < len(arguments):
            value = arguments[index]
            if value == "--config":
                index += 2
            elif value.startswith("--config="):
                index += 1
            else:
                if value != "desktop":
                    return None
                return "chat" if "--tkinter" in arguments[index + 1:] else "avatar"
        return None
    return _PROCESS_ROLES.get(module)


def _process_path(value: str, cwd: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(cwd) / path).resolve()


def _belongs_to_localpilot(process: psutil.Process, root: Path) -> bool:
    if process.pid == os.getpid():
        return False
    try:
        argv = [str(item) for item in process.cmdline()]
        if _process_role(argv) is None:
            return False
        cwd = process.cwd()
        selected_root = _argument(argv, "--root")
        return (_process_path(selected_root, cwd) if selected_root is not None else Path(cwd).resolve()) == root.resolve()
    except (psutil.Error, OSError, RuntimeError, ValueError):
        return False


@dataclass(frozen=True)
class _ProcessIdentity:
    process: Any
    role: str
    created_at: float
    executable: str
    argv: tuple[str, ...]


def _current_process(identity: _ProcessIdentity) -> bool:
    try:
        process = psutil.Process(identity.process.pid)
        return (
            abs(process.create_time() - identity.created_at) < 0.001
            and process.exe() == identity.executable
            and tuple(process.cmdline()) == identity.argv
        )
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError) as exc:
        raise RuntimeError("Unable to verify the LocalPilot process before graceful shutdown.") from exc


def _request_window_close(identity: _ProcessIdentity) -> None:
    if os.name != "nt":
        raise RuntimeError("LocalPilot's desktop must be closed before this update.")
    if not _current_process(identity):
        return
    import ctypes

    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    configured = _argument(list(identity.argv), "--config")
    cwd = identity.process.cwd()
    config = load_config(_process_path(configured or "localpilot.toml", cwd))
    tk_caption = f"{config.agent.name} · Desktop" if identity.role == "chat" else config.agent.name

    def visit(hwnd: int, _parameter: int) -> bool:
        owner = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(owner))
        if owner.value != identity.process.pid:
            return True
        caption = ctypes.create_unicode_buffer(256)
        window_class = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(ctypes.c_void_p(hwnd), caption, len(caption))
        user32.GetClassNameW(ctypes.c_void_p(hwnd), window_class, len(window_class))
        chat_window = (
            identity.role == "chat"
            and caption.value == "LocalPilot"
            and window_class.value.startswith("WindowsForms10.Window.")
        )
        tk_window = window_class.value == "TkTopLevel" and caption.value == tk_caption
        if not chat_window and not tk_window:
            return True
        # Minimize keeps the same WebView process and draft in a hidden Form.
        # Its normal closing guard still admits or vetoes this WM_CLOSE.
        if identity.role == "chat" or user32.IsWindowVisible(ctypes.c_void_p(hwnd)):
            if _current_process(identity):
                user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(owner))
                if owner.value == identity.process.pid:
                    user32.PostMessageW(ctypes.c_void_p(hwnd), 0x0010, 0, 0)  # WM_CLOSE
        return True

    user32.EnumWindows(callback_type(visit), 0)


def _wait_for_identity_exit(identity: _ProcessIdentity, deadline: float) -> None:
    if not _current_process(identity):
        return
    try:
        identity.process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except psutil.NoSuchProcess:
        return
    except (psutil.Error, OSError) as exc:
        raise RuntimeError(
            "LocalPilot did not close gracefully; no process was force-terminated. "
            "Finish the reply, save any draft, and close its windows before retrying."
        ) from exc


def _stop_localpilot_processes(
    root: Path, *, config_path: str | Path | None = None,
    include_background_worker: bool = True, timeout: float = 20.0,
) -> None:
    root = root.resolve()
    selected_config = Path(config_path or root / "localpilot.toml").resolve()
    matches = []
    for process in psutil.process_iter():
        if not _belongs_to_localpilot(process, root):
            continue
        try:
            argv = [str(value) for value in process.cmdline()]
            role = _process_role(argv)
            if role is None:
                raise ValueError("LocalPilot process role changed during identity capture")
            if role == "worker" and not include_background_worker:
                continue
            configured = _argument(argv, "--config")
            if configured is not None and _process_path(configured, process.cwd()) != selected_config:
                raise RuntimeError("A LocalPilot process uses another configuration; it was left running.")
            matches.append(_ProcessIdentity(process, role, process.create_time(), process.exe(), tuple(argv)))
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError, ValueError) as exc:
            raise RuntimeError("LocalPilot process identity could not be verified; it was left running.") from exc
    if not matches:
        return
    # Helpers receive EOF when their owning runtime exits. Track only verified
    # children of these processes so another checkout's provider is untouched.
    for identity in tuple(matches):
        if identity.role not in {"broker", "runtime"}:
            continue
        for child in identity.process.children():
            try:
                executable = Path(child.exe()).resolve()
                if executable.name == "LocalPilot.SystemSense.HardwareProvider.exe" and executable.is_relative_to(root / "localpilot" / "_hardware"):
                    matches.append(_ProcessIdentity(child, "helper", child.create_time(), child.exe(), tuple(child.cmdline())))
            except psutil.NoSuchProcess:
                continue
    config = load_config(selected_config)
    data_dir = (root / config.agent.data_dir).resolve()
    workers = [identity for identity in matches if identity.role == "worker"]
    worker_owner = None
    if workers:
        try:
            worker_owner = json.loads((data_dir / "background-worker.pid").read_text(encoding="utf-8"))
            owner_pid = int(worker_owner["pid"])
            if Path(worker_owner["root"]).resolve() != root or owner_pid not in {identity.process.pid for identity in workers}:
                raise ValueError("Worker owner does not match this configured process")
            worker_owner["pid"] = owner_pid
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("The background worker's PID ownership is unavailable; it was left running.") from exc
    brokers = [identity for identity in matches if identity.role == "broker"]
    client = None
    broker = None
    if brokers:
        client = BrokerClient(root, config)
        try:
            status = client.request("GET", "/v1/broker/status", timeout=2.0)
            broker = next(identity for identity in brokers if identity.process.pid == status.get("pid"))
            if not _current_process(broker):
                raise RuntimeError("The broker identity changed before shutdown.")
            if status.get("pending_requests") != 0:
                raise RuntimeError("LocalPilot is still responding; finish the response before retrying.")
        except (OSError, ValueError, StopIteration) as exc:
            raise RuntimeError(
                "LocalPilot could not stop its broker safely. Let any response finish before retrying; "
                "its desktop and worker were left running."
            ) from exc
    elif any(identity.role == "runtime" for identity in matches):
        raise RuntimeError("LocalPilot's runtime has no verified broker; close the existing runtime before retrying.")
    deadline = time.monotonic() + timeout
    for role in ("chat", "avatar"):
        group = [identity for identity in matches if identity.role == role]
        for identity in group:
            _request_window_close(identity)
        for identity in group:
            _wait_for_identity_exit(identity, deadline)
    if client is not None and broker is not None and _current_process(broker):
        try:
            result = client.request("POST", "/v1/broker/shutdown", {}, timeout=2.0)
            if result.get("status") != "stopping" or result.get("pid") != broker.process.pid:
                raise RuntimeError("The broker did not acknowledge graceful shutdown.")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "LocalPilot's broker did not stop safely. Let any response finish before retrying; "
                "the broker and worker were not force-terminated."
            ) from exc
    if worker_owner is not None:
        owner = next(identity for identity in workers if identity.process.pid == worker_owner["pid"])
        if _current_process(owner):
            request = {"target_pid": owner.process.pid, "requested_at": _utc_now()}
            (data_dir / "background-worker.stop").write_text(json.dumps(request), encoding="utf-8")
    for identity in matches:
        if identity.role not in {"chat", "avatar"}:
            _wait_for_identity_exit(identity, deadline)
    # A watchdog or concurrent launch can create a new PID after the snapshot.
    # Refuse to update loaded source rather than assuming the old set was final.
    for process in psutil.process_iter():
        if not _belongs_to_localpilot(process, root):
            continue
        try:
            current = psutil.Process(process.pid)
            if not include_background_worker and _process_role(list(current.cmdline())) == "worker":
                continue
        except psutil.NoSuchProcess:
            continue
        raise RuntimeError("A LocalPilot process remains or appeared during graceful shutdown; retry after it closes.")


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
    checked = _run(root, [str(python), "-m", "pip", "check"], timeout=60)
    if checked.returncode != 0:
        raise RuntimeError(checked.stderr.strip() or checked.stdout.strip() or "Updated LocalPilot dependencies are inconsistent.")
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
        # handed off; dependency installation can still download packages.
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
                _stop_localpilot_processes(root_path, config_path=config_path)
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
