from __future__ import annotations

"""Persistent illustrated desktop companion with a separate chat surface.

This module keeps the real native Astra avatar alive while the WebView chat is
open. It deliberately subclasses the production illustrated avatar so there is
one visible avatar renderer, one animation state machine, and one persisted
desktop position. The WebView is only the conversation surface.
"""

import ctypes
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from localpilot import native_avatar as _avatar
from localpilot.desktop_auto_update import launch_update_if_ready
from localpilot.desktop_instance import DesktopInstance, assert_no_legacy_desktop
from localpilot.webview_app import EXPANDED_SIZE, _desktop_python_executable

CHAT_GAP = 12
AUTO_UPDATE_POLL_MS = 15_000


def _chat_position_from_avatar(
    x: int,
    y: int,
    work_area: tuple[int, int, int, int] | None = None,
    *, scale: float = 1.0,
) -> tuple[int, int]:
    """Place chat beside the avatar without covering it.

    Prefer the left side because the default Astra position is tied to the
    Windows clock corner. Fall back to the right side if the avatar is near a
    monitor's left edge, then clamp the complete chat window to that monitor.
    """

    width, height = (round(value * scale) for value in EXPANDED_SIZE)
    if work_area is None:
        work_area = _avatar._monitor_work_area_for_point(
            int(x) + _avatar.AVATAR_SIZE // 2,
            int(y) + _avatar.AVATAR_SIZE // 2,
        )
    if work_area is None:
        work_area = _avatar._primary_work_area()

    raw_x = int(x) - width - CHAT_GAP
    raw_y = int(y) + _avatar.AVATAR_SIZE - height
    if work_area is not None:
        left, _top, right, _bottom = work_area
        if raw_x < left:
            right_candidate = int(x) + _avatar.AVATAR_SIZE + CHAT_GAP
            if right_candidate + width <= right:
                raw_x = right_candidate
    return _avatar._clamp_position(raw_x, raw_y, width, height, work_area)


def _launch_chat(
    root: Path,
    config_path: str | None,
    *,
    x: int,
    y: int,
    tail_left: bool = False,
) -> subprocess.Popen[Any] | None:
    executable = _desktop_python_executable()
    argv = [
        str(executable),
        "-m",
        "localpilot.webview_app",
        "--root",
        str(root),
        "--x",
        str(int(x)),
        "--y",
        str(int(y)),
        "--companion",
        "--physical-position",
    ]
    if tail_left:
        argv.append("--tail-left")
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    try:
        return subprocess.Popen(
            argv,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
            close_fds=True,
        )
    except OSError:
        return None


class NativeAvatarCompanion(_avatar.NativeAvatarApp):
    """The production illustrated avatar that remains alive beside chat."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._chat_process: subprocess.Popen[Any] | None = None
        self._chat_process_created_at: float | None = None
        self._broker_runtime_state = "restarting"
        self._update_thread: threading.Thread | None = None
        self._update_handoff_started = False
        self._chat_launch_lock = threading.Lock()
        self._desktop_instance: DesktopInstance | None = None
        super().__init__(*args, **kwargs)
        self.root.after(2_500, self._poll_auto_update)

    def _chat_is_alive(self) -> bool:
        process = self._chat_process
        return process is not None and process.poll() is None

    def _poll_auto_update(self) -> None:
        if self._stop.is_set() or self._update_handoff_started:
            return
        thread = self._update_thread
        if thread is None or not thread.is_alive():
            def worker() -> None:
                try:
                    # An unsent composer is not a broker foreground request.
                    # Keep this same chat process until the owner closes it.
                    if not self._chat_is_alive() and self.config.github.enabled:
                        launch_update_if_ready(
                            self.project_root,
                            self.state_store,
                            config_path=self.config_path,
                            remote=self.config.github.remote,
                            main_branch=self.config.github.main_branch,
                            parent_pid=os.getpid(),
                            commit_launch=self._commit_update_launch,
                        )
                except Exception as exc:
                    self.state_store.update(
                        update_check_error=(
                            f"Automatic update check failed: {type(exc).__name__}: {exc}"
                        )[:1000]
                    )
            self._update_thread = threading.Thread(
                target=worker,
                name="LocalPilotDesktopAutoUpdate",
                daemon=True,
            )
            self._update_thread.start()
        if not self._stop.is_set():
            self.root.after(AUTO_UPDATE_POLL_MS, self._poll_auto_update)

    def _commit_update_launch(self, spawn: Any) -> bool:
        with self._chat_launch_lock:
            if self._stop.is_set() or self._chat_is_alive() or self._update_handoff_started:
                return False
            if spawn():
                self._update_handoff_started = True
                return True
            return False

    def _drain_events(self) -> None:
        """Merge broker state with the chat's client-local presentation state.

        The broker remains authoritative for runtime/tool/error transitions.
        While chat is open, its frontend can additionally enter real UI states
        that the broker does not emit (notably ``listening`` while the composer
        is focused and ``speaking`` while response text is being revealed).
        The chat publishes those states through DesktopUIState so the one
        visible native Astra always matches what the conversation surface says.
        """

        try:
            while True:
                state = self._events.get_nowait()
                if state in _avatar._STATE_COLORS:
                    self._broker_runtime_state = state
        except queue.Empty:
            pass

        if getattr(self, "_update_handoff_started", False):
            self.runtime_state = "restarting"
            self._draw()
            self.close()
            return

        instance = getattr(self, "_desktop_instance", None)
        if instance is not None and instance.replacement_requested():
            if not self._chat_is_alive():
                self.close()
                return
            self._focus_chat()
        if instance is not None and instance.activation_requested():
            self.root.deiconify()
            self.root.lift()
            self.open_chat()

        display_state = self._broker_runtime_state
        if self._chat_is_alive():
            shared_state = self.state_store.read().get("companion_state")
            if isinstance(shared_state, str) and shared_state in _avatar._STATE_COLORS:
                display_state = shared_state

        self.runtime_state = display_state
        self._draw()
        if not self._stop.is_set():
            self.root.after(80, self._drain_events)

    def open_chat(self) -> None:
        with self._chat_launch_lock:
            if self._update_handoff_started:
                return
            self._open_chat()

    def _open_chat(self) -> None:
        if self._stop.is_set():
            return
        if self._chat_is_alive():
            self._focus_chat()
            return

        self.x, self.y = _avatar._recover_avatar_position(self.x, self.y)
        _avatar._set_window_position(self.root, self.x, self.y)
        self.state_store.update(avatar_x=self.x, avatar_y=self.y, companion_state=None)
        scale = 1.0
        if os.name == "nt":
            try:
                get_dpi = ctypes.windll.user32.GetDpiForWindow
                get_dpi.argtypes = [ctypes.c_void_p]
                get_dpi.restype = ctypes.c_uint
                scale = (get_dpi(_avatar._top_level_hwnd(self.root)) or 96) / 96
            except (AttributeError, OSError):
                pass
        chat_x, chat_y = _chat_position_from_avatar(self.x, self.y, scale=scale)
        process = _launch_chat(
            self.project_root,
            self.config_path,
            x=chat_x,
            y=chat_y,
            tail_left=chat_x >= self.x + _avatar.AVATAR_SIZE,
        )
        if process is None:
            self.runtime_state = "error"
            self._broker_runtime_state = "error"
            self._draw()
            return

        self._chat_process = process
        try:
            self._chat_process_created_at = psutil.Process(process.pid).create_time()
        except (psutil.Error, AttributeError):
            self._chat_process_created_at = None
        self._opening_chat = False
        self.root.after(400, self._watch_chat_process)

    def _focus_chat(self) -> None:
        """Raise the existing window without replacing its conversation/draft."""
        process = self._chat_process
        if process is None or os.name != "nt":
            return
        try:
            created_at = getattr(self, "_chat_process_created_at", None)
            owner = psutil.Process(process.pid)
            if created_at is None or abs(owner.create_time() - created_at) >= 0.001:
                return
            identities = {process.pid: created_at}
            # Windows venv pythonw is a redirector. Its real Python child owns
            # the Form; browser/render subprocesses are not desktop windows.
            for child in owner.children(recursive=True):
                try:
                    argv = [str(value) for value in child.cmdline()]
                    if not any(argv[index:index + 2] == ["-m", "localpilot.webview_app"] for index in range(len(argv) - 1)):
                        continue
                    selected_root = Path(argv[argv.index("--root") + 1]).resolve()
                    if selected_root == self.project_root:
                        identities[child.pid] = child.create_time()
                except (psutil.Error, OSError, ValueError, IndexError):
                    continue
            user32 = ctypes.windll.user32
            callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

            def visit(hwnd: int, _parameter: int) -> bool:
                window_pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(window_pid))
                identity = identities.get(window_pid.value)
                if identity is not None:
                    title = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(ctypes.c_void_p(hwnd), title, len(title))
                    if title.value != "LocalPilot":
                        return True
                    try:
                        if abs(psutil.Process(window_pid.value).create_time() - identity) >= 0.001:
                            return True
                    except psutil.Error:
                        return True
                    user32.ShowWindow(ctypes.c_void_p(hwnd), 9)  # SW_RESTORE
                    user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
                    return False
                return True

            user32.EnumWindows(callback_type(visit), 0)
        except (AttributeError, OSError, psutil.Error):
            pass

    def _watch_chat_process(self) -> None:
        process = self._chat_process
        if self._stop.is_set() or process is None:
            return
        if process.poll() is None:
            self.root.after(500, self._watch_chat_process)
            return
        self._chat_process = None
        self._chat_process_created_at = None
        self._opening_chat = False
        self.state_store.update(companion_state=None)
        self.runtime_state = self._broker_runtime_state
        self._draw()

    def close(self) -> None:
        process = self._chat_process
        if process is not None and process.poll() is None:
            # A running composer may contain a draft even without an active
            # broker request. Let the chat close normally before the owner exits.
            self._focus_chat()
            return
        self._chat_process = None
        self.state_store.update(companion_state=None)
        super().close()


def main(
    root: str | Path,
    config_path: str | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
    open_chat: bool = False,
) -> None:
    project_root = Path(root).resolve()
    instance = DesktopInstance(project_root, config_path)
    if not instance.acquire():
        if instance.request_activation():
            return
        deadline = time.monotonic() + 5.0
        while not instance.acquire():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "The existing LocalPilot desktop is still open. Finish the reply and save your draft, "
                    "then close its chat before opening the updated version."
                )
            time.sleep(0.05)
    try:
        assert_no_legacy_desktop(project_root, retired_owners=getattr(instance, "retired_owners", None))
        config = _avatar._legacy.load_config(config_path)
        client = _avatar._legacy.ensure_broker(project_root, config, config_path=config_path)
        app = NativeAvatarCompanion(
            client,
            config,
            project_root,
            config_path=config_path,
            initial_x=x,
            initial_y=y,
        )
        app._desktop_instance = instance
        if open_chat:
            app.root.after(100, app.open_chat)
        app.run()
    finally:
        instance.close()


def build_parser():
    return _avatar.build_parser()


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config, x=args.x, y=args.y)


if __name__ == "__main__":
    cli_main()
