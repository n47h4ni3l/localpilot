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
from pathlib import Path
from typing import Any

from localpilot import native_avatar as _avatar
from localpilot.webview_app import EXPANDED_SIZE, _desktop_python_executable

CHAT_GAP = 12


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
        self._broker_runtime_state = "restarting"
        super().__init__(*args, **kwargs)

    def _chat_is_alive(self) -> bool:
        process = self._chat_process
        return process is not None and process.poll() is None

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
        if self._stop.is_set():
            return
        if self._chat_is_alive():
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
        self._opening_chat = False
        self.root.after(400, self._watch_chat_process)

    def _watch_chat_process(self) -> None:
        process = self._chat_process
        if self._stop.is_set() or process is None:
            return
        if process.poll() is None:
            self.root.after(500, self._watch_chat_process)
            return
        self._chat_process = None
        self._opening_chat = False
        self.state_store.update(companion_state=None)
        self.runtime_state = self._broker_runtime_state
        self._draw()

    def close(self) -> None:
        process = self._chat_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        self._chat_process = None
        self.state_store.update(companion_state=None)
        super().close()


def main(
    root: str | Path,
    config_path: str | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> None:
    project_root = Path(root).resolve()
    config = _avatar._legacy.load_config(config_path)
    client = _avatar._legacy.ensure_broker(project_root, config, config_path=config_path)
    NativeAvatarCompanion(
        client,
        config,
        project_root,
        config_path=config_path,
        initial_x=x,
        initial_y=y,
    ).run()


def build_parser():
    return _avatar.build_parser()


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config, x=args.x, y=args.y)


if __name__ == "__main__":
    cli_main()
