from __future__ import annotations

"""Persistent illustrated desktop companion with a separate chat surface.

This module keeps the real native Astra avatar alive while the WebView chat is
open. It deliberately subclasses the production illustrated avatar so there is
one avatar renderer, one animation state machine, and one persisted desktop
position. The WebView is only the conversation surface.
"""

import os
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
) -> tuple[int, int]:
    """Place chat beside the avatar without covering it.

    Prefer the left side because the default Astra position is tied to the
    Windows clock corner. Fall back to the right side if the avatar is near a
    monitor's left edge, then clamp the complete chat window to that monitor.
    """

    width, height = EXPANDED_SIZE
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
    ]
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
        super().__init__(*args, **kwargs)

    def open_chat(self) -> None:
        if self._stop.is_set():
            return
        if self._chat_process is not None and self._chat_process.poll() is None:
            return

        self.x, self.y = _avatar._recover_avatar_position(self.x, self.y)
        _avatar._set_window_position(self.root, self.x, self.y)
        self.state_store.update(avatar_x=self.x, avatar_y=self.y)
        chat_x, chat_y = _chat_position_from_avatar(self.x, self.y)
        process = _launch_chat(
            self.project_root,
            self.config_path,
            x=chat_x,
            y=chat_y,
        )
        if process is None:
            self.runtime_state = "error"
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

    def close(self) -> None:
        process = self._chat_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        self._chat_process = None
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
