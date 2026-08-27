from __future__ import annotations

import argparse
import ctypes
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from localpilot.config import Config, load_config
from localpilot.desktop import BrokerClient, ensure_broker
from localpilot.desktop_state import DesktopUIState
from localpilot.webview_app import EXPANDED_SIZE, _desktop_python_executable

AVATAR_SIZE = 128
EDGE_INSET = 24
_TRANSPARENT_KEY = "#010203"
_STATE_COLORS = {
    "idle": "#63d9b8",
    "listening": "#7eead1",
    "thinking": "#f0c24e",
    "working": "#5fa8ff",
    "speaking": "#b98cff",
    "success": "#6fde8e",
    "uncertain": "#f0a24e",
    "error": "#ff6b6b",
    "restarting": "#8592a8",
    "sleeping": "#63d9b8",
    "offline": "#4b5568",
}


def _primary_work_area(*, platform_name: str | None = None) -> tuple[int, int, int, int] | None:
    """Return the primary Windows work area as absolute virtual-desktop coordinates."""
    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt":
        return None

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    try:
        ok = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def _initial_avatar_position(
    state: dict[str, Any],
    work_area: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    saved_x = state.get("avatar_x")
    saved_y = state.get("avatar_y")
    if isinstance(saved_x, int) and isinstance(saved_y, int):
        return saved_x, saved_y
    if work_area is not None:
        left, top, right, bottom = work_area
        return right - AVATAR_SIZE - EDGE_INSET, bottom - AVATAR_SIZE - EDGE_INSET
    return EDGE_INSET, EDGE_INSET


def _chat_position_from_avatar(x: int, y: int) -> tuple[int, int]:
    """Keep the expanded chat's bottom-right corner anchored to the avatar."""
    width, height = EXPANDED_SIZE
    return x + AVATAR_SIZE - width, y + AVATAR_SIZE - height


def _launch_webview(
    root: Path,
    config_path: str | None,
    *,
    x: int,
    y: int,
) -> bool:
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
        subprocess.Popen(
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
        return False
    return True


def _set_window_position(root: Any, x: int, y: int) -> None:
    """Move by absolute virtual-desktop coordinates; Tk's negative geometry is ambiguous."""
    if os.name == "nt":
        try:
            root.update_idletasks()
            hwnd = int(root.winfo_id())
            flags = 0x0001 | 0x0010  # SWP_NOSIZE | SWP_NOACTIVATE
            ctypes.windll.user32.SetWindowPos(hwnd, -1, int(x), int(y), 0, 0, flags)
            return
        except (AttributeError, OSError, ValueError):
            pass
    root.geometry(f"{AVATAR_SIZE}x{AVATAR_SIZE}{int(x):+d}{int(y):+d}")


class NativeAvatarApp:
    """Small Windows-native transparent companion; WebView is used only for chat."""

    def __init__(
        self,
        client: BrokerClient,
        config: Config,
        root: str | Path,
        *,
        config_path: str | None = None,
        initial_x: int | None = None,
        initial_y: int | None = None,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.client = client
        self.config = config
        self.project_root = Path(root).resolve()
        self.config_path = config_path
        self.state_store = DesktopUIState(self.project_root / config.agent.data_dir)
        stored = self.state_store.read()
        if initial_x is not None and initial_y is not None:
            stored["avatar_x"] = int(initial_x)
            stored["avatar_y"] = int(initial_y)
        self.x, self.y = _initial_avatar_position(stored, _primary_work_area())
        self.always_on_top = bool(stored.get("always_on_top", True))

        self.root = tk.Tk()
        self.root.title(config.agent.name)
        self.root.overrideredirect(True)
        self.root.resizable(False, False)
        self.root.configure(bg=_TRANSPARENT_KEY)
        if os.name == "nt":
            self.root.wm_attributes("-transparentcolor", _TRANSPARENT_KEY)
        self.root.wm_attributes("-topmost", self.always_on_top)
        self.root.geometry(f"{AVATAR_SIZE}x{AVATAR_SIZE}+0+0")
        self.root.update_idletasks()
        _set_window_position(self.root, self.x, self.y)

        self.canvas = tk.Canvas(
            self.root,
            width=AVATAR_SIZE,
            height=AVATAR_SIZE,
            bg=_TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)
        self.runtime_state = "restarting"
        self.frame = 0
        self._drag_origin: tuple[int, int, int, int] | None = None
        self._dragged = False
        self._stop = threading.Event()
        self._events: queue.Queue[str] = queue.Queue()
        self._after_event_id = 0

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._draw()
        threading.Thread(target=self._poll_events, daemon=True).start()
        self.root.after(80, self._drain_events)
        self.root.after(180, self._animate)

    def _press(self, event: Any) -> None:
        self._drag_origin = (int(event.x_root), int(event.y_root), self.x, self.y)
        self._dragged = False

    def _drag(self, event: Any) -> None:
        if self._drag_origin is None:
            return
        mouse_x, mouse_y, start_x, start_y = self._drag_origin
        dx = int(event.x_root) - mouse_x
        dy = int(event.y_root) - mouse_y
        if abs(dx) + abs(dy) >= 4:
            self._dragged = True
        self.x = start_x + dx
        self.y = start_y + dy
        _set_window_position(self.root, self.x, self.y)

    def _release(self, _event: Any) -> None:
        self.state_store.update(avatar_x=self.x, avatar_y=self.y)
        self._drag_origin = None
        if not self._dragged:
            self.open_chat()

    def _poll_events(self) -> None:
        while not self._stop.is_set():
            try:
                query = urllib.parse.urlencode({"after": self._after_event_id, "wait": 20})
                events = self.client.request("GET", f"/v1/events?{query}", timeout=25).get("events", [])
                for event in events:
                    self._after_event_id = max(self._after_event_id, int(event.get("id") or 0))
                    event_type = str(event.get("type") or "")
                    payload = dict(event.get("payload") or {})
                    if event_type == "runtime.state":
                        self._events.put(str(payload.get("state") or "idle"))
                    elif event_type.startswith("tool."):
                        self._events.put("working")
                    elif event_type in {"runtime.error", "message.failed"}:
                        self._events.put("error")
            except Exception:
                self._events.put("offline")
                self._stop.wait(1.0)

    def _drain_events(self) -> None:
        try:
            while True:
                state = self._events.get_nowait()
                if state in _STATE_COLORS:
                    self.runtime_state = state
        except queue.Empty:
            pass
        self._draw()
        if not self._stop.is_set():
            self.root.after(80, self._drain_events)

    def _animate(self) -> None:
        self.frame += 1
        self._draw()
        if not self._stop.is_set():
            self.root.after(180, self._animate)

    def _draw(self) -> None:
        self.canvas.delete("all")
        color = _STATE_COLORS.get(self.runtime_state, _STATE_COLORS["error"])
        cell = 7
        cols = rows = 15
        offset = (AVATAR_SIZE - cell * cols) // 2
        cx, cy = 7.5, 7.9

        # Sparse square motes make a pixel glow without introducing an opaque
        # backing surface. They are intentionally actual pixels, not alpha blur.
        if self.runtime_state not in {"offline", "error"}:
            mote_phase = self.frame % 6
            for mx, my in ((1 + mote_phase // 2, 4), (13 - mote_phase // 2, 9), (5, 1)):
                self.canvas.create_rectangle(
                    offset + mx * cell,
                    offset + my * cell,
                    offset + (mx + 1) * cell - 1,
                    offset + (my + 1) * cell - 1,
                    fill=color,
                    outline="",
                )

        for gy in range(rows):
            for gx in range(cols):
                u = (gx + 0.5 - cx) / 5.1
                vr = 4.5 if gy < cy else 6.0
                v = (gy + 0.5 - cy) / vr
                if u * u + v * v <= 1.0:
                    x0 = offset + gx * cell
                    y0 = offset + gy * cell
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x0 + cell - 1,
                        y0 + cell - 1,
                        fill=color,
                        outline="",
                    )

        eye = "#f0fbff"
        if self.runtime_state in {"sleeping", "offline"}:
            for ex in (5, 9):
                x0 = offset + ex * cell
                y0 = offset + 6 * cell + cell // 2
                self.canvas.create_rectangle(x0, y0, x0 + cell - 1, y0 + 2, fill=eye, outline="")
        elif self.runtime_state == "restarting" and (self.frame // 3) % 2:
            return
        else:
            eye_width = cell + 3 if self.runtime_state == "listening" else cell
            for ex in (5, 9):
                x0 = offset + ex * cell
                y0 = offset + 6 * cell
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + eye_width - 1,
                    y0 + cell - 1,
                    fill=eye,
                    outline="",
                )

    def open_chat(self) -> None:
        chat_x, chat_y = _chat_position_from_avatar(self.x, self.y)
        self.state_store.update(avatar_x=self.x, avatar_y=self.y)
        if _launch_webview(
            self.project_root,
            self.config_path,
            x=chat_x,
            y=chat_y,
        ):
            self.close()
        else:
            self.runtime_state = "error"
            self._draw()

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self.state_store.update(avatar_x=self.x, avatar_y=self.y)
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(
    root: str | Path,
    config_path: str | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> None:
    project_root = Path(root).resolve()
    config = load_config(config_path)
    client = ensure_broker(project_root, config, config_path=config_path)
    NativeAvatarApp(
        client,
        config,
        project_root,
        config_path=config_path,
        initial_x=x,
        initial_y=y,
    ).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-native-avatar")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    return parser


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config, x=args.x, y=args.y)


if __name__ == "__main__":
    cli_main()
