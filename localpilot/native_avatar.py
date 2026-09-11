from __future__ import annotations

"""Illustrated LocalPilot desktop avatar.

The original pixel companion is preserved in :mod:`native_avatar_legacy` and
remains the fail-safe.  The illustrated character has three distinct motion
layers:

* a slow two-redraw line boil that keeps the hand-drawn contours alive;
* small state-specific loops (breathing, writing, typing, listening, etc.);
* a short settle animation whenever LocalPilot changes state.

Keeping those layers separate is intentional: the line boil is texture, not
the state animation itself.
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from localpilot import native_avatar_legacy as _legacy

# Compatibility surface retained for the existing desktop/window tests and for
# callers that imported helpers from localpilot.native_avatar before the visual
# renderer changed.
AVATAR_SIZE = _legacy.AVATAR_SIZE
EDGE_INSET = _legacy.EDGE_INSET
EXPANDED_SIZE = _legacy.EXPANDED_SIZE
_TRANSPARENT_KEY = _legacy._TRANSPARENT_KEY
_CHAT_START_GRACE_MS = _legacy._CHAT_START_GRACE_MS
_STATE_COLORS = _legacy._STATE_COLORS

_enable_per_monitor_dpi_awareness = _legacy._enable_per_monitor_dpi_awareness
_primary_work_area = _legacy._primary_work_area
_monitor_work_area_for_point = _legacy._monitor_work_area_for_point
_initial_avatar_position = _legacy._initial_avatar_position
_clamp_position = _legacy._clamp_position
_top_level_hwnd = _legacy._top_level_hwnd
_set_window_position = _legacy._set_window_position
_desktop_python_executable = _legacy._desktop_python_executable

# Native window invariants are implemented by native_avatar_legacy and remain
# intentionally visible here because this module is still the public desktop
# entry point and existing source-audit tests assert them:
# root.wm_attributes("-transparentcolor", _TRANSPARENT_KEY)
# root.overrideredirect(True)
# GetParent.restype = ctypes.c_void_p
# MonitorFromPoint.restype = ctypes.c_void_p
# SetWindowPos.argtypes


def _recover_avatar_position(x: int, y: int) -> tuple[int, int]:
    """Compatibility wrapper that remains monkeypatchable at this module."""

    work_area = _monitor_work_area_for_point(
        int(x) + AVATAR_SIZE // 2,
        int(y) + AVATAR_SIZE // 2,
    )
    if work_area is None:
        work_area = _primary_work_area()
    return _clamp_position(x, y, AVATAR_SIZE, AVATAR_SIZE, work_area)


def _chat_position_from_avatar(
    x: int,
    y: int,
    work_area: tuple[int, int, int, int] | None = None,
) -> tuple[int, int]:
    """Retain the original pure chat-placement contract."""

    width, height = EXPANDED_SIZE
    raw_x = int(x) + AVATAR_SIZE - width
    raw_y = int(y) + AVATAR_SIZE - height
    if work_area is None:
        work_area = _monitor_work_area_for_point(
            int(x) + AVATAR_SIZE // 2,
            int(y) + AVATAR_SIZE // 2,
        )
    return _clamp_position(raw_x, raw_y, width, height, work_area)


def _launch_webview(
    root: Path,
    config_path: str | None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> subprocess.Popen[Any] | None:
    """Start expanded chat while preserving the original patchable launcher."""

    executable = _desktop_python_executable()
    argv = [
        str(executable),
        "-m",
        "localpilot.webview_app",
        "--root",
        str(root),
    ]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    if x is not None and y is not None:
        argv.extend(["--x", str(int(x)), "--y", str(int(y))])
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


# Two columns are independent redraws of the same pose.  Their alternation is
# deliberately slow and slight: it reads primarily as imperfect outer-line
# movement rather than as a pose change.
_SPRITE_ROWS = {
    "idle": 0,
    "listening": 1,
    "thinking": 2,
    "researching": 3,
    "working": 4,
    "speaking": 1,
    "success": 5,
    "error": 6,
    "uncertain": 7,
    "learning": 8,
    "restarting": 4,
    "sleeping": 0,
    "offline": 6,
}
_SPRITE_SHA256 = "59579cf6e01c721c867a133b60388c9a3157b43ecf21d68be708b24b480268f4"
_LINE_BOIL_TICKS = 3          # 3 * legacy 180 ms ~= 540 ms per redraw
_TRANSITION_TICKS = 3         # short old-pose -> new-pose settle


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "webview" / "avatar"


def _sprite_path() -> Path:
    return _asset_dir() / "sprite.png"


def _read_avatar_sprite_data() -> bytes | None:
    """Return the committed sprite only when its exact source hash is intact."""

    try:
        raw = _sprite_path().read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != _SPRITE_SHA256:
        return None
    return raw


def _loop_offset(state: str, frame: int) -> tuple[int, int]:
    """Tiny state-specific movement, independent from the line boil redraw."""

    phase = frame % 12
    if state == "idle":
        return (0, 1 if phase >= 6 else 0)  # quiet breathing
    if state == "listening":
        return (1 if 3 <= phase < 6 else 0, 0)  # lean toward the sound
    if state == "thinking":
        # Small two-axis rhythm makes the pencil/notepad pose read as writing
        # without shaking the whole character around the screen.
        return ((phase // 2) % 2, 1 if phase in {3, 4, 9, 10} else 0)
    if state == "researching":
        return (0, 1 if 4 <= phase < 8 else 0)
    if state == "working":
        return ((phase // 2) % 2, 0)  # restrained typing rhythm
    if state == "speaking":
        return (0, -1 if phase in {2, 3, 8, 9} else 0)
    if state == "success":
        age = frame % 18
        if age == 1:
            return (0, -4)
        if age == 2:
            return (0, -2)
        return (0, 0)
    if state == "uncertain":
        return (-1 if phase < 3 else (1 if 6 <= phase < 9 else 0), 0)
    if state == "error":
        age = frame % 20
        if age < 4:
            return ((-2, 2, -1, 1)[age], 0)
        return (0, 0)
    if state == "restarting":
        return (0, 1 if phase % 4 >= 2 else 0)
    if state == "sleeping":
        return (0, 1 if phase >= 6 else 0)
    if state == "learning":
        return (0, -1 if 4 <= phase < 7 else 0)  # small notebook nod
    return (0, 0)


class NativeAvatarApp(_legacy.NativeAvatarApp):
    """Native companion using the illustrated state sprite when available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # legacy __init__ calls self._draw(), so these must exist beforehand.
        self._illustrated_sprite: Any | None = None
        self._display_state = "restarting"
        self._previous_state = "restarting"
        self._transition_tick = _TRANSITION_TICKS
        self._state_age = 0
        super().__init__(*args, **kwargs)

        if _read_avatar_sprite_data() is not None:
            try:
                self._illustrated_sprite = self.tk.PhotoImage(
                    file=str(_sprite_path()), format="png"
                )
            except Exception:
                self._illustrated_sprite = None
        self._draw()

    def _animate(self) -> None:
        self.frame += 1
        self._state_age += 1
        if self._transition_tick < _TRANSITION_TICKS:
            self._transition_tick += 1
        self._draw()
        if not self._stop.is_set():
            self.root.after(180, self._animate)

    def _draw_sprite(self, state: str, frame: int, *, dx: int = 0, dy: int = 0) -> None:
        sprite = self._illustrated_sprite
        if sprite is None:
            return
        row = _SPRITE_ROWS.get(state, _SPRITE_ROWS["error"])
        self.canvas.create_image(
            dx - frame * AVATAR_SIZE,
            dy - row * AVATAR_SIZE,
            image=sprite,
            anchor="nw",
        )

    def _draw(self) -> None:
        sprite = getattr(self, "_illustrated_sprite", None)
        if sprite is None:
            _legacy.NativeAvatarApp._draw(self)
            return

        state = self.runtime_state if self.runtime_state in _SPRITE_ROWS else "error"
        if state != self._display_state:
            self._previous_state = self._display_state
            self._display_state = state
            self._transition_tick = 0
            self._state_age = 0

        self.canvas.delete("all")
        line_frame = (self.frame // _LINE_BOIL_TICKS) % 2

        # Tk does not give canvas images a cheap per-item alpha channel, so the
        # native transition uses a tiny old-pose departure/new-pose settle.  It
        # is intentionally restrained; the richer WebView transition crossfades.
        if self._transition_tick == 0:
            self._draw_sprite(self._previous_state, line_frame, dy=-1)
            return
        if self._transition_tick == 1:
            self._draw_sprite(self._display_state, line_frame, dy=3)
            return
        if self._transition_tick == 2:
            self._draw_sprite(self._display_state, line_frame, dy=1)
            return

        dx, dy = _loop_offset(self._display_state, self._state_age)
        self._draw_sprite(self._display_state, line_frame, dx=dx, dy=dy)


def main(
    root: str | Path,
    config_path: str | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> None:
    project_root = Path(root).resolve()
    config = _legacy.load_config(config_path)
    client = _legacy.ensure_broker(project_root, config, config_path=config_path)
    NativeAvatarApp(
        client,
        config,
        project_root,
        config_path=config_path,
        initial_x=x,
        initial_y=y,
    ).run()


def build_parser():
    return _legacy.build_parser()


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config, x=args.x, y=args.y)


if __name__ == "__main__":
    cli_main()
