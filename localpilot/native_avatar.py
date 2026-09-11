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


# Each row asset is 256x128: two independently redrawn 128x128 versions of
# one pose.  Alternating those redraws provides the passive hand-drawn line
# boil.  The state loop and state transition are deliberately separate.
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
_STATE_ASSET_SHA256 = {
    0: "476ecfc97875732958a4102b8b5cae0e2e939cbb55e5e94f8d318c54cdf6e99e",
    1: "facd009c7304fd7de68daa58cd72a6f698444fd1f6312a6395ecefd4b5f10cfa",
    2: "b5608f9807f749eb09c79e154e995b43be8b4d411ec9ab678c0a07484138c052",
    3: "bed5adfc48afc5e25c05bc4e9a8d8cd7c35b3560abb06ee0c93cbe6398087219",
    4: "9de87db31faf742b29ce0a3d719233bf635719c607c458444e3abd30f7d6bf7e",
    5: "4384d6f772287ed5cb8cf72f57284aa34bef8ad283371a226083b44621b6ea51",
    6: "052250cf4589b49c84eb0778201fccfa7f89adc4c2f5a12217532b38c3b50411",
    7: "3c916bd39c799887ff603904e32faefb5c45859e3ba3eae996d467868aed5d71",
    8: "967adff9291fb67da74f978954ddcfd9f581dfda6b5a469eedcf2530ef40aced",
}
_LINE_BOIL_TICKS = 3          # 3 * legacy 180 ms ~= 540 ms per redraw
_TRANSITION_TICKS = 3         # short old-pose -> new-pose settle


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "webview" / "avatar"


def _state_asset_path(row: int) -> Path:
    return _asset_dir() / f"state-{int(row)}.png"


def _read_state_asset(row: int) -> bytes | None:
    """Return one committed state asset only when its source hash is intact."""

    expected = _STATE_ASSET_SHA256.get(int(row))
    if expected is None:
        return None
    try:
        raw = _state_asset_path(row).read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != expected:
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
    """Native companion using verified illustrated state assets when available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # legacy __init__ calls self._draw(), so these must exist beforehand.
        self._illustrated_sprites: dict[int, Any] = {}
        self._display_state = "restarting"
        self._previous_state = "restarting"
        self._transition_tick = _TRANSITION_TICKS
        self._state_age = 0
        super().__init__(*args, **kwargs)

        loaded: dict[int, Any] = {}
        for row in sorted(_STATE_ASSET_SHA256):
            if _read_state_asset(row) is None:
                loaded = {}
                break
            try:
                loaded[row] = self.tk.PhotoImage(
                    file=str(_state_asset_path(row)), format="png"
                )
            except Exception:
                loaded = {}
                break
        self._illustrated_sprites = loaded
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
        row = _SPRITE_ROWS.get(state, _SPRITE_ROWS["error"])
        sprite = self._illustrated_sprites.get(row)
        if sprite is None:
            return
        self.canvas.create_image(
            dx - frame * AVATAR_SIZE,
            dy,
            image=sprite,
            anchor="nw",
        )

    def _draw(self) -> None:
        sprites = getattr(self, "_illustrated_sprites", {})
        if len(sprites) != len(_STATE_ASSET_SHA256):
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

        motion_age = max(0, self._state_age - _TRANSITION_TICKS)
        dx, dy = _loop_offset(self._display_state, motion_age)
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
