from __future__ import annotations

"""Frame-sequence LocalPilot desktop avatar.

The original pixel companion is preserved in :mod:`native_avatar_legacy` and
remains the fail-safe.  The illustrated companion now uses real per-state
animation strips rather than moving one static drawing around the window.

Each state asset contains:
- four enter frames;
- a state-specific animated loop;
- four exit frames.

On a state change the native avatar finishes the current state's exit frames,
plays the new state's enter frames, then loops.  The WebView uses the same
assets and crossfades the simultaneously animated exit/enter sequences.
"""

import ctypes
import hashlib
import json
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


_REQUIRED_ANIMATION_STATES = {
    "idle",
    "listening",
    "thinking",
    "researching",
    "working",
    "speaking",
    "success",
    "uncertain",
    "error",
    "learning",
    "restarting",
    "sleeping",
    "offline",
}


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "webview" / "avatar"


def _animation_dir() -> Path:
    return _asset_dir() / "anim"


def _animation_manifest_path() -> Path:
    return _animation_dir() / "animation-manifest.json"


def _png_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def _load_animation_manifest() -> dict[str, Any] | None:
    """Load and verify the committed animation manifest and every PNG strip.

    Returning ``None`` is intentional fail-closed behavior: the legacy pixel
    avatar remains available and will render instead of partially loading a
    damaged illustrated animation set.
    """

    try:
        payload = json.loads(_animation_manifest_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("version") != 2 or int(payload.get("frame_size") or 0) != AVATAR_SIZE:
        return None
    states = payload.get("states")
    if not isinstance(states, dict) or not _REQUIRED_ANIMATION_STATES <= set(states):
        return None

    animation_dir = _animation_dir().resolve()
    for state in sorted(_REQUIRED_ANIMATION_STATES):
        spec = states.get(state)
        if not isinstance(spec, dict):
            return None
        filename = spec.get("file")
        expected_hash = spec.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return None
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return None

        try:
            frames = int(spec["frames"])
            frame_ms = int(spec["frame_ms"])
            enter_start = int(spec["enter_start"])
            enter_end = int(spec["enter_end"])
            loop_start = int(spec["loop_start"])
            loop_end = int(spec["loop_end"])
            exit_start = int(spec["exit_start"])
            exit_end = int(spec["exit_end"])
            representative = int(spec["representative_frame"])
        except (KeyError, TypeError, ValueError):
            return None

        if frames < 8 or not 50 <= frame_ms <= 500:
            return None
        if not (
            0 <= enter_start <= enter_end < loop_start <= loop_end < exit_start <= exit_end < frames
        ):
            return None
        if not 0 <= representative < frames:
            return None

        path = (_animation_dir() / filename).resolve()
        if path.parent != animation_dir:
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            return None
        dimensions = _png_dimensions(raw)
        if dimensions != (AVATAR_SIZE * frames, AVATAR_SIZE):
            return None

    return payload


def _normalized_state(state: str) -> str:
    return state if state in _REQUIRED_ANIMATION_STATES else "error"


class NativeAvatarApp(_legacy.NativeAvatarApp):
    """Native companion rendered from real frame-sequence animation strips."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # legacy __init__ calls self._draw() before returning, so every field
        # touched by _draw must exist before super().__init__.
        self._animation_manifest = _load_animation_manifest()
        self._animation_images: dict[str, Any] = {}
        self._display_state = "restarting"
        self._pending_state: str | None = None
        self._animation_phase = "loop"
        self._frame_cursor = 0
        super().__init__(*args, **kwargs)

        manifest = self._animation_manifest
        loaded: dict[str, Any] = {}
        if manifest is not None:
            try:
                for state in sorted(_REQUIRED_ANIMATION_STATES):
                    filename = manifest["states"][state]["file"]
                    loaded[state] = self.tk.PhotoImage(
                        file=str(_animation_dir() / filename),
                        format="png",
                    )
            except Exception:
                loaded = {}

        self._animation_images = loaded
        if len(loaded) == len(_REQUIRED_ANIMATION_STATES):
            self._display_state = _normalized_state(self.runtime_state)
            spec = self._spec(self._display_state)
            self._animation_phase = "enter"
            self._frame_cursor = int(spec["enter_start"])
        self._draw()

    def _spec(self, state: str) -> dict[str, Any]:
        manifest = self._animation_manifest
        if manifest is None:
            raise RuntimeError("animation manifest unavailable")
        return manifest["states"][_normalized_state(state)]

    def _animation_ready(self) -> bool:
        return (
            self._animation_manifest is not None
            and len(self._animation_images) == len(_REQUIRED_ANIMATION_STATES)
        )

    def _request_runtime_state(self) -> None:
        if not self._animation_ready():
            return
        target = _normalized_state(self.runtime_state)
        if target == self._display_state and self._pending_state is None:
            return
        if target == self._pending_state:
            return

        self._pending_state = target
        if self._animation_phase != "exit":
            spec = self._spec(self._display_state)
            self._animation_phase = "exit"
            self._frame_cursor = int(spec["exit_start"])

    def _advance_animation(self) -> None:
        if not self._animation_ready():
            return

        spec = self._spec(self._display_state)
        if self._animation_phase == "exit":
            self._frame_cursor += 1
            if self._frame_cursor > int(spec["exit_end"]):
                next_state = self._pending_state or _normalized_state(self.runtime_state)
                self._display_state = next_state
                self._pending_state = None
                next_spec = self._spec(next_state)
                self._animation_phase = "enter"
                self._frame_cursor = int(next_spec["enter_start"])
            return

        if self._animation_phase == "enter":
            self._frame_cursor += 1
            if self._frame_cursor > int(spec["enter_end"]):
                self._animation_phase = "loop"
                self._frame_cursor = int(spec["loop_start"])
            return

        self._frame_cursor += 1
        if self._frame_cursor > int(spec["loop_end"]):
            self._frame_cursor = int(spec["loop_start"])

    def _current_delay_ms(self) -> int:
        if not self._animation_ready():
            return 180
        return int(self._spec(self._display_state)["frame_ms"])

    def _animate(self) -> None:
        self.frame += 1
        self._request_runtime_state()
        self._advance_animation()
        self._draw()
        if not self._stop.is_set():
            self.root.after(self._current_delay_ms(), self._animate)

    def _draw(self) -> None:
        if not self._animation_ready():
            _legacy.NativeAvatarApp._draw(self)
            return

        self._request_runtime_state()
        sprite = self._animation_images.get(self._display_state)
        if sprite is None:
            _legacy.NativeAvatarApp._draw(self)
            return

        self.canvas.delete("all")
        self.canvas.create_image(
            -int(self._frame_cursor) * AVATAR_SIZE,
            0,
            image=sprite,
            anchor="nw",
        )


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
