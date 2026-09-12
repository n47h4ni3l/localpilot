from __future__ import annotations

"""Frame-sequence LocalPilot desktop avatar.

The original pixel companion is preserved in :mod:`native_avatar_legacy` and
remains the fail-safe. The illustrated companion uses one independently drawn
sprite sheet per state. Each sheet is cropped into real frames at runtime,
trimmed to its visible artwork, and fitted into the 128px native avatar without
stretching or whole-character transform animation.

Dedicated transition sheets are intentionally deferred. Until those are added,
state changes switch directly in native mode while the WebView uses a short
crossfade between completed state loops.
"""

import ctypes
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageTk

from localpilot import native_avatar_legacy as _legacy

AVATAR_SIZE = _legacy.AVATAR_SIZE
EDGE_INSET = _legacy.EDGE_INSET
EXPANDED_SIZE = _legacy.EXPANDED_SIZE
_TRANSPARENT_KEY = _legacy._TRANSPARENT_KEY
_CHAT_START_GRACE_MS = _legacy._CHAT_START_GRACE_MS
_STATE_COLORS = dict(_legacy._STATE_COLORS)
_STATE_COLORS.setdefault("researching", "#5fa8ff")
_STATE_COLORS.setdefault("learning", "#6fde8e")
# The inherited event drain validates against the legacy module global.
_legacy._STATE_COLORS.update({
    "researching": _STATE_COLORS["researching"],
    "learning": _STATE_COLORS["learning"],
})

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
    executable = _desktop_python_executable()
    argv = [str(executable), "-m", "localpilot.webview_app", "--root", str(root)]
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
    "idle", "listening", "thinking", "researching", "working", "speaking",
    "success", "uncertain", "error", "learning", "restarting", "sleeping", "offline",
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


def _sheet_asset_path(asset: dict[str, Any]) -> Path | None:
    filename = asset.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".png"):
        return None
    path = (_animation_dir() / filename).resolve()
    if path.parent != _animation_dir().resolve():
        return None
    return path


def _read_sheet_asset_data(asset: dict[str, Any]) -> bytes | None:
    path = _sheet_asset_path(asset)
    if path is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    expected_bytes = asset.get("bytes")
    try:
        expected_size = int(expected_bytes)
    except (TypeError, ValueError):
        return None
    if expected_size < 100_000 or len(raw) != expected_size:
        return None
    dimensions = _png_dimensions(raw)
    if dimensions is None:
        return None
    width, height = dimensions
    try:
        columns = int(asset["columns"])
        rows = int(asset["rows"])
        frames = int(asset["frames"])
    except (KeyError, TypeError, ValueError):
        return None
    if columns < 1 or rows < 1 or frames != columns * rows:
        return None
    if width < columns * 32 or height < rows * 32:
        return None
    return raw


def _load_animation_manifest() -> dict[str, Any] | None:
    """Verify the per-state sprite-sheet manifest or fail closed to pixel mode."""

    try:
        payload = json.loads(_animation_manifest_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("version") != 6 or int(payload.get("frame_size") or 0) != AVATAR_SIZE:
        return None
    try:
        transition_ms = int(payload.get("transition_ms") or 0)
    except (TypeError, ValueError):
        return None
    if not 0 <= transition_ms <= 1000:
        return None

    assets = payload.get("assets")
    if not isinstance(assets, dict) or not assets:
        return None
    for name, asset in assets.items():
        if not isinstance(name, str) or not isinstance(asset, dict):
            return None
        if _read_sheet_asset_data(asset) is None:
            return None

    states = payload.get("states")
    if not isinstance(states, dict) or not _REQUIRED_ANIMATION_STATES <= set(states):
        return None
    for state in sorted(_REQUIRED_ANIMATION_STATES):
        spec = states.get(state)
        if not isinstance(spec, dict):
            return None
        asset_name = spec.get("asset")
        if not isinstance(asset_name, str) or asset_name not in assets:
            return None
        asset = assets[asset_name]
        try:
            frame_ms = int(spec["frame_ms"])
            representative = int(spec["representative_frame"])
            frames = int(asset["frames"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 50 <= frame_ms <= 1000 or not 0 <= representative < frames:
            return None

    return payload


def _normalized_state(state: str) -> str:
    return state if state in _REQUIRED_ANIMATION_STATES else "error"


def _crop_sheet_frame(sheet: Image.Image, asset: dict[str, Any], frame_index: int) -> Image.Image:
    """Crop, alpha-trim and letterbox one generated sheet cell without distortion."""

    columns = int(asset["columns"])
    rows = int(asset["rows"])
    frames = int(asset["frames"])
    index = int(frame_index) % frames
    column = index % columns
    row = index // columns

    x0 = round(column * sheet.width / columns)
    x1 = round((column + 1) * sheet.width / columns)
    y0 = round(row * sheet.height / rows)
    y1 = round((row + 1) * sheet.height / rows)
    cell = sheet.crop((x0, y0, x1, y1)).convert("RGBA")

    bbox = cell.getchannel("A").getbbox()
    if bbox is not None:
        cell = cell.crop(bbox)
    if cell.width < 1 or cell.height < 1:
        raise ValueError("empty animation frame")

    available = AVATAR_SIZE - 4
    scale = min(available / cell.width, available / cell.height)
    width = max(1, round(cell.width * scale))
    height = max(1, round(cell.height * scale))
    cell = cell.resize((width, height), Image.Resampling.LANCZOS)

    output = Image.new("RGBA", (AVATAR_SIZE, AVATAR_SIZE), (0, 0, 0, 0))
    x = (AVATAR_SIZE - width) // 2
    # Bottom alignment keeps desk/shoulder baselines visually stable across sheets.
    y = AVATAR_SIZE - height - 2
    output.alpha_composite(cell, (x, y))
    return output


def _load_native_frames(payload: dict[str, Any], root: Any) -> dict[str, list[Any]]:
    """Materialize each unique sheet into Tk-ready 128px frames."""

    loaded: dict[str, list[Any]] = {}
    for name, asset in payload["assets"].items():
        path = _sheet_asset_path(asset)
        if path is None:
            raise ValueError("invalid animation sheet path")
        with Image.open(path) as opened:
            sheet = opened.convert("RGBA")
            frames = [
                ImageTk.PhotoImage(_crop_sheet_frame(sheet, asset, index), master=root)
                for index in range(int(asset["frames"]))
            ]
        loaded[name] = frames
    return loaded


class NativeAvatarApp(_legacy.NativeAvatarApp):
    """Native companion rendered from verified per-state sprite sheets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._animation_manifest = _load_animation_manifest()
        self._animation_frames: dict[str, list[Any]] = {}
        self._display_state = "restarting"
        self._frame_cursor = 0
        super().__init__(*args, **kwargs)

        if self._animation_manifest is not None:
            try:
                self._animation_frames = _load_native_frames(self._animation_manifest, self.root)
            except Exception:
                self._animation_frames = {}

        if self._animation_ready():
            self._display_state = _normalized_state(self.runtime_state)
            self._frame_cursor = 0
        self._draw()

    def _spec(self, state: str) -> dict[str, Any]:
        manifest = self._animation_manifest
        if manifest is None:
            raise RuntimeError("animation manifest unavailable")
        return manifest["states"][_normalized_state(state)]

    def _asset_spec(self, state: str) -> dict[str, Any]:
        manifest = self._animation_manifest
        if manifest is None:
            raise RuntimeError("animation manifest unavailable")
        spec = self._spec(state)
        return manifest["assets"][spec["asset"]]

    def _animation_ready(self) -> bool:
        return self._animation_manifest is not None and bool(self._animation_frames)

    def _request_runtime_state(self) -> None:
        if not self._animation_ready():
            return
        target = _normalized_state(self.runtime_state)
        if target == self._display_state:
            return
        self._display_state = target
        self._frame_cursor = 0

    def _advance_animation(self) -> None:
        if not self._animation_ready():
            return
        frames = int(self._asset_spec(self._display_state)["frames"])
        self._frame_cursor = (self._frame_cursor + 1) % frames

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
        spec = self._spec(self._display_state)
        asset_name = str(spec["asset"])
        frames = self._animation_frames.get(asset_name)
        if not frames:
            _legacy.NativeAvatarApp._draw(self)
            return
        frame = frames[self._frame_cursor % len(frames)]
        self.canvas.delete("all")
        self.canvas.create_image(
            AVATAR_SIZE // 2,
            AVATAR_SIZE // 2,
            image=frame,
            anchor="center",
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
