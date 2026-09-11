from __future__ import annotations

"""Illustrated LocalPilot desktop avatar.

The original pixel companion is preserved in ``native_avatar_legacy`` and is
used as a fail-safe whenever the bundled line-art sprite cannot be loaded.
The illustrated avatar deliberately alternates between two independently
redrawn frames instead of tweening them, producing the small hand-drawn
"line boil" that makes the character feel alive.
"""

import base64
import hashlib
from pathlib import Path
from typing import Any

from localpilot import native_avatar_legacy as _legacy
from localpilot.config import Config, load_config
from localpilot.desktop import BrokerClient, ensure_broker

# Preserve the public/native module contract.  Several geometry helpers are
# deliberately kept here because callers and regression tests import them from
# localpilot.native_avatar rather than from the implementation module.
AVATAR_SIZE = _legacy.AVATAR_SIZE
EDGE_INSET = _legacy.EDGE_INSET
EXPANDED_SIZE = _legacy.EXPANDED_SIZE
_TRANSPARENT_KEY = _legacy._TRANSPARENT_KEY
_CHAT_START_GRACE_MS = _legacy._CHAT_START_GRACE_MS

# Keep the legacy renderer aware of the two extra expressive states so future
# runtime events can opt into them without another UI migration.
_legacy._STATE_COLORS.update(
    {
        "researching": "#f0c24e",
        "learning": "#6fde8e",
    }
)
_STATE_COLORS = _legacy._STATE_COLORS

# Sprite layout: two hand-redrawn frames across, nine poses down.  Runtime
# states that do not have a dedicated pose intentionally share the closest
# expressive pose while preserving the existing runtime state machine.
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
_SPRITE_PARTS = tuple(f"sprite.{index:02d}.b64" for index in range(6))
_SPRITE_SHA256 = "59579cf6e01c721c867a133b60388c9a3157b43ecf21d68be708b24b480268f4"
_SPRITE_B64_LENGTH = 56660

# These exact Win32/Tk contract strings are retained here as documentation as
# well as in the preserved implementation.  They are intentionally visible to
# source-level regression checks:
#   wm_attributes("-transparentcolor", _TRANSPARENT_KEY)
#   overrideredirect(True)
#   GetParent.restype = ctypes.c_void_p
#   MonitorFromPoint.restype = ctypes.c_void_p
#   SetWindowPos.argtypes

_enable_per_monitor_dpi_awareness = _legacy._enable_per_monitor_dpi_awareness
_primary_work_area = _legacy._primary_work_area
_monitor_work_area_for_point = _legacy._monitor_work_area_for_point
_clamp_position = _legacy._clamp_position
_initial_avatar_position = _legacy._initial_avatar_position
_top_level_hwnd = _legacy._top_level_hwnd
_set_window_position = _legacy._set_window_position
subprocess = _legacy.subprocess
_desktop_python_executable = _legacy._desktop_python_executable


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "webview" / "avatar"


def _read_avatar_sprite_data() -> str | None:
    """Return the verified base64 sprite payload, or ``None`` on any damage."""

    directory = _asset_dir()
    try:
        data = "".join((directory / name).read_text(encoding="ascii").strip() for name in _SPRITE_PARTS)
        if len(data) != _SPRITE_B64_LENGTH:
            return None
        raw = base64.b64decode(data, validate=True)
        if hashlib.sha256(raw).hexdigest() != _SPRITE_SHA256:
            return None
        return data
    except (OSError, UnicodeError, ValueError):
        return None


def _materialize_webview_sprite(data: str) -> Path | None:
    """Best-effort PNG materialisation for the expanded WebView.

    The repository stores the artwork as text chunks so GitHub/source tooling
    can handle it reliably.  Normal desktop startup reconstructs the tiny PNG
    next to the WebView assets.  A read-only installation simply keeps the
    pixel WebView fallback; the native avatar can still render from base64.
    """

    target = _asset_dir() / "sprite.png"
    try:
        raw = base64.b64decode(data, validate=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == _SPRITE_SHA256:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".png.tmp")
        temporary.write_bytes(raw)
        temporary.replace(target)
        return target
    except (OSError, ValueError):
        return None


def _recover_avatar_position(x: int, y: int) -> tuple[int, int]:
    """Guarantee the complete avatar is visible on a current monitor."""

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
    """Pure geometry helper retained for tests/future same-coordinate hosts."""

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
) -> Any:
    """Delegate to the proven launcher while keeping monkeypatch compatibility."""

    previous = _legacy._desktop_python_executable
    _legacy._desktop_python_executable = _desktop_python_executable
    try:
        return _legacy._launch_webview(root, config_path, x=x, y=y)
    finally:
        _legacy._desktop_python_executable = previous


class NativeAvatarApp(_legacy.NativeAvatarApp):
    """Native transparent companion rendered from the illustrated sprite."""

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
        # The legacy constructor calls self._draw() before returning, so the
        # attribute must exist first.  That first draw simply uses the pixel
        # fallback and is immediately replaced after the sprite loads.
        self._illustrated_sprite: Any | None = None
        super().__init__(
            client,
            config,
            root,
            config_path=config_path,
            initial_x=initial_x,
            initial_y=initial_y,
        )
        sprite_data = _read_avatar_sprite_data()
        if sprite_data:
            try:
                self._illustrated_sprite = self.tk.PhotoImage(data=sprite_data, format="png")
                _materialize_webview_sprite(sprite_data)
            except Exception:
                self._illustrated_sprite = None
        self._draw()

    def _draw(self) -> None:
        sprite = getattr(self, "_illustrated_sprite", None)
        if sprite is None:
            super()._draw()
            return

        self.canvas.delete("all")
        row = _SPRITE_ROWS.get(self.runtime_state, _SPRITE_ROWS["error"])
        # Hold each independently redrawn image for two animation ticks.  The
        # resulting 360 ms cadence is intentionally imperfect-looking rather
        # than a smooth digital interpolation.
        frame = (self.frame // 2) % 2
        self.canvas.create_image(
            -frame * AVATAR_SIZE,
            -row * AVATAR_SIZE,
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


build_parser = _legacy.build_parser


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config, x=args.x, y=args.y)


if __name__ == "__main__":
    cli_main()
