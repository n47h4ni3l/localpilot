"""Windows-only native host shaping for the LocalPilot comic WebView.

The WebView2 document itself can render transparently, but the WinForms parent
is still a normal rectangular window.  A colour-key TransparencyKey removes the
rectangle, but WebView2 hosted inside that colour-keyed Form stops receiving
mouse input reliably and can leave composition ghosting.  LocalPilot therefore
keeps a normal interactive WinForms host and clips the native window to the
actual comic surfaces instead.

The public helper keeps its old name so the existing WebView startup wiring does
not need a second platform-specific path.
"""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

from localpilot.comic_geometry import pixel, pixels

_HOST_PAPER_RGB = (248, 242, 232)

# Keep event handlers alive for the lifetime of the process.  pythonnet event
# subscriptions can otherwise lose a Python callback after garbage collection.
_RESIZE_HANDLERS: dict[int, Any] = {}


@dataclass(frozen=True)
class ComicHostGeometry:
    """Physical-pixel native regions matching the fixed CSS comic geometry."""

    chat_rect: tuple[int, int, int, int]
    chat_outline: tuple[tuple[int, int], ...]
    tail_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    notepad_rect: tuple[int, int, int, int] | None
    notepad_outline: tuple[tuple[int, int], ...]
    binding_holes: tuple[tuple[int, int, int, int], ...]


def _scale_value(value: float, scale: float) -> int:
    return int(round(value * scale))


def rounded_outline(rect, radii_x, radii_y, scale):
    """Trace CSS's four elliptical corners, then rasterize at the current DPI.

    Four-degree segments stay below a physical pixel even at 300% scaling.
    Using a single circular GDI radius used to expose pale crescent fringes.
    """
    left, top, right, bottom = rect
    corners = (
        (right - radii_x[1], top + radii_y[1], radii_x[1], radii_y[1], -90),
        (right - radii_x[2], bottom - radii_y[2], radii_x[2], radii_y[2], 0),
        (left + radii_x[3], bottom - radii_y[3], radii_x[3], radii_y[3], 90),
        (left + radii_x[0], top + radii_y[0], radii_x[0], radii_y[0], 180),
    )
    return tuple(
        (_scale_value(cx + rx * math.cos(math.radians(start + step * 90 / 24)), scale),
         _scale_value(cy + ry * math.sin(math.radians(start + step * 90 / 24)), scale))
        for cx, cy, rx, ry, start in corners for step in range(25)
    )


def comic_host_geometry(
    client_width: int, client_height: int, scale: float = 1.0,
    *, systemsense_open: bool = False, tail_left: bool = False,
) -> ComicHostGeometry:
    """Match the CSS border boxes, including their padding-box containing block.

    Visibility is explicit: a wide chat window must not expose a phantom
    notepad. All shape dimensions come from comic-geometry.css.
    """
    scale = float(scale or 1.0)
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0

    logical_width = max(1.0, client_width / scale)
    logical_height = max(1.0, client_height / scale)

    reserve = pixel("systemsense-width") + pixel("systemsense-gap") if systemsense_open else 0
    chat_right = logical_width - pixel("chat-right-inset")
    chat_left = max(pixel("surface-left-inset") + reserve, chat_right - pixel("chat-panel-width"))
    if tail_left:
        chat_left = pixel("chat-right-inset")
        chat_right = min(chat_left + pixel("chat-panel-width"), logical_width - pixel("surface-left-inset") - reserve)
    chat_top = pixel("surface-top-inset")
    chat_bottom = max(chat_top + 1, logical_height - pixel("surface-bottom-inset"))
    border = pixel("chat-border")

    # Absolutely positioned pseudo-elements use the panel's padding edge,
    # which is one border width *inside* its outer native rectangle.
    tail_tip_x = chat_right - border + pixel("tail-right")
    tail_base_x = tail_tip_x - pixel("tail-width")
    if tail_left:
        tail_tip_x = chat_left + border - pixel("tail-right")
        tail_base_x = tail_tip_x + pixel("tail-width")
    tail_base_bottom_y = chat_bottom - border - pixel("tail-offset")
    tail_tip_y = tail_base_bottom_y - pixel("tail-bottom")
    tail_base_top_y = tail_tip_y - pixel("tail-top")
    tail_points = (
        (tail_base_x, tail_base_top_y),
        (tail_tip_x, tail_tip_y),
        (tail_base_x, tail_base_bottom_y),
    )

    notepad_rect: tuple[int, int, int, int] | None = None
    notepad_outline = ()
    holes = []
    if systemsense_open:
        notepad_right = chat_left - pixel("systemsense-gap")
        notepad_left = notepad_right - pixel("systemsense-width")
        if tail_left:
            notepad_left = chat_right + pixel("systemsense-gap")
            notepad_right = notepad_left + pixel("systemsense-width")
        notepad_top = chat_top + border + pixel("notepad-inset")
        notepad_bottom = chat_bottom - border - pixel("notepad-inset")
        notepad_outline = rounded_outline(
            (notepad_left, notepad_top, notepad_right, notepad_bottom),
            pixels("notepad-radii"), pixels("notepad-radii"), scale,
        )
        note_border = pixel("notepad-border")
        cx = notepad_left + note_border + pixel("binding-left") + pixel("binding-width") / 2
        cy = notepad_top + note_border + pixel("binding-top") + pixel("binding-pitch") / 2
        # Match CSS's strip rounded down to complete background tiles.
        strip_height = notepad_bottom - notepad_top - 2 * note_border - 2 * pixel("binding-top")
        row_count = max(0, int(strip_height // pixel("binding-pitch")))
        radius = pixel("binding-hole-radius")
        for _ in range(row_count):
            holes.append(tuple(_scale_value(v, scale) for v in (cx-radius, cy-radius, cx+radius, cy+radius)))
            cy += pixel("binding-pitch")
        notepad_rect = (
            _scale_value(notepad_left, scale),
            _scale_value(notepad_top, scale),
            _scale_value(notepad_right, scale),
            _scale_value(notepad_bottom, scale),
        )

    scaled_tail = tuple(
        (_scale_value(x, scale), _scale_value(y, scale)) for x, y in tail_points
    )
    return ComicHostGeometry(
        chat_rect=(
            _scale_value(chat_left, scale),
            _scale_value(chat_top, scale),
            _scale_value(chat_right, scale),
            _scale_value(chat_bottom, scale),
        ),
        chat_outline=rounded_outline(
            (chat_left, chat_top, chat_right, chat_bottom),
            pixels("chat-radii-x"), pixels("chat-radii-y"), scale,
        ),
        tail_points=(scaled_tail[0], scaled_tail[1], scaled_tail[2]),
        notepad_rect=notepad_rect,
        notepad_outline=notepad_outline,
        binding_holes=tuple(holes),
    )


def _native_handle(native: Any) -> int:
    handle = native.Handle
    to_int64 = getattr(handle, "ToInt64", None)
    if callable(to_int64):
        return int(to_int64())
    to_int32 = getattr(handle, "ToInt32", None)
    if callable(to_int32):
        return int(to_int32())
    return int(handle)


def _apply_win32_region(native: Any, geometry: ComicHostGeometry) -> None:
    """Apply a disjoint rounded window region using GDI; caller owns UI thread."""
    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    RGN_OR = 2
    RGN_DIFF = 4
    ALTERNATE = 1

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    # GDI object and HWND handles are pointer-sized on 64-bit Windows. Explicit
    # ctypes signatures prevent the default c_int conversion from truncating
    # valid handles on the machines LocalPilot actually runs on.
    gdi32.CreateEllipticRgn.argtypes = [ctypes.c_int] * 4
    gdi32.CreateEllipticRgn.restype = wintypes.HANDLE
    gdi32.CreatePolygonRgn.argtypes = [ctypes.POINTER(POINT), ctypes.c_int, ctypes.c_int]
    gdi32.CreatePolygonRgn.restype = wintypes.HANDLE
    gdi32.CombineRgn.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.c_int]
    gdi32.CombineRgn.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.DeleteObject.restype = wintypes.BOOL
    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HANDLE, wintypes.BOOL]
    user32.SetWindowRgn.restype = ctypes.c_int

    def polygon(outline):
        points = (POINT * len(outline))(*(POINT(x, y) for x, y in outline))
        return gdi32.CreatePolygonRgn(points, len(points), ALTERNATE)

    chat = polygon(geometry.chat_outline)
    if not chat:
        raise OSError("CreatePolygonRgn failed for chat surface")

    owned_regions: list[Any] = []
    try:
        tail = polygon(geometry.tail_points)
        if not tail:
            raise OSError("CreatePolygonRgn failed for speech tail")
        owned_regions.append(tail)
        if not gdi32.CombineRgn(chat, chat, tail, RGN_OR):
            raise OSError("CombineRgn failed for speech tail")

        if geometry.notepad_rect is not None:
            notepad = polygon(geometry.notepad_outline)
            if not notepad:
                raise OSError("CreatePolygonRgn failed for SystemSense surface")
            owned_regions.append(notepad)
            if not gdi32.CombineRgn(chat, chat, notepad, RGN_OR):
                raise OSError("CombineRgn failed for SystemSense surface")
            for bounds in geometry.binding_holes:
                hole = gdi32.CreateEllipticRgn(*bounds)
                if not hole:
                    raise OSError("CreateEllipticRgn failed for binding hole")
                owned_regions.append(hole)
                if not gdi32.CombineRgn(chat, chat, hole, RGN_DIFF):
                    raise OSError("CombineRgn failed for binding hole")

        # On success Windows owns `chat`; it must not be DeleteObject'd here.
        if not user32.SetWindowRgn(_native_handle(native), chat, True):
            raise OSError("SetWindowRgn failed")
        chat = None
    finally:
        for region in owned_regions:
            if region:
                gdi32.DeleteObject(region)
        if chat:
            gdi32.DeleteObject(chat)


def make_host_background_transparent(
    window: Any,
    *,
    platform_name: str | None = None,
    color_factory: Callable[[int, int, int], Any] | None = None,
    action_factory: Callable[[Callable[[], None]], Any] | None = None,
    region_applier: Callable[[Any, ComicHostGeometry], None] | None = None,
) -> bool:
    """Shape the Windows host to the comic surfaces without disabling input.

    The historical function name is retained for compatibility.  Unlike the
    previous implementation, this function deliberately does *not* set
    ``TransparencyKey``.  Colour-keyed WinForms hosts interfere with WebView2
    hit testing.  Instead the parent Form stays opaque/clickable and its native
    window region is clipped to the speech bubble, tail and optional notepad.
    """

    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt":
        return False

    native = getattr(window, "native", None)
    if native is None:
        return False

    region_applier = region_applier or _apply_win32_region

    try:
        if color_factory is None:
            from System.Drawing import Color  # type: ignore[import-not-found]

            paper = Color.FromArgb(255, *_HOST_PAPER_RGB)
        else:
            paper = color_factory(*_HOST_PAPER_RGB)

        def apply() -> None:
            # A normal BackColor is intentional.  The shaped native Region, not
            # a TransparencyKey/layered window, removes the rectangular chrome.
            native.BackColor = paper
            scale = float(getattr(native, "_scale", 1.0) or 1.0)
            if hasattr(native, "MinimumSize"):
                from System.Drawing import Size  # type: ignore[import-not-found]

                min_width, min_height = window.min_size
                native.MinimumSize = Size(round(min_width * scale), round(min_height * scale))
            size = native.ClientSize
            geometry = comic_host_geometry(
                int(size.Width), int(size.Height), scale,
                systemsense_open=bool(getattr(window, "_comic_systemsense_open", False)),
                tail_left=bool(getattr(window, "_comic_tail_left", False)),
            )
            region_applier(native, geometry)

        def marshal(callback: Callable[[], None]) -> None:
            if bool(getattr(native, "InvokeRequired", False)):
                if action_factory is None:
                    from System import Action  # type: ignore[import-not-found]

                    native.Invoke(Action(callback))
                else:
                    native.Invoke(action_factory(callback))
            else:
                callback()

        marshal(apply)

        key = _native_handle(native)
        if key not in _RESIZE_HANDLERS:
            def on_resize(*_args: Any) -> None:
                try:
                    apply()
                except Exception:
                    # A transient resize should never take the chat down.  The
                    # next resize or loaded retry can restore the silhouette.
                    return

            native.Resize += on_resize
            _RESIZE_HANDLERS[key] = on_resize
    except Exception:
        return False

    return True
