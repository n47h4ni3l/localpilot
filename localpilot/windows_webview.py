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
import os
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

_HOST_PAPER_RGB = (248, 242, 232)
_CHAT_PANEL_WIDTH = 458.0
_CHAT_RIGHT_INSET = 34.0
_SURFACE_TOP_INSET = 8.0
_SURFACE_BOTTOM_INSET = 10.0
_SYSTEMSENSE_WIDTH = 360.0
_SYSTEMSENSE_GAP = 22.0
_SYSTEMSENSE_THRESHOLD = 700.0

# Keep event handlers alive for the lifetime of the process.  pythonnet event
# subscriptions can otherwise lose a Python callback after garbage collection.
_RESIZE_HANDLERS: dict[int, Any] = {}


@dataclass(frozen=True)
class ComicHostGeometry:
    """Physical-pixel native regions matching the fixed CSS comic geometry."""

    chat_rect: tuple[int, int, int, int]
    chat_radius: int
    tail_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    notepad_rect: tuple[int, int, int, int] | None
    notepad_radius: int


def _scale_value(value: float, scale: float) -> int:
    return int(round(value * scale))


def comic_host_geometry(client_width: int, client_height: int, scale: float = 1.0) -> ComicHostGeometry:
    """Return the native window silhouette for the current logical host size."""
    scale = float(scale or 1.0)
    if scale <= 0:
        scale = 1.0

    logical_width = max(1.0, client_width / scale)
    logical_height = max(1.0, client_height / scale)

    chat_left = max(0.0, logical_width - _CHAT_RIGHT_INSET - _CHAT_PANEL_WIDTH)
    chat_top = _SURFACE_TOP_INSET
    chat_right = min(logical_width, chat_left + _CHAT_PANEL_WIDTH)
    chat_bottom = max(chat_top + 1.0, logical_height - _SURFACE_BOTTOM_INSET)

    # Matches comic-shell.css' outlined tail closely enough that the native
    # region clips away the rectangular host without clipping the visible ink.
    tail_center_y = max(chat_top + 24.0, chat_bottom - 45.0)
    tail_tip_x = min(logical_width, chat_right + 33.0)
    tail_points = (
        (chat_right - 2.0, tail_center_y - 19.0),
        (tail_tip_x, tail_center_y),
        (chat_right - 2.0, tail_center_y + 15.0),
    )

    notepad_rect: tuple[int, int, int, int] | None = None
    if logical_width >= _SYSTEMSENSE_THRESHOLD:
        notepad_right = chat_left - _SYSTEMSENSE_GAP
        notepad_left = max(0.0, notepad_right - _SYSTEMSENSE_WIDTH)
        notepad_top = chat_top + 8.0
        notepad_bottom = max(notepad_top + 1.0, chat_bottom - 8.0)
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
        chat_radius=max(2, _scale_value(34.0, scale)),
        tail_points=(scaled_tail[0], scaled_tail[1], scaled_tail[2]),
        notepad_rect=notepad_rect,
        notepad_radius=max(2, _scale_value(10.0, scale)),
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
    ALTERNATE = 1

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    # GDI object and HWND handles are pointer-sized on 64-bit Windows. Explicit
    # ctypes signatures prevent the default c_int conversion from truncating
    # valid handles on the machines LocalPilot actually runs on.
    gdi32.CreateRoundRectRgn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gdi32.CreateRoundRectRgn.restype = wintypes.HANDLE
    gdi32.CreatePolygonRgn.argtypes = [ctypes.POINTER(POINT), ctypes.c_int, ctypes.c_int]
    gdi32.CreatePolygonRgn.restype = wintypes.HANDLE
    gdi32.CombineRgn.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.c_int]
    gdi32.CombineRgn.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.DeleteObject.restype = wintypes.BOOL
    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HANDLE, wintypes.BOOL]
    user32.SetWindowRgn.restype = ctypes.c_int

    left, top, right, bottom = geometry.chat_rect
    chat = gdi32.CreateRoundRectRgn(
        left,
        top,
        right + 1,
        bottom + 1,
        geometry.chat_radius,
        geometry.chat_radius,
    )
    if not chat:
        raise OSError("CreateRoundRectRgn failed for chat surface")

    owned_regions: list[Any] = []
    try:
        points = (POINT * 3)(*(POINT(x, y) for x, y in geometry.tail_points))
        tail = gdi32.CreatePolygonRgn(points, 3, ALTERNATE)
        if not tail:
            raise OSError("CreatePolygonRgn failed for speech tail")
        owned_regions.append(tail)
        gdi32.CombineRgn(chat, chat, tail, RGN_OR)

        if geometry.notepad_rect is not None:
            n_left, n_top, n_right, n_bottom = geometry.notepad_rect
            notepad = gdi32.CreateRoundRectRgn(
                n_left,
                n_top,
                n_right + 1,
                n_bottom + 1,
                geometry.notepad_radius,
                geometry.notepad_radius,
            )
            if not notepad:
                raise OSError("CreateRoundRectRgn failed for SystemSense surface")
            owned_regions.append(notepad)
            gdi32.CombineRgn(chat, chat, notepad, RGN_OR)

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
            size = native.ClientSize
            geometry = comic_host_geometry(int(size.Width), int(size.Height), scale)
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
