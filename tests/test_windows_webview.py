from __future__ import annotations

import itertools

from localpilot.windows_webview import (
    _HOST_PAPER_RGB,
    ComicHostGeometry,
    comic_host_geometry,
    make_host_background_transparent,
)


_handles = itertools.count(100)


class FakeHandle:
    def __init__(self, value: int) -> None:
        self.value = value

    def ToInt64(self) -> int:
        return self.value


class FakeSize:
    def __init__(self, width: int, height: int) -> None:
        self.Width = width
        self.Height = height


class FakeEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def emit(self) -> None:
        for handler in tuple(self.handlers):
            handler(self, None)


class FakeNative:
    def __init__(self, *, invoke_required: bool = False, width: int = 500, height: int = 640) -> None:
        self.InvokeRequired = invoke_required
        self.BackColor = None
        # Deliberately keep a sentinel here. The new implementation must never
        # assign TransparencyKey because that made WebView2 visually correct but
        # unable to receive mouse input on the real Windows desktop.
        self.TransparencyKey = "untouched"
        self.ClientSize = FakeSize(width, height)
        self.Resize = FakeEvent()
        self.Handle = FakeHandle(next(_handles))
        self._scale = 1.0
        self.invoked = False

    def Invoke(self, action) -> None:
        self.invoked = True
        action()


class FakeWindow:
    def __init__(self, native=None) -> None:
        self.native = native


def _color_factory(r: int, g: int, b: int):
    return (r, g, b)


def test_closed_chat_geometry_matches_comic_shell_and_tail():
    geometry = comic_host_geometry(500, 640)

    assert geometry.chat_rect == (8, 8, 466, 630)
    assert geometry.tail_points == ((464, 566), (499, 585), (464, 600))
    assert geometry.notepad_rect is None


def test_systemsense_geometry_is_disjoint_to_left_of_chat():
    geometry = comic_host_geometry(882, 640)

    assert geometry.chat_rect == (390, 8, 848, 630)
    assert geometry.notepad_rect == (8, 16, 368, 622)
    assert geometry.tail_points[1] == (881, 585)
    # The 22px visual gap must remain outside the native window region.
    assert geometry.notepad_rect[2] + 22 == geometry.chat_rect[0]


def test_geometry_scales_to_windows_dpi():
    geometry = comic_host_geometry(750, 960, scale=1.5)

    assert geometry.chat_rect == (12, 12, 699, 945)
    assert geometry.tail_points[1] == (748, 878)
    assert geometry.notepad_rect is None


def test_non_windows_host_is_left_untouched():
    native = FakeNative()
    window = FakeWindow(native)
    applied: list[ComicHostGeometry] = []

    assert make_host_background_transparent(
        window,
        platform_name="posix",
        color_factory=_color_factory,
        region_applier=lambda _native, geometry: applied.append(geometry),
    ) is False
    assert native.BackColor is None
    assert native.TransparencyKey == "untouched"
    assert applied == []


def test_windows_host_uses_native_region_without_transparency_key():
    native = FakeNative()
    window = FakeWindow(native)
    applied: list[ComicHostGeometry] = []

    assert make_host_background_transparent(
        window,
        platform_name="nt",
        color_factory=_color_factory,
        region_applier=lambda _native, geometry: applied.append(geometry),
    ) is True

    assert native.BackColor == _HOST_PAPER_RGB
    assert native.TransparencyKey == "untouched"
    assert applied == [comic_host_geometry(500, 640)]
    assert len(native.Resize.handlers) == 1


def test_windows_host_marshals_native_writes_to_ui_thread_when_required():
    native = FakeNative(invoke_required=True)
    window = FakeWindow(native)
    applied = []

    assert make_host_background_transparent(
        window,
        platform_name="nt",
        color_factory=_color_factory,
        action_factory=lambda callback: callback,
        region_applier=lambda _native, geometry: applied.append(geometry),
    ) is True

    assert native.invoked is True
    assert native.BackColor == _HOST_PAPER_RGB
    assert native.TransparencyKey == "untouched"
    assert len(applied) == 1


def test_resize_hook_rebuilds_region_when_systemsense_opens():
    native = FakeNative(width=500, height=640)
    applied: list[ComicHostGeometry] = []

    assert make_host_background_transparent(
        FakeWindow(native),
        platform_name="nt",
        color_factory=_color_factory,
        region_applier=lambda _native, geometry: applied.append(geometry),
    ) is True
    assert applied[-1].notepad_rect is None

    native.ClientSize = FakeSize(882, 640)
    native.Resize.emit()

    assert applied[-1].notepad_rect == (8, 16, 368, 622)
    assert applied[-1].chat_rect == (390, 8, 848, 630)


def test_windows_host_gracefully_declines_without_native_form():
    assert make_host_background_transparent(
        FakeWindow(None),
        platform_name="nt",
        color_factory=_color_factory,
        region_applier=lambda _native, _geometry: None,
    ) is False
