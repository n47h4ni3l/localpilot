from __future__ import annotations

import itertools
import pytest
from localpilot.comic_geometry import pixel

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
    # CSS positions the triangle against the padding edge, inside the 3px border.
    assert geometry.tail_points == ((461, 551), (494, 569), (461, 582))
    assert geometry.notepad_rect is None


def test_systemsense_geometry_keeps_binding_strip_and_gap():
    geometry = comic_host_geometry(882, 640, systemsense_open=True)

    assert geometry.chat_rect == (390, 8, 848, 630)
    # The paper and its punched binding fit inside the same native outline.
    assert geometry.notepad_rect == (8, 19, 368, 619)
    assert geometry.tail_points[1] == (876, 569)
    # The 22px visual gap must remain outside the native window region.
    assert geometry.notepad_rect[2] + 22 == geometry.chat_rect[0]


def test_geometry_scales_to_windows_dpi():
    geometry = comic_host_geometry(750, 960, scale=1.5)

    assert geometry.chat_rect == (12, 12, 699, 945)
    assert geometry.tail_points[1] == (741, 854)
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
    window = FakeWindow(native)
    applied: list[ComicHostGeometry] = []

    assert make_host_background_transparent(
        window,
        platform_name="nt",
        color_factory=_color_factory,
        region_applier=lambda _native, geometry: applied.append(geometry),
    ) is True
    assert applied[-1].notepad_rect is None

    window._comic_systemsense_open = True
    native.ClientSize = FakeSize(882, 640)
    native.Resize.emit()

    assert applied[-1].notepad_rect == (8, 19, 368, 619)
    assert applied[-1].chat_rect == (390, 8, 848, 630)


def test_windows_host_gracefully_declines_without_native_form():
    assert make_host_background_transparent(
        FakeWindow(None),
        platform_name="nt",
        color_factory=_color_factory,
        region_applier=lambda _native, _geometry: None,
    ) is False


@pytest.mark.parametrize("scale", [1, 1.25, 1.5, 1.75, 2, 3])
@pytest.mark.parametrize("opened,width", [(False, 420), (False, 500), (True, 802), (True, 882)])
def test_all_artwork_fits_at_supported_sizes_and_dpi(scale, opened, width):
    geometry = comic_host_geometry(round(width * scale), round(640 * scale), scale,
                                   systemsense_open=opened)
    for x, y in geometry.chat_outline + geometry.tail_points + geometry.notepad_outline:
        assert 0 <= x <= round(width * scale)
        assert 0 <= y <= round(640 * scale)
    assert geometry.chat_rect[0] < geometry.chat_rect[2]
    if opened:
        left, top, right, bottom = geometry.notepad_rect
        for hl, ht, hr, hb in geometry.binding_holes:
            assert left < hl < hr < right
            assert top < ht < hb < bottom
        assert geometry.chat_rect[0] - right == round(pixel("systemsense-gap") * scale)


def test_wide_closed_host_does_not_expose_or_intercept_empty_notepad_space():
    geometry = comic_host_geometry(1000, 640)
    assert geometry.notepad_rect is None
    assert geometry.notepad_outline == ()
    assert geometry.binding_holes == ()


def test_asymmetric_elliptical_corners_keep_the_outer_corner_transparent():
    geometry = comic_host_geometry(500, 640)
    # Top-right horizontal radius is 28; bottom-right is 38. Using one GDI
    # diameter of 34 previously left crescents of opaque host in both corners.
    assert geometry.chat_outline[0] == (438, 8)
    assert geometry.chat_outline[24] == (466, 44)
    assert geometry.chat_outline[49] == (428, 630)
    assert (466, 8) not in geometry.chat_outline


@pytest.mark.parametrize("height", [520, 600, 640, 644, 645, 648, 720])
def test_native_binding_hole_count_matches_complete_css_background_tiles(height):
    geometry = comic_host_geometry(882, height, systemsense_open=True)
    _, top, _, bottom = geometry.notepad_rect
    strip_height = bottom - top - 2 * pixel("notepad-border") - 2 * pixel("binding-top")
    assert len(geometry.binding_holes) == int(strip_height // pixel("binding-pitch"))


@pytest.mark.parametrize("scale", [1, 1.25, 1.5, 1.75, 2, 3])
@pytest.mark.parametrize("opened,width", [(False, 420), (False, 500), (True, 802), (True, 882)])
def test_left_facing_tail_and_notepad_fit_without_reversing_the_paper(scale, opened, width):
    geometry = comic_host_geometry(round(width * scale), round(640 * scale), scale,
                                   systemsense_open=opened, tail_left=True)
    assert geometry.tail_points[1][0] < geometry.chat_rect[0]
    assert geometry.tail_points[0][0] > geometry.chat_rect[0]
    for x, y in geometry.chat_outline + geometry.tail_points + geometry.notepad_outline:
        assert 0 <= x <= round(width * scale)
        assert 0 <= y <= round(640 * scale)
    if opened:
        left, top, right, bottom = geometry.notepad_rect
        # Each CSS boundary rasterizes independently at fractional scaling.
        assert abs(left - geometry.chat_rect[2] - pixel("systemsense-gap") * scale) <= 1
        assert all(left < hl < hr < left + 30 * scale and top < ht < hb < bottom
                   for hl, ht, hr, hb in geometry.binding_holes)
