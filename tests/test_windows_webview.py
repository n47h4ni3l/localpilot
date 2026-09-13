from __future__ import annotations

from localpilot.windows_webview import _HOST_KEY_RGB, make_host_background_transparent


class FakeNative:
    def __init__(self, *, invoke_required: bool = False) -> None:
        self.InvokeRequired = invoke_required
        self.BackColor = None
        self.TransparencyKey = None
        self.invoked = False

    def Invoke(self, action) -> None:
        self.invoked = True
        action()


class FakeWindow:
    def __init__(self, native=None) -> None:
        self.native = native


def _color_factory(r: int, g: int, b: int):
    return (r, g, b)


def test_non_windows_host_is_left_untouched():
    native = FakeNative()
    window = FakeWindow(native)

    assert make_host_background_transparent(
        window,
        platform_name="posix",
        color_factory=_color_factory,
    ) is False
    assert native.BackColor is None
    assert native.TransparencyKey is None


def test_windows_host_uses_same_colour_for_backdrop_and_transparency_key():
    native = FakeNative()
    window = FakeWindow(native)

    assert make_host_background_transparent(
        window,
        platform_name="nt",
        color_factory=_color_factory,
    ) is True
    assert native.BackColor == _HOST_KEY_RGB
    assert native.TransparencyKey == _HOST_KEY_RGB


def test_windows_host_marshals_native_writes_to_ui_thread_when_required():
    native = FakeNative(invoke_required=True)
    window = FakeWindow(native)

    assert make_host_background_transparent(
        window,
        platform_name="nt",
        color_factory=_color_factory,
        action_factory=lambda callback: callback,
    ) is True
    assert native.invoked is True
    assert native.BackColor == _HOST_KEY_RGB
    assert native.TransparencyKey == _HOST_KEY_RGB


def test_windows_host_gracefully_declines_without_native_form():
    assert make_host_background_transparent(
        FakeWindow(None),
        platform_name="nt",
        color_factory=_color_factory,
    ) is False
