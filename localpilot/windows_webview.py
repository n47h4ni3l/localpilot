"""Windows-only native host polish for the transparent LocalPilot WebView.

pywebview makes the WebView2 surface transparent, but on WinForms those pixels
still reveal the opaque parent Form.  The comic shell therefore needs the Form
background itself to be colour-key transparent as well.  This module keeps that
platform-specific detail out of the broker-backed chat implementation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Deliberately obscure, near-black key that is not used by the comic artwork.
# A dark key also avoids a bright fringe while Windows is resolving antialiased
# pixels around the ink outline.
_HOST_KEY_RGB = (1, 2, 3)


def make_host_background_transparent(
    window: Any,
    *,
    platform_name: str | None = None,
    color_factory: Callable[[int, int, int], Any] | None = None,
    action_factory: Callable[[Callable[[], None]], Any] | None = None,
) -> bool:
    """Make pywebview's native WinForms host transparent behind WebView2.

    ``transparent=True`` clears the Chromium surface but does not clear the
    parent Form.  Setting the Form's ``BackColor`` and ``TransparencyKey`` to
    the same colour removes the rectangular host that otherwise shows through
    the transparent HTML margins, speech-tail space and SystemSense gap.

    pywebview dispatches ``shown``/``loaded`` callbacks on Python worker
    threads, so native property writes are marshalled back onto the WinForms UI
    thread when required.  The factories are injectable only to keep this
    behaviour unit-testable without constructing a real Windows Form.
    """

    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt":
        return False

    native = getattr(window, "native", None)
    if native is None:
        return False

    try:
        if color_factory is None:
            from System.Drawing import Color  # type: ignore[import-not-found]

            key = Color.FromArgb(255, *_HOST_KEY_RGB)
        else:
            key = color_factory(*_HOST_KEY_RGB)

        def apply() -> None:
            native.BackColor = key
            native.TransparencyKey = key

        if bool(getattr(native, "InvokeRequired", False)):
            if action_factory is None:
                from System import Action  # type: ignore[import-not-found]

                native.Invoke(Action(apply))
            else:
                native.Invoke(action_factory(apply))
        else:
            apply()
    except Exception:
        # The comic UI must still remain usable if a future pywebview backend
        # changes its native object shape.  CSS transparency remains the safe
        # fallback; this helper only removes the WinForms parent rectangle.
        return False

    return True
