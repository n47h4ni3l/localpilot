"""Shared comic dimensions, read directly from the production CSS contract.

The browser and native window region must use the same literal pixel values.
Do not duplicate these values in Python or override them in another stylesheet.
CSS border radii are radii (CreateRoundRectRgn, in contrast, takes diameters).
"""

from pathlib import Path
import re


GEOMETRY_CSS = Path(__file__).with_name("webview") / "comic-geometry.css"
_SOURCE = GEOMETRY_CSS.read_text(encoding="utf-8")


def pixels(name: str) -> tuple[float, ...]:
    matches = re.findall(r"--" + re.escape(name) + r":\s*([^;]+);", _SOURCE)
    if len(matches) != 1 or not re.fullmatch(r"\d+(?:\.\d+)?px(?:\s+\d+(?:\.\d+)?px)*", matches[0]):
        raise ValueError(f"Expected one literal pixel definition for --{name}")
    return tuple(float(value.removesuffix("px")) for value in matches[0].split())


def pixel(name: str) -> float:
    value, = pixels(name)
    return value
