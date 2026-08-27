"""WebView2-hosted desktop companion.

This is the production replacement for the Tkinter proof of concept in
``desktop.py``. It reuses ``BrokerClient``/``ensure_broker`` from that module
so both entry points share one broker-lifecycle implementation.

Division of responsibility, on purpose:

- All conversation data (sessions, messages, the event stream) flows through
  the real broker HTTP API, fetched directly by the frontend in
  ``webview/app.js``. This module never touches that data.
- This module's only jobs are: get the broker running, get a token, open a
  webview window, and hand the frontend that token. It also exposes a small
  ``js_api`` bridge for the few things that are properties of this OS window
  rather than of LocalPilot's agent state — resizing/moving the window,
  toggling always-on-top, and registering a Windows Startup entry. None of
  those have a broker endpoint, and none of them should: they're not
  LocalPilot state, they're this process's window-manager state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import webview
from webview.window import FixPoint

from localpilot.config import load_config
from localpilot.desktop import BrokerClient, ensure_broker

WEBVIEW_DIR = Path(__file__).resolve().parent / "webview"
INDEX_HTML = WEBVIEW_DIR / "index.html"

COMPACT_SIZE = (168, 220)
EXPANDED_SIZE = (420, 640)
MIN_SIZE = (160, 160)
EDGE_INSET = 24

_ANCHOR_BOTTOM_RIGHT = FixPoint.SOUTH | FixPoint.EAST


def _startup_shortcut_path() -> Path:
    """Location of the Startup-folder entry used for 'start with Windows'."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "LocalPilot.lnk"


def _write_startup_shortcut(target: Path, root: Path, config_path: str | None) -> None:
    """Create a .lnk that relaunches the desktop companion at login.

    Uses a PowerShell one-shot against the WScript.Shell COM object rather
    than adding a binary .lnk-writing dependency, consistent with how the
    rest of this codebase reaches for PowerShell for narrow Windows-native
    operations (see tools/windows.py, tools/windows_actions.py) instead of
    pulling in a new third-party package for one small piece of behaviour.
    """
    exe = str(Path(sys.executable).resolve())
    argv = ["-m", "localpilot.webview_app", "--root", str(root.resolve())]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    arguments = subprocess.list2cmdline(argv)

    # The script is constant: paths and arguments cross the process boundary
    # only as environment values, so apostrophes, quotes and PowerShell
    # metacharacters can never become source code.
    ps = (
        "$s = New-Object -ComObject WScript.Shell; "
        "$sc = $s.CreateShortcut([Environment]::GetEnvironmentVariable('LOCALPILOT_SHORTCUT_PATH')); "
        "$sc.TargetPath = [Environment]::GetEnvironmentVariable('LOCALPILOT_EXECUTABLE'); "
        "$sc.Arguments = [Environment]::GetEnvironmentVariable('LOCALPILOT_ARGUMENTS'); "
        "$sc.WorkingDirectory = [Environment]::GetEnvironmentVariable('LOCALPILOT_WORKING_DIRECTORY'); "
        "$sc.Save()"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "LOCALPILOT_SHORTCUT_PATH": str(target.resolve()),
            "LOCALPILOT_EXECUTABLE": exe,
            "LOCALPILOT_ARGUMENTS": arguments,
            "LOCALPILOT_WORKING_DIRECTORY": str(root.resolve()),
        }
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        check=True,
        capture_output=True,
        env=environment,
        timeout=10,
    )


class WindowBridge:
    """The js_api surface exposed to the frontend — window chrome only.

    Every method here is a native-window or OS operation with no broker
    equivalent. Conversation data never passes through this object.
    """

    def __init__(self, window: webview.Window, root: Path, config_path: str | None) -> None:
        self._window = window
        self._root = root
        self._config_path = config_path

    def expand(self) -> dict[str, Any]:
        w, h = EXPANDED_SIZE
        self._window.resize(w, h, fix_point=_ANCHOR_BOTTOM_RIGHT)
        return {"ok": True}

    def collapse(self) -> dict[str, Any]:
        w, h = COMPACT_SIZE
        self._window.resize(w, h, fix_point=_ANCHOR_BOTTOM_RIGHT)
        return {"ok": True}

    def set_always_on_top(self, value: bool) -> dict[str, Any]:
        self._window.on_top = bool(value)
        return {"ok": True}

    def get_start_with_windows(self) -> dict[str, Any]:
        return {"ok": True, "enabled": _startup_shortcut_path().exists()}

    def set_start_with_windows(self, value: bool) -> dict[str, Any]:
        if os.name != "nt":
            return {"ok": False, "reason": "not-windows"}
        target = _startup_shortcut_path()
        try:
            if value:
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_startup_shortcut(target, self._root, self._config_path)
            else:
                target.unlink(missing_ok=True)
            return {"ok": True, "enabled": value}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "reason": str(exc)}

    def open_config_file(self) -> dict[str, Any]:
        if not self._config_path:
            return {"ok": False, "reason": "no-config-path"}
        path = Path(self._config_path)
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(
                    ["xdg-open", str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return {"ok": True}
        except OSError as exc:
            return {"ok": False, "reason": str(exc)}


def _bridge_payload(client: BrokerClient, config_path: str | None) -> dict[str, Any]:
    return {"baseUrl": client.base_url, "token": client.token, "hasConfigPath": bool(config_path)}


def _initial_position(width: int, height: int) -> tuple[int | None, int | None]:
    """Anchor the initial window to the bottom-right of the primary screen.

    Best-effort: if screen enumeration isn't available in this environment,
    fall back to letting the OS/window manager choose a position rather than
    failing to launch.
    """
    try:
        screen = webview.screens[0]
        x = max(0, screen.width - width - EDGE_INSET)
        y = max(0, screen.height - height - EDGE_INSET)
        return x, y
    except Exception:
        return None, None


def main(root: str | Path, config_path: str | None = None) -> None:
    root = Path(root).resolve()
    config = load_config(config_path)
    client = ensure_broker(root, config, config_path=config_path)

    width, height = COMPACT_SIZE
    x, y = _initial_position(width, height)

    window = webview.create_window(
        "LocalPilot",
        url=str(INDEX_HTML),
        width=width,
        height=height,
        x=x,
        y=y,
        min_size=MIN_SIZE,
        frameless=True,
        easy_drag=False,  # dragging is scoped to .pywebview-drag-region elements only
        on_top=True,
        background_color="#0B121D",
        transparent=True,
    )

    bridge = WindowBridge(window, root, config_path)
    window.expose(
        bridge.expand,
        bridge.collapse,
        bridge.set_always_on_top,
        bridge.get_start_with_windows,
        bridge.set_start_with_windows,
        bridge.open_config_file,
    )

    def on_loaded() -> None:
        payload = _bridge_payload(client, config_path)
        window.evaluate_js(f"window.__initLocalPilot({json.dumps(payload)})")

    window.events.loaded += on_loaded

    webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-desktop")
    parser.add_argument("--root", required=True, help="LocalPilot project root")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    return parser


def cli_main() -> None:
    args = build_parser().parse_args()
    main(args.root, args.config)


if __name__ == "__main__":
    cli_main()
