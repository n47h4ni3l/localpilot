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

# Compact mode is intentionally only the avatar plus a small transparent hit
# target/glow margin. The full chrome appears only after the owner opens chat.
COMPACT_SIZE = (144, 144)
EXPANDED_SIZE = (420, 640)
MIN_SIZE = (120, 120)
EDGE_INSET = 24

_ANCHOR_BOTTOM_RIGHT = FixPoint.SOUTH | FixPoint.EAST


def _desktop_python_executable(
    executable: str | Path | None = None,
    *,
    platform_name: str | None = None,
) -> Path:
    """Prefer ``pythonw.exe`` for the persistent Windows GUI host.

    Normal ``localpilot desktop`` should not leave a console window behind.
    Explicit ``python -m localpilot.webview_app`` remains a foreground/debug
    path and therefore keeps its console and diagnostics.
    """
    platform_name = os.name if platform_name is None else platform_name
    current = Path(executable or sys.executable).resolve()
    if platform_name != "nt" or current.name.lower() == "pythonw.exe":
        return current
    pythonw = current.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else current


def _should_detach_gui(argv0: str, *, platform_name: str | None = None) -> bool:
    """Detach only the normal Windows console-script entry point."""
    platform_name = os.name if platform_name is None else platform_name
    return platform_name == "nt" and Path(argv0).stem.lower() == "localpilot"


def _launch_detached(root: Path, config_path: str | None) -> bool:
    """Relaunch the WebView host under pythonw so normal desktop use is consoleless."""
    executable = _desktop_python_executable()
    if executable.name.lower() != "pythonw.exe":
        return False

    argv = [str(executable), "-m", "localpilot.webview_app", "--root", str(root)]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])

    creationflags = (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    subprocess.Popen(
        argv,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        creationflags=creationflags,
        close_fds=True,
    )
    return True


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
    exe = str(_desktop_python_executable())
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


def _initial_screen() -> Any | None:
    """Return pywebview's first screen, or None when display discovery is unavailable."""
    try:
        return webview.screens[0]
    except Exception:
        return None


def _position_on_screen(screen: Any, width: int, height: int) -> tuple[int, int]:
    """Place the companion at a screen's bottom-right using virtual-desktop coordinates.

    ``Screen.x``/``Screen.y`` may be negative or non-zero on multi-monitor
    Windows layouts. They are part of the absolute logical-pixel coordinate,
    so they must not be discarded or clamped.
    """
    return (
        int(screen.x + screen.width - width - EDGE_INSET),
        int(screen.y + screen.height - height - EDGE_INSET),
    )


def _initial_position(
    width: int,
    height: int,
    screen: Any | None = None,
) -> tuple[int | None, int | None]:
    """Anchor the initial window to the bottom-right of the selected screen.

    Best-effort: if screen enumeration or geometry is unavailable in this
    environment, fall back to letting the OS/window manager choose a position
    rather than failing to launch.
    """
    screen = _initial_screen() if screen is None else screen
    if screen is None:
        return None, None
    try:
        return _position_on_screen(screen, width, height)
    except (AttributeError, TypeError, ValueError):
        return None, None


def main(root: str | Path, config_path: str | None = None) -> None:
    root = Path(root).resolve()

    # The normal `localpilot desktop` console-script invocation relaunches the
    # long-lived GUI under pythonw and returns immediately to the owner's shell.
    # Direct module execution stays foreground so failures remain diagnosable.
    if _should_detach_gui(sys.argv[0]) and _launch_detached(root, config_path):
        return

    config = load_config(config_path)
    client = ensure_broker(root, config, config_path=config_path)

    width, height = COMPACT_SIZE
    screen = _initial_screen()
    x, y = _initial_position(width, height, screen)

    window = webview.create_window(
        "LocalPilot",
        url=str(INDEX_HTML),
        width=width,
        height=height,
        x=x,
        y=y,
        screen=screen,
        min_size=MIN_SIZE,
        frameless=True,
        easy_drag=False,  # dragging is scoped to .pywebview-drag-region elements only
        shadow=False,
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
