"""WebView2-hosted conversation surface for the LocalPilot desktop companion.

The illustrated Astra avatar is a separate native transparent window. The
normal desktop path keeps that avatar alive while this module hosts only the
comic speech-bubble chat. SystemSense remains the same authenticated read-only
surface and is presented as a separate notepad to the left of the conversation.
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
from localpilot.comic_geometry import pixel
from localpilot.desktop import BrokerClient, ensure_broker
from localpilot.desktop_state import DesktopUIState
from localpilot.desktop_update_status import (
    check_for_updates as check_desktop_updates,
    current_update_status,
    set_automatic_updates as set_desktop_automatic_updates,
)
from localpilot.process import hidden_process_creation_flags
from localpilot.windows_webview import make_host_background_transparent

WEBVIEW_DIR = Path(__file__).resolve().parent / "webview"
INDEX_HTML = WEBVIEW_DIR / "index.html"

EXPANDED_SIZE = (int(pixel("chat-window-width")), int(pixel("chat-window-height")))
MIN_SIZE = (int(pixel("chat-min-width")), int(pixel("chat-min-height")))
SYSTEMSENSE_WIDTH = int(pixel("systemsense-width"))
SYSTEMSENSE_GAP = int(pixel("systemsense-gap"))
SYSTEMSENSE_SIZE = (EXPANDED_SIZE[0] + SYSTEMSENSE_WIDTH + SYSTEMSENSE_GAP, EXPANDED_SIZE[1])
NATIVE_AVATAR_SIZE = 128
EDGE_INSET = 24
COMPANION_MODULE = "localpilot.native_avatar_companion"

_COMPANION_STATES = frozenset(
    {
        "idle",
        "listening",
        "thinking",
        "researching",
        "working",
        "speaking",
        "success",
        "uncertain",
        "error",
        "learning",
        "restarting",
        "sleeping",
        "offline",
    }
)

_ANCHOR_BOTTOM_RIGHT = FixPoint.SOUTH | FixPoint.EAST


def _desktop_python_executable(
    executable: str | Path | None = None,
    *,
    platform_name: str | None = None,
) -> Path:
    """Prefer ``pythonw.exe`` for persistent Windows GUI processes."""
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


def _launch_module_detached(
    module: str,
    root: Path,
    config_path: str | None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> bool:
    executable = _desktop_python_executable()
    if os.name == "nt" and executable.name.lower() != "pythonw.exe":
        return False

    argv = [str(executable), "-m", module, "--root", str(root)]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    if x is not None and y is not None:
        argv.extend(["--x", str(int(x)), "--y", str(int(y))])

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    try:
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
    except OSError:
        return False
    return True


def _launch_detached(root: Path, config_path: str | None) -> bool:
    """Normal ``localpilot desktop`` starts the persistent Astra companion."""
    return _launch_module_detached(COMPANION_MODULE, root, config_path)


def _launch_native_avatar(
    root: Path,
    config_path: str | None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> bool:
    return _launch_module_detached(COMPANION_MODULE, root, config_path, x=x, y=y)


def _startup_shortcut_path() -> Path:
    """Location of the Startup-folder entry used for 'start with Windows'."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "LocalPilot.lnk"


def _write_startup_shortcut(target: Path, root: Path, config_path: str | None) -> None:
    """Create a .lnk that starts the persistent illustrated companion at login."""
    exe = str(_desktop_python_executable())
    argv = ["-m", COMPANION_MODULE, "--root", str(root.resolve())]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    arguments = subprocess.list2cmdline(argv)

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
        creationflags=hidden_process_creation_flags(),
    )


class WindowBridge:
    """Window-manager bridge only; conversation data remains on the broker."""

    def __init__(
        self,
        window: webview.Window,
        root: Path,
        config_path: str | None,
        *,
        avatar_external: bool = False,
    ) -> None:
        self._window = window
        self._root = root
        self._config_path = config_path
        self._config = load_config(config_path)
        self._state = DesktopUIState(root / self._config.agent.data_dir)
        self._avatar_spawned = avatar_external
        self._avatar_external = bool(avatar_external)
        self._exit_requested = False
        self._systemsense_open = False

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    @property
    def avatar_external(self) -> bool:
        return self._avatar_external

    def _surface_size(self) -> tuple[int, int]:
        extra = SYSTEMSENSE_WIDTH + SYSTEMSENSE_GAP if self._systemsense_open else 0
        return (max(MIN_SIZE[0] + extra, int(self._window.width)),
                max(MIN_SIZE[1], int(self._window.height)))

    def expand(self) -> dict[str, Any]:
        w, h = self._surface_size()
        self._window.resize(w, h, fix_point=self._resize_anchor())
        return {"ok": True}

    def _resize_anchor(self):
        return (FixPoint.SOUTH | FixPoint.WEST) if getattr(self._window, "_comic_tail_left", False) else _ANCHOR_BOTTOM_RIGHT

    def set_systemsense_open(self, value: bool) -> dict[str, Any]:
        """Reserve transparent space to the left for the real SystemSense panel."""
        enabled = bool(value)
        if enabled == self._systemsense_open:
            return {"ok": True, "open": enabled}
        w, h = self._surface_size()
        extra = SYSTEMSENSE_WIDTH + SYSTEMSENSE_GAP
        w += extra if enabled else -extra
        self._systemsense_open = enabled
        self._window._comic_systemsense_open = enabled
        self._window.min_size = (MIN_SIZE[0] + (extra if enabled else 0), MIN_SIZE[1])
        # Lower the minimum before closing. Raising it before opening would
        # enlarge WinForms at its top-left corner, defeating the resize anchor.
        if not enabled:
            make_host_background_transparent(self._window)
        self._window.resize(w, h, fix_point=self._resize_anchor())
        make_host_background_transparent(self._window)
        return {"ok": True, "open": enabled}

    def _avatar_position(self) -> tuple[int | None, int | None]:
        values = self._state.read()
        saved_x = values.get("avatar_x")
        saved_y = values.get("avatar_y")
        if isinstance(saved_x, int) and isinstance(saved_y, int):
            return saved_x, saved_y
        try:
            x = int(self._window.x + self._window.width + EDGE_INSET)
            y = int(self._window.y + self._window.height - NATIVE_AVATAR_SIZE)
            return x, y
        except Exception:
            return None, None

    def ensure_avatar(self) -> bool:
        if self._avatar_spawned:
            return True
        x, y = self._avatar_position()
        if _launch_native_avatar(self._root, self._config_path, x=x, y=y):
            self._avatar_spawned = True
            if isinstance(x, int) and isinstance(y, int):
                self._state.update(avatar_x=x, avatar_y=y)
            return True
        return False

    def set_companion_state(self, state: str) -> dict[str, Any]:
        """Publish the real chat UI state to the already-running native Astra."""
        normalized = str(state or "").strip().lower()
        if not self._avatar_external:
            return {"ok": False, "reason": "no-external-avatar"}
        if normalized not in _COMPANION_STATES:
            return {"ok": False, "reason": "invalid-state"}
        self._state.update(companion_state=normalized)
        return {"ok": True, "state": normalized}

    def clear_companion_state(self) -> dict[str, Any]:
        self._state.update(companion_state=None)
        return {"ok": True}

    def collapse(self) -> dict[str, Any]:
        if not self._avatar_external and not self.ensure_avatar():
            return {"ok": False, "reason": "native-avatar-launch-failed"}
        self.clear_companion_state()
        self._window.destroy()
        return {"ok": True}

    def exit_companion(self) -> dict[str, Any]:
        """Close this chat surface; an external avatar remains untouched."""
        self._exit_requested = True
        self.clear_companion_state()
        self._window.destroy()
        return {"ok": True}

    def mark_native_close(self, *_args: Any) -> None:
        self.clear_companion_state()
        if self._avatar_external or not self._avatar_spawned:
            self._exit_requested = True

    def set_always_on_top(self, value: bool) -> dict[str, Any]:
        enabled = bool(value)
        native = getattr(self._window, "native", None)
        if native is not None and bool(getattr(native, "InvokeRequired", False)):
            # pywebview's WinForms set_on_top assigns TopMost directly. Its
            # exposed API runs on a worker thread; marshal before changing it.
            from System import Action

            native.Invoke(Action(lambda: setattr(self._window, "on_top", enabled)))
        else:
            self._window.on_top = enabled
        self._state.update(always_on_top=enabled)
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

    def get_update_settings(self) -> dict[str, Any]:
        return current_update_status(self._root, self._state).as_dict()

    def set_automatic_updates(self, value: bool) -> dict[str, Any]:
        return set_desktop_automatic_updates(self._root, self._state, bool(value)).as_dict()

    def check_for_updates(self) -> dict[str, Any]:
        return check_desktop_updates(
            self._root,
            self._state,
            remote=self._config.github.remote,
            main_branch=self._config.github.main_branch,
        ).as_dict()

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
    return {
        "baseUrl": client.base_url,
        "token": client.token,
        "hasConfigPath": bool(config_path),
    }


def _initial_screen() -> Any | None:
    try:
        return webview.screens[0]
    except Exception:
        return None


def _screen_for_position(x: int, y: int) -> Any | None:
    try:
        for screen in webview.screens:
            if (
                int(screen.x) <= x < int(screen.x + screen.width)
                and int(screen.y) <= y < int(screen.y + screen.height)
            ):
                return screen
    except Exception:
        return None
    return None


def _position_on_screen(screen: Any, width: int, height: int) -> tuple[int, int]:
    return (
        int(screen.x + screen.width - width - NATIVE_AVATAR_SIZE - EDGE_INSET),
        int(screen.y + screen.height - height - EDGE_INSET),
    )


def _initial_position(
    width: int,
    height: int,
    screen: Any | None = None,
) -> tuple[int | None, int | None]:
    screen = _initial_screen() if screen is None else screen
    if screen is None:
        return None, None
    try:
        return _position_on_screen(screen, width, height)
    except (AttributeError, TypeError, ValueError):
        return None, None


def _install_expanded_window_chrome(window: webview.Window) -> None:
    """Install only the drag affordance; the real toolbar owns close/collapse."""
    script = r"""
(() => {
  const title = document.querySelector('.header-text');
  if (title) title.classList.add('pywebview-drag-region');
})();
"""
    window.evaluate_js(script)


def main(
    root: str | Path,
    config_path: str | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
    companion: bool = False,
    physical_position: bool = False,
    tail_left: bool = False,
) -> None:
    root = Path(root).resolve()

    if _should_detach_gui(sys.argv[0]) and _launch_detached(root, config_path):
        return

    config = load_config(config_path)
    client = ensure_broker(root, config, config_path=config_path)

    width, height = EXPANDED_SIZE
    if x is not None and y is not None:
        screen = _screen_for_position(int(x), int(y))
        window_x, window_y = int(x), int(y)
    else:
        screen = _initial_screen()
        window_x, window_y = _initial_position(width, height, screen)

    ui_state = DesktopUIState(root / config.agent.data_dir).read()
    window = webview.create_window(
        "LocalPilot",
        url=str(INDEX_HTML),
        width=width,
        height=height,
        x=window_x,
        y=window_y,
        screen=screen,
        min_size=MIN_SIZE,
        frameless=True,
        easy_drag=True,
        shadow=False,
        on_top=bool(ui_state.get("always_on_top", True)),
        background_color="#F8F2E8",
        transparent=True,
    )

    bridge = WindowBridge(window, root, config_path, avatar_external=companion)
    window._comic_tail_left = tail_left
    window.expose(
        bridge.expand,
        bridge.collapse,
        bridge.exit_companion,
        bridge.set_systemsense_open,
        bridge.set_companion_state,
        bridge.clear_companion_state,
        bridge.set_always_on_top,
        bridge.get_start_with_windows,
        bridge.set_start_with_windows,
        bridge.get_update_settings,
        bridge.set_automatic_updates,
        bridge.check_for_updates,
        bridge.open_config_file,
    )

    def on_shown() -> None:
        # WinForms removes its frame after assigning the initial Size, which
        # otherwise leaves the actual client 16x39 pixels short at 100% DPI.
        # Reapply the requested logical size once the frameless Form exists.
        window.resize(width, height)
        if physical_position and x is not None and y is not None and os.name == "nt":
            # Tk hands off absolute physical desktop coordinates. pywebview's
            # launch x/y are logical, so place the native Form after creation.
            from System import Action
            from System.Drawing import Point

            window.native.Invoke(Action(lambda: setattr(window.native, "Location", Point(int(x), int(y)))))
        make_host_background_transparent(window)

    def on_loaded() -> None:
        # Retry after WebView2 has finished loading as well.  This covers the
        # small timing window where the native Form exists but the transparent
        # Chromium child has not finished attaching to it yet.
        make_host_background_transparent(window)
        window.evaluate_js("document.getElementById('app').classList.add('is-expanded')")
        if tail_left:
            window.evaluate_js("document.getElementById('app').classList.add('tail-left')")
        _install_expanded_window_chrome(window)
        payload = _bridge_payload(client, config_path)
        window.evaluate_js(f"window.__initLocalPilot({json.dumps(payload)})")

    window.events.shown += on_shown
    window.events.loaded += on_loaded
    try:
        window.events.closing += bridge.mark_native_close
    except AttributeError:
        pass

    try:
        webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False)
    finally:
        bridge.clear_companion_state()
        if not bridge.exit_requested and not bridge.avatar_external:
            bridge.ensure_avatar()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-desktop-chat")
    parser.add_argument("--root", required=True, help="LocalPilot project root")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--physical-position", action="store_true", help="Coordinates are native physical desktop pixels")
    parser.add_argument("--tail-left", action="store_true", help="Chat is to the right of the native companion")
    parser.add_argument(
        "--companion",
        action="store_true",
        help="Chat was launched by an already-running native avatar",
    )
    return parser


def cli_main() -> None:
    args = build_parser().parse_args()
    main(
        args.root,
        args.config,
        x=args.x,
        y=args.y,
        companion=args.companion,
        physical_position=args.physical_position,
        tail_left=args.tail_left,
    )


if __name__ == "__main__":
    cli_main()
