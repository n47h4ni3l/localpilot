"""WebView2-hosted conversation surface for the LocalPilot desktop companion.

The illustrated Astra avatar is a separate native transparent window. The
normal desktop path keeps that avatar alive while this module hosts only the
comic speech-bubble chat. SystemSense remains the same authenticated read-only
surface and is styled as a separate notepad below the conversation.
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
from localpilot.desktop_state import DesktopUIState
from localpilot.process import hidden_process_creation_flags

WEBVIEW_DIR = Path(__file__).resolve().parent / "webview"
INDEX_HTML = WEBVIEW_DIR / "index.html"

EXPANDED_SIZE = (500, 640)
MIN_SIZE = (420, 520)
NATIVE_AVATAR_SIZE = 128
EDGE_INSET = 24
COMPANION_MODULE = "localpilot.native_avatar_companion"

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
        self._state = DesktopUIState(root / load_config(config_path).agent.data_dir)
        self._avatar_spawned = avatar_external
        self._avatar_external = bool(avatar_external)
        self._exit_requested = False

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    @property
    def avatar_external(self) -> bool:
        return self._avatar_external

    def expand(self) -> dict[str, Any]:
        w, h = EXPANDED_SIZE
        self._window.resize(w, h, fix_point=_ANCHOR_BOTTOM_RIGHT)
        return {"ok": True}

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

    def collapse(self) -> dict[str, Any]:
        if not self._avatar_external and not self.ensure_avatar():
            return {"ok": False, "reason": "native-avatar-launch-failed"}
        self._window.destroy()
        return {"ok": True}

    def exit_companion(self) -> dict[str, Any]:
        """Close this chat surface; an external avatar remains untouched."""
        self._exit_requested = True
        self._window.destroy()
        return {"ok": True}

    def mark_native_close(self, *_args: Any) -> None:
        if self._avatar_external or not self._avatar_spawned:
            self._exit_requested = True

    def set_always_on_top(self, value: bool) -> dict[str, Any]:
        enabled = bool(value)
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
    """Install a drag affordance and close button without remote assets."""
    script = r"""
(() => {
  const header = document.querySelector('.panel-header');
  if (!header) return;
  const title = document.querySelector('.header-text');
  if (title) title.classList.add('pywebview-drag-region');
  if (document.getElementById('close-app-btn')) return;
  const button = document.createElement('button');
  button.className = 'icon-btn';
  button.id = 'close-app-btn';
  button.type = 'button';
  button.setAttribute('aria-label', 'Close chat bubble');
  button.textContent = '×';
  button.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (window.pywebview && window.pywebview.api && window.pywebview.api.exit_companion) {
      window.pywebview.api.exit_companion();
    }
  });
  header.appendChild(button);
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
        shadow=True,
        on_top=bool(ui_state.get("always_on_top", True)),
        background_color="#F8F2E8",
        transparent=False,
    )

    bridge = WindowBridge(window, root, config_path, avatar_external=companion)
    window.expose(
        bridge.expand,
        bridge.collapse,
        bridge.exit_companion,
        bridge.set_always_on_top,
        bridge.get_start_with_windows,
        bridge.set_start_with_windows,
        bridge.open_config_file,
    )

    def on_loaded() -> None:
        window.evaluate_js("document.getElementById('app').classList.add('is-expanded')")
        _install_expanded_window_chrome(window)
        payload = _bridge_payload(client, config_path)
        window.evaluate_js(f"window.__initLocalPilot({json.dumps(payload)})")

    window.events.loaded += on_loaded
    try:
        window.events.closing += bridge.mark_native_close
    except AttributeError:
        pass

    try:
        webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False)
    finally:
        if not bridge.exit_requested and not bridge.avatar_external:
            bridge.ensure_avatar()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot-desktop-chat")
    parser.add_argument("--root", required=True, help="LocalPilot project root")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
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
    )


if __name__ == "__main__":
    cli_main()
