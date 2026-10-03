from __future__ import annotations

import os
import subprocess


def current_process_elevated() -> bool | None:
    """Return this Windows process's administrator token state when available."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return None


def hidden_process_creation_flags() -> int:
    """Return flags that keep console-mode child processes hidden on Windows."""
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
