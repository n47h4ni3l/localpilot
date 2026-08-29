from __future__ import annotations

import os
import subprocess


def hidden_process_creation_flags() -> int:
    """Return flags that keep console-mode child processes hidden on Windows."""
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
