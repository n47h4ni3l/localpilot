from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class DesktopUIState:
    """Tiny private state store shared by the native avatar and WebView chat."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir).resolve() / "desktop-ui-state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def defaults() -> dict[str, Any]:
        return {
            "avatar_x": None,
            "avatar_y": None,
            "always_on_top": True,
        }

    def read(self) -> dict[str, Any]:
        values = self.defaults()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return values
        if not isinstance(raw, dict):
            return values
        for key in values:
            if key in raw:
                values[key] = raw[key]
        if not isinstance(values["avatar_x"], int):
            values["avatar_x"] = None
        if not isinstance(values["avatar_y"], int):
            values["avatar_y"] = None
        values["always_on_top"] = bool(values["always_on_top"])
        return values

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            values = self.read()
            for key, value in changes.items():
                if key in values:
                    values[key] = value
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path)
            return values
