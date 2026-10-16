"""Small, presentation-safe records of executed tools; independent of model reasoning."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def activity_preview(value: Any, limit: int = 1600) -> str:
    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 5:
            return "…"
        if isinstance(item, dict):
            return {str(key): "[redacted]" if re.search(
                r"password|secret|authorization|api[_-]?key|access[_-]?token|data_base64|credential",
                str(key), re.I) else clean(val, depth + 1) for key, val in list(item.items())[:40]}
        if isinstance(item, (list, tuple)):
            return [clean(val, depth + 1) for val in item[:30]]
        return str(item)[:limit] if isinstance(item, str) else item

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    text = clean(value)
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", text)
    text = re.sub(r'''(?i)((?:password|passwd|api[_-]?key|secret|access[_-]?token|authorization|credential)["']?\s*[:=]\s*)'''
                  r'''(?:"(?:\\.|[^"\\])*(?:"|$)|'(?:\\.|[^'\\])*(?:'|$)|[^\s,;]+)''',
                  r"\1[redacted]", text)
    return text[:limit] + ("… [preview truncated]" if len(text) > limit else "")


def update_steps(steps: list[dict[str, Any]], event_type: str,
                 payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    tool = str(payload.get("tool") or "Unknown tool")[:160]
    step = None
    if event_type == "tool.completed":
        step = next((item for item in reversed(steps)
                     if item.get("tool") == tool and item.get("status") == "running"), None)
    if step is None:
        step = {"id": str(uuid.uuid4()), "tool": tool, "status": "running", "started_at": now}
        steps.append(step)
    if "arguments_preview" in payload:
        step["arguments_preview"] = activity_preview(payload["arguments_preview"])
    if event_type == "tool.completed":
        step.update(status="succeeded" if payload.get("ok") else "failed", completed_at=now)
        started = datetime.fromisoformat(step["started_at"])
        step["duration_ms"] = max(0, round((datetime.fromisoformat(now) - started).total_seconds() * 1000))
        if "result_preview" in payload:
            step["result_preview"] = activity_preview(payload["result_preview"])
        if not payload.get("ok"):
            step["error"] = step.get("result_preview") or "The tool did not complete successfully."
    return steps[-256:], dict(step)
