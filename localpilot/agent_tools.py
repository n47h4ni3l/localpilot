"""Stateless tool-category data and tool-result helpers for LocalPilotAgent.

Extracted verbatim (no logic changes) from agent.py as part of the low-risk
mechanical decomposition. Every name here is still reachable exactly as
before -- module-level constants are re-imported into agent.py, and every
method is left behind on LocalPilotAgent as a `staticmethod(...)` shim, so
both `self._foo(...)` and `LocalPilotAgent._foo(...)` call sites (both
patterns are used elsewhere in agent.py) keep resolving unchanged."""

import json
import re
from typing import Any

_REPOSITORY_TOOLS = {
    "list_repository_tree",
    "read_repository_file",
    "search_repository",
    "inspect_project_dependencies",
    "get_repository_status",
    "get_runtime_lifecycle",
}

_GITHUB_TOOLS = {
    "get_github_repository",
    "list_github_pull_requests",
    "get_github_pull_request",
    "get_github_pull_request_diff",
    "list_github_issues",
    "get_github_issue",
}

_PC_TOOLS = {
    "get_system_summary",
    "get_storage_summary",
    "get_top_processes",
    "get_startup_items",
    "get_active_power_plan",
    "get_defender_summary",
    "get_device_problem_summary",
    "get_system_sense_summary",
    "inspect_hardware_inventory",
    "inspect_driver_inventory",
    "get_system_sense_history",
    "get_workload_correlations",
    "inspect_raw_system_sense",
}

_LIBRARY_TOOLS = {
    "get_library_summary",
    "search_library",
    "read_library_passage",
}

_TOOL_FAILURE_MARKERS = (
    "tool error:",
    "unknown tool:",
    "requires confirmation and is unavailable",
    "github read failed:",
    "github cli is not available",
    "powershell error:",
    "git is not available.",
    "no bounded https results were found",
    "local library is disabled",
    "local library root does not exist",
    "no indexed library passages matched",
    "library extraction failed",
    "no matches found.",
)


def _forbidden_tools(prompt: str) -> frozenset[str]:
    text = " ".join(str(prompt).lower().split())
    if re.search(
        r"\b(?:do not|don['’]?t|without) (?:use|using|search|browse|consult|access)?\s*"
        r"(?:the )?(?:public )?(?:web|internet|online sources?)\b",
        text,
    ):
        return frozenset({"search_public_web", "fetch_public_https"})
    return frozenset()


def _tool_evidence_source(name: str) -> str | None:
    if name in _REPOSITORY_TOOLS:
        return "trusted repository"
    if name in _GITHUB_TOOLS:
        return "private GitHub"
    if name in _PC_TOOLS:
        return "Windows/PC state"
    if name in _LIBRARY_TOOLS:
        return "local library"
    if name == "fetch_public_https":
        return "public HTTPS"
    if name == "search_public_web":
        return "public web discovery"
    return None


def _tool_result_success(result: Any) -> bool:
    text = str(result).strip().lower()
    return bool(text) and not any(marker in text for marker in _TOOL_FAILURE_MARKERS)


def _tool_result_audit_preview(name: str, result: Any) -> str:
    """Keep one-use action capabilities out of durable audit previews."""
    if name == "set_active_power_plan" and isinstance(result, dict):
        safe_result = dict(result)
        if safe_result.get("rollback_token"):
            safe_result["rollback_token"] = "<redacted>"
        return str(safe_result)[:1200]
    return str(result)[:1200]


def _tool_arguments_for_audit(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact one-use capabilities while preserving reviewable tool intent."""
    safe_arguments = dict(arguments)
    if name == "restore_power_plan" and safe_arguments.get("rollback_token"):
        safe_arguments["rollback_token"] = "<redacted>"
    return safe_arguments


def _chunk_value(chunk: Any, name: str) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name)
    return getattr(chunk, name, None)


def _tool_call_parts(call: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(call, dict):
        fn = call.get("function", {})
        if isinstance(fn, dict):
            return str(fn.get("name") or ""), dict(fn.get("arguments") or {})
    fn = getattr(call, "function", None)
    return str(getattr(fn, "name", "") or ""), dict(getattr(fn, "arguments", None) or {})


def _tool_cache_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)

