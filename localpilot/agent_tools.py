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

from localpilot.tools.web import canonical_public_web_url, _DEFAULT_MAX_CHARS

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
    "inspect_process_identity",
    "get_startup_items",
    "get_active_power_plan",
    "get_defender_summary",
    "get_device_problem_summary",
    "get_system_sense_summary",
    "get_machine_location",
    "inspect_hardware_inventory",
    "inspect_driver_inventory",
    "get_system_sense_history",
    "get_systemsense_watch_report",
    "inspect_systemsense_watch_process",
    "get_memory_watch_report",
    "inspect_memory_watch_process",
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
    if name == "get_machine_location":
        return "machine location"
    if name in {"get_systemsense_watch_report", "get_memory_watch_report"}:
        return "SystemSense watch"
    if name in {"inspect_systemsense_watch_process", "inspect_memory_watch_process"}:
        # Historical watch investigations must use the watch-bound inspector
        # because it verifies process instance identity and preserves the
        # observation-time executable fingerprint.
        return "process identity"
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
    """Recognize real result failures without treating quoted source as errors."""
    text = str(result).strip().lower()
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Repository sources prefix each returned source line with its number.
    # An error-like string *inside* such a line remains legitimate evidence.
    if lines[0].startswith("repository file: "):
        return len(lines) > 1
    if lines[0].startswith("repository search: "):
        return len(lines) > 1 and lines[1:] != ["no matches found."]
    # Other registered readers may wrap an actual error on their second
    # line ("Private GitHub ...\nGitHub read failed: ...", or
    # "Public web search ...\nNo bounded HTTPS results were found.").
    # Check only the envelope, never an arbitrary occurrence deep in data.
    return not any(
        line.startswith(marker)
        for line in lines[:2]
        for marker in _TOOL_FAILURE_MARKERS
    )


def _tool_result_audit_preview(name: str, result: Any) -> str:
    """Keep one-use action capabilities out of durable audit previews."""
    if name == "consult_external_mentor":
        # Never retain hosted responses or transmitted questions in durable audit.
        return "[External mentor response omitted from durable audit]"
    if name == "set_active_power_plan" and isinstance(result, dict):
        safe_result = dict(result)
        if safe_result.get("rollback_token"):
            safe_result["rollback_token"] = "<redacted>"
        return str(safe_result)[:1200]
    return str(result)[:1200]


def _tool_arguments_for_audit(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact one-use capabilities while preserving reviewable tool intent."""
    if name == "consult_external_mentor":
        return {"question_chars": len(str(arguments.get("question") or ""))}
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


# GPT-OSS/Ollama sometimes appends a channel/format marker to a tool name.
# Only these observed, exact suffixes are transport noise, not new tools.
_KNOWN_CHANNEL_SUFFIXES = ("<|channel|>commentary", "<|channel|>json")


def _advertised_tool_names(tools: list[Any] | None) -> frozenset[str]:
    """Extract only tool names actually offered in this exact model request."""
    names: set[str] = set()
    for tool in tools or ():
        if isinstance(tool, dict):
            function = tool.get("function")
            if not isinstance(function, dict):
                function = tool
            name = function.get("name")
        else:
            name = getattr(tool, "__name__", None)
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def _recover_readonly_tool_call(
    call: Any, *, exposed: frozenset[str], recoverable: frozenset[str],
) -> tuple[Any, str | None]:
    """Recover only uniquely identified read-only calls; never infer arguments.

    A name must resolve to a registered, policy-permitted READ_ONLY tool
    that was explicitly advertised to the model on this request. Preserve
    unsupported/ambiguous names verbatim for the ordinary unknown-tool path.
    """
    original, arguments = _tool_call_parts(call)
    if original in exposed or not original or len(original) > 180:
        return call, None
    candidates: list[tuple[str, str]] = []
    if original.count("<|channel|>") == 1:
        for suffix in _KNOWN_CHANNEL_SUFFIXES:
            if original.endswith(suffix):
                candidates.append((original[:-len(suffix)], "gpt_oss_channel_suffix"))
    stripped = original.strip(" \t\r\n")
    if stripped != original:
        candidates.append((stripped, "surrounding_whitespace"))
    # A case-only difference is safe when exactly one exposed tool matches.
    case_matches = [name for name in exposed if name.casefold() == original.casefold()]
    if len(case_matches) == 1:
        candidates.append((case_matches[0], "case_only"))

    # One observed legacy namespace spelling with an independently verified
    # compatible repository-search signature. Never translate arbitrary
    # namespaces or infer/rename missing argument keys.
    if (original == "repo_browser.search"
            and isinstance(arguments.get("query"), str)
            and bool(arguments["query"].strip())
            and set(arguments) <= {"query", "path", "max_results"}):
        candidates.append(("search_repository", "known_readonly_alias"))

    matches = [(name, kind) for name, kind in candidates if name in exposed and name in recoverable]
    if len({name for name, _ in matches}) != 1:
        return call, None
    target, kind = matches[0]
    if isinstance(call, dict) and isinstance(call.get("function"), dict):
        return {
            **call,
            "function": {**call["function"], "name": target},
        }, kind
    # Ollama may supply a typed ToolCall; use a plain wire-equivalent
    # call without altering the original object or any supplied arguments.
    return {"function": {"name": target, "arguments": arguments}}, kind


def _tool_cache_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    args = dict(args)
    if name == "fetch_public_https":
        args["url"] = canonical_public_web_url(args.get("url", ""))
        args.setdefault("max_chars", _DEFAULT_MAX_CHARS)
        args.setdefault("start_char", 0)
    elif name == "search_public_web" and "query" in args:
        args["query"] = " ".join(str(args["query"]).casefold().split())
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)

