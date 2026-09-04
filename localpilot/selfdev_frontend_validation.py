"""Frontend (HTML/CSS/JS) candidate-write validation for the self-development
pipeline.

Extracted verbatim (no logic changes) from selfdev.py. Enforces that
frontend writes stay inside localpilot/webview, declare the exact required
CSP, and contain no active-content/XSS-sink/unapproved-bridge-call
patterns. Re-imported as bare names into selfdev.py wherever still
referenced there (CandidateTools.validate_project_write and friends)."""

import re
from html.parser import HTMLParser
from pathlib import Path

_FRONTEND_SUFFIXES = {".html", ".css", ".js"}

_FRONTEND_ROOT = Path("localpilot/webview")

_FRONTEND_BRIDGE_METHODS = {
    "expand",
    "collapse",
    "set_always_on_top",
    "get_start_with_windows",
    "set_start_with_windows",
    "open_config_file",
}

_REQUIRED_CSP = {
    "default-src": {"'none'"},
    "script-src": {"'self'"},
    "style-src": {"'self'"},
    "connect-src": {"http://127.0.0.1:*", "http://localhost:*"},
    "img-src": {"'self'", "data:"},
    "font-src": {"'self'"},
    "object-src": {"'none'"},
    "base-uri": {"'none'"},
    "form-action": {"'none'"},
    "worker-src": {"'none'"},
}


def _is_frontend_path(relative: Path) -> bool:
    return relative == _FRONTEND_ROOT or _FRONTEND_ROOT in relative.parents


def _validate_frontend_reference(owner: Path, value: str) -> None:
    reference = str(value).strip()
    if not reference or reference.startswith("#"):
        return
    if (
        reference.startswith(("//", "\\\\", "/", "\\"))
        or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", reference)
    ):
        raise ValueError("Frontend resources must be local relative files.")
    path_text = reference.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    raw = Path(path_text)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Frontend resource reference escapes localpilot/webview.")
    target = owner.parent / raw
    if not _is_frontend_path(target):
        raise ValueError("Frontend resource reference escapes localpilot/webview.")


def _parse_csp(content: str) -> dict[str, set[str]]:
    directives: dict[str, set[str]] = {}
    for raw in content.split(";"):
        tokens = raw.strip().split()
        if tokens:
            directives[tokens[0].lower()] = set(tokens[1:])
    return directives


class _CandidateHTMLValidator(HTMLParser):
    """Reject active or remote HTML outside the companion's fixed policy."""

    _FORBIDDEN_TAGS = {"base", "embed", "form", "iframe", "object"}
    _REFERENCE_ATTRIBUTES = {"href", "poster", "src"}

    def __init__(self, relative: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.relative = relative
        self.csp: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in self._FORBIDDEN_TAGS:
            raise ValueError(f"Frontend HTML tag is not allowed: {normalized_tag}")
        values = {str(name).lower(): value for name, value in attrs}
        for name, value in values.items():
            if name.startswith("on") or name == "style":
                raise ValueError(f"Inline frontend HTML attribute is not allowed: {name}")
            if name in self._REFERENCE_ATTRIBUTES and value is not None:
                _validate_frontend_reference(self.relative, value)
        if normalized_tag == "script" and not values.get("src"):
            raise ValueError("Inline frontend scripts are not allowed.")
        if (
            normalized_tag == "meta"
            and str(values.get("http-equiv") or "").lower() == "content-security-policy"
        ):
            self.csp = str(values.get("content") or "")

    handle_startendtag = handle_starttag


def _validate_frontend_candidate(relative: Path, content: str) -> None:
    suffix = relative.suffix.lower()
    if suffix not in _FRONTEND_SUFFIXES:
        return
    if not _is_frontend_path(relative):
        raise ValueError("Frontend files may be edited only inside localpilot/webview.")

    if suffix == ".html":
        parser = _CandidateHTMLValidator(relative)
        parser.feed(content)
        parser.close()
        if parser.csp is None:
            raise ValueError("Frontend HTML must declare a Content-Security-Policy meta tag.")
        directives = _parse_csp(parser.csp)
        for directive, required in _REQUIRED_CSP.items():
            if directives.get(directive) != required:
                raise ValueError(f"Frontend CSP must keep the exact {directive} policy.")
        if "'unsafe-inline'" in parser.csp or "'unsafe-eval'" in parser.csp:
            raise ValueError("Frontend CSP may not enable unsafe inline or eval execution.")
        return

    if suffix == ".css":
        if re.search(r"@import\b|expression\s*\(|-moz-binding\b|\bbehavior\s*:", content, re.I):
            raise ValueError("Frontend CSS contains a disallowed active-content feature.")
        for match in re.finditer(r"url\(\s*(['\"]?)(.*?)\1\s*\)", content, re.I):
            _validate_frontend_reference(relative, match.group(2))
        return

    forbidden_javascript = {
        "dynamic code execution": r"\beval\s*\(|\bnew\s+Function\b|\bFunction\s*\(",
        "HTML injection sink": r"\.innerHTML\b|\.outerHTML\b|insertAdjacentHTML\s*\(|document\.write\s*\(",
        "browser persistence": r"\blocalStorage\b|\bsessionStorage\b|\bindexedDB\b|document\.cookie\b",
        "page navigation": r"\b(?:window\.)?(?:location|opener|parent|top)\b\s*(?:=|\.)",
        "alternate network channel": r"\bWebSocket\s*\(|\bEventSource\s*\(|sendBeacon\s*\(|XMLHttpRequest\b",
        "remote URL": r"(?:https?|wss?)://|['\"]//",
        "direct native bridge call": r"window\.pywebview\.api\.[A-Za-z_$]",
    }
    for label, pattern in forbidden_javascript.items():
        if re.search(pattern, content):
            raise ValueError(f"Frontend JavaScript contains a disallowed {label}.")
    for match in re.finditer(r"\bbridge\s*\(\s*(['\"])([^'\"]+)\1", content):
        if match.group(2) not in _FRONTEND_BRIDGE_METHODS:
            raise ValueError(f"Frontend JavaScript requests an unapproved native bridge method: {match.group(2)}")

