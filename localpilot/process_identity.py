from __future__ import annotations

import re
from collections.abc import Iterable


_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|pwd|token|secret|api[-_]?key|authorization|"
    r"cookie|session|credential|client[-_]?secret|access[-_]?key)",
    re.IGNORECASE,
)
_URL_CREDENTIAL = re.compile(
    r"(?P<prefix>https?://[^/\s:@]+:)(?P<secret>[^@\s/]+)@",
    re.IGNORECASE,
)


def sanitize_command_line(parts: Iterable[object]) -> str:
    """Return a bounded command line with likely credential values redacted.

    Process command lines are useful identity evidence but can contain secrets.
    LocalPilot never needs those secret values to identify an executable, so
    redact common credential-bearing flag values before persistence/model use.
    """
    output: list[str] = []
    redact_next = False
    for raw in parts:
        arg = str(raw or "")
        if redact_next:
            output.append("<redacted>")
            redact_next = False
            continue

        if "=" in arg:
            key, value = arg.split("=", 1)
            if _SENSITIVE_KEY.search(key):
                output.append(f"{key}=<redacted>")
                continue

        normalized_key = arg.lstrip("-/").rstrip(":")
        if _SENSITIVE_KEY.fullmatch(normalized_key):
            output.append(arg)
            redact_next = True
            continue

        arg = _URL_CREDENTIAL.sub(
            lambda match: match.group("prefix") + "<redacted>@",
            arg,
        )
        output.append(arg)

    return " ".join(output)[:4000]
