"""Narrow source-answer checks for inspected repository symbols and line citations.

Checks only exact, clearly attributed names and source-line locators. They do
not attempt to grade all numerical reasoning, prove absence of an unseen
implementation, or turn an incomplete scan into a complete source map.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_READ_HEADER = re.compile(r"^Repository file:\s+(.+?)\s+lines\s+\d+-\d+", re.I)
_READ_LINE = re.compile(r"^(\d+):\s*(.*)$")
_SEARCH_LINE = re.compile(r"^([^:\s][^:\n]*\.(?:py|toml|ps1|js|json|md|yml|yaml|txt)):(\d+):\s*(.*)$", re.I)
_SOURCE_REFERENCE = re.compile(
    r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:py|toml|ps1|js|json|md|yml|yaml|txt)):(\d+)\b",
    re.I,
)
# Match named, inspectable constants, not every numeral in natural language.
_NAMED_NUMBER = re.compile(
    r"(?:"
    r"\x60(?P<quoted>[A-Za-z_][\w.]*)\x60|"
    r"(?P<upper>_?[A-Z][A-Z0-9_]{2,})|"
    r"(?P<snake>[a-z][a-z0-9]*_[a-z0-9_]+)"
    r")\s*(?:=|:|\bis\b|\bequals?\b|\bdefaults?\s+to\b|\bcapped\s+at\b|"
    r"\blimited\s+to\b|\bset\s+to\b)\s*(?P<number>\d+(?:\.\d+)?)\b",
)


def inspected_repository_lines(
    messages: list[dict[str, Any]],
) -> dict[str, dict[int, str]]:
    """Index only numbered text genuinely returned by permitted read tools."""
    index: dict[str, dict[int, str]] = defaultdict(dict)
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_name") not in {
            "read_repository_file", "search_repository",
        }:
            continue
        body = str(message.get("content") or "")
        if "Tool error:" in body or "Not executed:" in body:
            continue
        path = None
        for line in body.splitlines():
            header = _READ_HEADER.match(line)
            if header and message.get("tool_name") == "read_repository_file":
                path = header.group(1)
                continue
            if message.get("tool_name") == "read_repository_file" and path:
                numbered = _READ_LINE.match(line)
                if numbered:
                    index[path][int(numbered.group(1))] = numbered.group(2)
            elif message.get("tool_name") == "search_repository":
                hit = _SEARCH_LINE.match(line)
                if hit:
                    index[hit.group(1)][int(hit.group(2))] = hit.group(3)
    return dict(index)


def repository_answer_risks(
    prompt: str,
    answer: str,
    messages: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Verify explicit symbol=value claims and citations against observed lines.

    The user can supply hypothetical numerical premises; those are not
    independently verified repository values and must remain labelled as
    premises in the answer, rather than silently becoming live source facts.
    """
    request = str(prompt)
    content = str(answer)
    if not re.search(r"\b(?:repo(?:sitory)?|source code|codebase|implementation|"
                     r"module|function|class|call path|imports?)\b", request, re.I):
        return ()
    source = inspected_repository_lines(messages)
    if not source:
        return ()
    risks: list[str] = []

    citations = [(path, int(line)) for path, line in _SOURCE_REFERENCE.findall(content)]
    if citations and any(line not in source.get(path, {}) for path, line in citations):
        risks.append("repository_line_citation_unverified")

    requested_locations = re.search(
        r"\b(?:cite|citing|citations?|file[- ]and[- ]line|source locations?|"
        r"implementation locations?|line references?)\b",
        request, re.I,
    )
    if requested_locations and not citations and not re.search(
        r"\b(?:no (?:repository|source)|not inspected|unavailable|"
        r"unresolved|could not inspect|cannot verify)\b", content, re.I,
    ):
        risks.append("requested_repository_line_citations_missing")

    all_source_lines = [line for lines in source.values() for line in lines.values()]
    for match in _NAMED_NUMBER.finditer(content):
        key = next((g for g in (match.group("quoted"), match.group("upper"),
                                match.group("snake")) if g), "")
        value = match.group("number")
        start = content[max(0, match.start() - 65):match.start()]
        if re.search(
            r"\b(?:not|isn['’]?t|never|incorrect(?:ly)?|instead of|"
            r"not (?:established|supported|confirmed)|"
            r"does not (?:establish|prove)(?: that)?)\s*(?::|that)?\s*$",
            start, re.I,
        ):
            continue
        # The prompt may explicitly supply a worked example. Its numerical
        # premise is not a claim about the checked repository.
        if any(
            original.group("number") == value
            and key == next((g for g in (
                original.group("quoted"), original.group("upper"), original.group("snake")
            ) if g), "")
            for original in _NAMED_NUMBER.finditer(request)
        ) and re.search(r"\b(?:suppose|assume|given|hypothetical|example|premise)\b",
                        request, re.I):
            continue
        symbol = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)", re.I)
        numeric = re.compile(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])")
        # A numeric token appearing elsewhere on the page is not adequate:
        # the named symbol and exact value must co-occur in an observed line.
        if not any(symbol.search(line) and numeric.search(line)
                   for line in all_source_lines):
            risks.append("named_numeric_constant_not_supported_by_source")
            break

    # "Complete integration" is stronger than the small source slices
    # generally available. A traced cross-file implementation needs at least
    # two distinct observed files, unless the answer explicitly scopes gaps.
    cross_module_request = re.search(
        r"\b(?:across (?:modules|files)|cross[- ]module|"
        r"end[- ]to[- ]end (?:implementation|call path)|"
        r"trace (?:the )?(?:integration|implementation) (?:path|flow))\b",
        request, re.I,
    )
    claimed_complete = re.search(
        r"\b(?:fully|completely|entire|end[- ]to[- ]end)\s+"
        r"(?:traced|verified|mapped|confirmed)\b|"
        r"\b(?:complete|entire)\s+(?:call chain|implementation path|integration flow)\b",
        content, re.I,
    )
    qualified = re.search(
        r"\b(?:partial|not fully|not complete|unverified|unresolved|"
        r"only (?:the )?inspected|not inspected|cannot establish|"
        r"remaining (?:files|links|hops))\b",
        content, re.I,
    )
    if cross_module_request and claimed_complete and len(source) < 2 and not qualified:
        risks.append("cross_module_completeness_not_established")

    return tuple(dict.fromkeys(risks))
