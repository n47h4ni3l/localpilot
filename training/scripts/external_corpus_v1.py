"""Deterministic curation primitives for External Corpus v1.

The module is importable with the standard library alone. Heavy acquisition,
Parquet, and tokenizer dependencies are loaded only by the command-line tools.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import re
import tokenize
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol


TRANSFORMATION_VERSION = "external-corpus-v1.0.1"
ACQUISITION_DATE = "2026-09-08"
SEED = 3407
MAX_SEQUENCE_LENGTH = 1024
NEAR_DUPLICATE_THRESHOLD = 0.85
CODE_DUPLICATE_THRESHOLD = 0.90
SHINGLE_WIDTH = 5
SENSITIVE_CREDENTIAL_RE = re.compile(
    r"(?:"
    r"gh[pousr]_[A-Za-z0-9]{36,}"
    r"|github_pat_[A-Za-z0-9_]{40,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|xox[baprs]-[0-9A-Za-z-]{10,}"
    r"|[sr]k_live_[0-9A-Za-z]{16,}"
    r"|glpat-[0-9A-Za-z_-]{20,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|pypi-[A-Za-z0-9_-]{50,}"
    r"|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r")"
)

DATASETS: dict[str, dict[str, Any]] = {
    "nemotron_swe_v2": {
        "repo_id": "nvidia/Nemotron-SFT-SWE-v2",
        "revision": "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
        "license": "CC-BY-4.0 (components: Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause)",
        "ceiling": 6000,
        "paths": ["data/swe.jsonl", "data/agentless.jsonl", "README.md"],
    },
    "open_code_instruct": {
        "repo_id": "nvidia/OpenCodeInstruct",
        "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
        "license": "CC-BY-4.0",
        "ceiling": 6000,
        "paths": ["data/train-00000-of-00050.parquet..data/train-00049-of-00050.parquet", "README.md"],
    },
    "nemotron_agentic_v2": {
        "repo_id": "nvidia/Nemotron-SFT-Agentic-v2",
        "revision": "7c804833427f633ccd53b582dbf02525fd680f78",
        "license": "CC-BY-4.0 (components: Apache-2.0, MIT)",
        "ceiling": 3000,
        "paths": ["data/tool_calling.jsonl", "data/search.jsonl", "README.md"],
    },
    "xlam_60k": {
        "repo_id": "Salesforce/xlam-function-calling-60k",
        "revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
        "license": "CC-BY-4.0",
        "ceiling": 2000,
        "paths": ["xlam_function_calling_60k.json"],
        "expected_sha256": "4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6",
    },
    "codeact_instruct": {
        "repo_id": "neulab/agent-data-collection",
        "source_subdirectory": "codeactinstruct",
        "revision": "68744540161f136acacb2aca1d54bdaee7b00e24",
        "license": "Apache-2.0",
        "ceiling": 1500,
        "paths": ["codeactinstruct/full_std.jsonl", "codeactinstruct/README.md", "codeactinstruct/LICENSE"],
    },
    "swe_care": {
        "repo_id": "inclusionAI/SWE-CARE",
        "revision": "3b3a625ef26bd497a3a13485c0dc9ece537f24d0",
        "license": "Apache-2.0",
        "ceiling": 1500,
        "paths": ["data/dev-00000-of-00001.parquet", "README.md"],
        "excluded_paths": ["data/test-00000-of-00001.parquet"],
    },
}

# CodeActInstruct is a repackaging of several constituent datasets.  A wrapper
# license is not enough provenance for an accepted row, so every accepted
# constituent is tied to an immutable upstream revision and license reference.
# WikiTableQuestions is deliberately absent: the currently published card and
# loader disagree about the exact CC-BY/CC-BY-SA terms, which makes its
# constituent provenance unresolved for this release.
CODEACT_CONSTITUENTS: dict[str, dict[str, str]] = {
    "decision_making/alfworld": {
        "upstream_source_family": "ALFWorld",
        "repository": "alfworld/alfworld",
        "revision": "aaba6870f86c5be6a08a491f32a50b906227bc3e",
        "license": "MIT",
        "license_reference": "https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/LICENSE",
    },
    "reasoning/hotpotqa": {
        "upstream_source_family": "HotpotQA",
        "repository": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "license": "CC-BY-SA-4.0",
        "license_reference": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/blob/1908d6afbbead072334abe2965f91bd2709910ab/README.md",
    },
    "code_generation/APPS": {
        "upstream_source_family": "APPS",
        "repository": "codeparrot/apps",
        "revision": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        "license": "MIT",
        "license_reference": "https://huggingface.co/datasets/codeparrot/apps/blob/21e74ddf8de1a21436da12e3e653065c5213e9d1/README.md",
    },
}
for _math_subject in (
    "algebra",
    "number_theory",
    "intermediate_algebra",
    "prealgebra",
    "counting_and_probability",
    "precalculus",
    "geometry",
):
    CODEACT_CONSTITUENTS[f"reasoning/{_math_subject}"] = {
        "upstream_source_family": "MATH",
        "repository": "hendrycks/competition_math",
        "revision": "71b758ecc688b2822d07ffa7f8393299f1dc7cac",
        "license": "MIT",
        "license_reference": "https://huggingface.co/datasets/hendrycks/competition_math/blob/71b758ecc688b2822d07ffa7f8393299f1dc7cac/README.md",
    }

# Backwards-compatible read-only view used by a few reporting callers.
CODEACT_LICENSES = {name: details["license"] for name, details in CODEACT_CONSTITUENTS.items()}

PLACEHOLDER_RE = re.compile(
    r"(?:\{\{\s*(?:todo|placeholder|fill[_ -]?me|insert[_ -]?here)[^{}]{0,60}\}\}|<\s*(?:todo|placeholder|fill[_ -]?me|insert[_ -]?here)\s*>|"
    r"\b(?:YOUR_API_KEY|REPLACE_ME|TBD_RESPONSE)\b)",
    re.IGNORECASE,
)
TRUNCATION_RE = re.compile(r"(?:\[\s*truncated\s*\]|<\s*truncated\s*>|…\s*\(truncated\))", re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", re.DOTALL)
SEARCH_REPLACE_RE = re.compile(
    r"###\s+(?P<path>[^\r\n]+)\s*\r?\n<<<<<<< SEARCH\r?\n(?P<search>.*?)\r?\n=======\r?\n(?P<replace>.*?)\r?\n>>>>>>> REPLACE",
    re.DOTALL,
)
PATH_RE = re.compile(r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
TEST_RE = re.compile(r"\b(?:pytest|unittest|npm\s+test|cargo\s+test|go\s+test|mvn\s+test|gradle\s+test|test[s]?)\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\b(?:fail(?:ed|ure)?|error|exception|traceback|non[- ]zero|wrong|incorrect)\b", re.IGNORECASE)
PASS_RE = re.compile(r"\b(?:pass(?:ed|ing)?|success(?:ful(?:ly)?)?|all tests)\b", re.IGNORECASE)
EDIT_RE = re.compile(r"\b(?:apply_patch|str_replace|write|edit|sed\s+-i|cat\s+>|patch)\b", re.IGNORECASE)
INSPECT_RE = re.compile(r"\b(?:rg|grep|find|ls|cat|sed\s+-n|read|open|git\s+(?:status|diff|show))\b", re.IGNORECASE)
TERMINAL_FAILURE_RE = re.compile(
    r"(?:\b(?:i|we)\s+(?:am\s+|are\s+)?(?:unable|not able)\b|"
    r"\b(?:i|we)\s+(?:cannot|can't|could not|couldn't)\s+(?:complete|answer|retrieve|find|access|perform|provide)\b|"
    r"\bfailed\s+to\s+(?:complete|answer|retrieve|find|access|perform|provide)\b|"
    r"\b(?:please\s+(?:provide|share|enter)|need\s+(?:your|an?))[^.\n]{0,60}\b(?:api[ _-]?key|credential|access token)\b)",
    re.IGNORECASE,
)
MEANINGFUL_TOKEN_RE = re.compile(r"[\w.+#:/-]+", re.UNICODE)
DEPENDENCY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get", "in", "is", "it", "of",
    "on", "or", "result", "search", "the", "this", "to", "tool", "true", "use", "with", "your",
}


class Tokenizer(Protocol):
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]: ...


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def normalized_pair(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompt = "\n".join(
        normalize_text(message["content"])
        for message in messages
        if message["role"] in {"system", "user"}
    )
    target = "\n".join(
        normalize_text(message["content"])
        for message in messages
        if message["role"] in {"assistant", "tool"}
    )
    return prompt, target


def token_shingles(value: str, width: int = SHINGLE_WIDTH) -> set[str]:
    tokens = re.findall(r"[\w.+#:/-]+", normalize_text(value), re.UNICODE)
    if not tokens:
        return set()
    if len(tokens) < width:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _python_signature(code: str) -> set[str]:
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        result = []
        for item in stream:
            if item.type in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}:
                continue
            if item.type == tokenize.NAME:
                result.append(item.string if keyword.iskeyword(item.string) else "NAME")
            elif item.type == tokenize.STRING:
                result.append("STRING")
            elif item.type == tokenize.NUMBER:
                result.append("NUMBER")
            else:
                result.append(item.string)
        return token_shingles(" ".join(result), width=4)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return set()


def code_signature(messages: list[dict[str, str]]) -> set[str]:
    blocks: list[str] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        blocks.extend(CODE_BLOCK_RE.findall(message["content"]))
        if "<execute>" in message["content"]:
            blocks.extend(re.findall(r"<execute>(.*?)</execute>", message["content"], re.DOTALL | re.IGNORECASE))
    signatures: set[str] = set()
    for block in blocks:
        signatures.update(_python_signature(block))
    return signatures


def validate_unicode(value: str) -> bool:
    try:
        encoded = value.encode("utf-8", "strict")
        return "\ufffd" not in value and encoded.decode("utf-8", "strict") == value
    except UnicodeError:
        return False


def boilerplate_heavy(value: str) -> bool:
    lines = [normalize_text(line) for line in value.splitlines() if normalize_text(line)]
    if len(lines) < 10:
        return False
    return 1 - len(set(lines)) / len(lines) >= 0.35


def global_content_rejection(messages: Any) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "malformed_messages"
    roles: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            return "malformed_messages"
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str) or not content.strip():
            return "malformed_messages"
        if not validate_unicode(content):
            return "invalid_unicode"
        if SENSITIVE_CREDENTIAL_RE.search(content):
            return "sensitive_credential_pattern"
        if PLACEHOLDER_RE.search(content):
            return "unresolved_placeholder"
        if TRUNCATION_RE.search(content):
            return "truncated_output"
        if boilerplate_heavy(content):
            return "boilerplate_heavy"
        roles.append(role)
    if "user" not in roles or "assistant" not in roles:
        return "malformed_roles"
    if roles[-1] != "assistant":
        return "malformed_final_role"
    final_assistant = next((item["content"] for item in reversed(messages) if item["role"] == "assistant"), "")
    if not final_assistant.strip():
        return "empty_output"
    prompt, target = normalized_pair(messages)
    if not target:
        return "empty_output"
    prompt_tokens, target_tokens = set(prompt.split()), set(target.split())
    if len(prompt_tokens) >= 20 and len(target_tokens) >= 20 and len(prompt_tokens & target_tokens) / max(1, len(target_tokens)) >= 0.90:
        return "prompt_copying"
    return None


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _schema_properties(tool: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    parameters = tool.get("parameters", {})
    if not isinstance(parameters, dict):
        return {}, set()
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        return parameters["properties"], set(parameters.get("required", []))
    return parameters, {name for name, spec in parameters.items() if isinstance(spec, dict) and "default" not in spec}


def _plausible_type(value: Any, declared: Any) -> bool:
    name = str(declared or "").casefold()
    if name in {"str", "string"}:
        return isinstance(value, str)
    if name in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if name in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name in {"bool", "boolean"}:
        return isinstance(value, bool)
    if name in {"array", "list"}:
        return isinstance(value, list)
    if name in {"object", "dict"}:
        return isinstance(value, dict)
    return True


def validate_calls(tools: Any, calls: Any) -> str | None:
    if not isinstance(tools, list) or not tools or not isinstance(calls, list) or not calls:
        return "malformed_tool_json"
    schemas: dict[str, dict[str, Any]] = {}
    for wrapper in tools:
        tool = wrapper.get("function", wrapper) if isinstance(wrapper, dict) else None
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            return "malformed_tool_schema"
        schemas[tool["name"]] = tool
    for call in calls:
        function = call.get("function", call) if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") not in schemas:
            return "unknown_tool"
        arguments = function.get("arguments", {})
        try:
            arguments = parse_json(arguments)
        except (TypeError, ValueError):
            return "malformed_tool_json"
        if not isinstance(arguments, dict):
            return "malformed_tool_arguments"
        properties, required = _schema_properties(schemas[function["name"]])
        if not required.issubset(arguments):
            return "missing_required_tool_argument"
        for name, value in arguments.items():
            spec = properties.get(name, {})
            if isinstance(spec, dict) and not _plausible_type(value, spec.get("type")):
                return "implausible_tool_argument_type"
    return None


def canonical_tool_signature(tools: list[dict[str, Any]]) -> str:
    return sha256_text(canonical_json(compact_tools(tools)))


def _normalize_parameter_schema(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        return parameters
    type_names = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, raw_spec in parameters.items():
        if not isinstance(raw_spec, dict):
            continue
        spec = dict(raw_spec)
        if spec.get("type") in type_names:
            spec["type"] = type_names[str(spec["type"])]
        properties[str(name)] = spec
        if "default" not in spec:
            required.append(str(name))
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = sorted(required)
    return result


def compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain the callable semantics needed to choose and populate each tool."""
    cleaned = []
    for wrapper in tools:
        tool = wrapper.get("function", wrapper)
        contract: dict[str, Any] = {"name": tool.get("name")}
        if isinstance(tool.get("description"), str) and tool["description"].strip():
            contract["description"] = tool["description"].strip()
        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            contract["parameters"] = _normalize_parameter_schema(parameters)
        cleaned.append(contract)
    return sorted(cleaned, key=lambda item: str(item["name"]))


def native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool contracts to the OpenAI shape consumed by gpt-oss."""
    return [{"type": "function", "function": tool} for tool in compact_tools(tools)]


def canonical_tool_contracts(tools: list[dict[str, Any]]) -> dict[str, str]:
    """Return a stable signature for each individual callable contract."""
    signatures: dict[str, str] = {}
    for contract in compact_tools(tools):
        name = str(contract.get("name") or "")
        if name:
            signatures[name] = sha256_text(canonical_json({
                "name": name,
                "parameters": contract.get("parameters", {"type": "object", "properties": {}}),
            }))
    return signatures


def native_tool_call(call: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize a bare or wrapped call without retaining hidden reasoning."""
    function = call.get("function", call)
    if not isinstance(function, dict):
        raise ValueError("Tool call function must be an object")
    arguments = function.get("arguments", {})
    try:
        arguments = parse_json(arguments)
    except (TypeError, ValueError):
        pass
    return {
        "id": str(call.get("id") or f"call_{index + 1}"),
        "type": "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": canonical_json(arguments) if not isinstance(arguments, str) else arguments,
        },
    }


def meaningful_tokens(value: Any) -> set[str]:
    """Conservative value tokens for tool dependency/use checks.

    JSON object keys describe a schema and are frequently repeated even when
    one result is not actually used by the next call.  Parse structured values
    and tokenize only their leaves so a shared key such as ``order_id`` cannot
    manufacture dependency evidence between unrelated values.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = value
        if parsed is not value:
            return meaningful_tokens(parsed)
    if isinstance(value, dict):
        result: set[str] = set()
        for nested in value.values():
            result.update(meaningful_tokens(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for nested in value:
            result.update(meaningful_tokens(nested))
        return result
    if value is None or isinstance(value, bool):
        return set()
    tokens = {
        cleaned
        for token in MEANINGFUL_TOKEN_RE.findall(normalize_text(value))
        if (cleaned := token.strip(".,;:!?()[]{}'\""))
        and cleaned not in DEPENDENCY_STOPWORDS
        and (len(cleaned) >= 3 or any(char.isdigit() for char in cleaned))
    }
    return tokens


def tool_result_failed(value: Any) -> bool:
    """Classify an actual tool outcome, avoiding false hits in successful data."""
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = None
    if isinstance(parsed, dict):
        success = parsed.get("success")
        if success is False:
            return True
        if success is True:
            return False
        status = normalize_text(parsed.get("status") or parsed.get("state") or "")
        if status in {"error", "failed", "failure", "cancelled", "canceled"}:
            return True
        error_value = parsed.get("error")
        if error_value is not None and error_value is not False and error_value != "" and error_value != []:
            return True
        for key in ("result", "response"):
            if isinstance(parsed.get(key), dict) and tool_result_failed(parsed[key]):
                return True
    text = normalize_text(value)
    return bool(
        re.match(r"^(?:error|exception|traceback|failed|failure|not found|request failed)\b", text)
        or re.search(r"\b(?:status|state)\s*[:=]\s*[\"']?(?:error|failed|failure)\b", text)
    )


def terminal_failure(value: str) -> bool:
    return bool(TERMINAL_FAILURE_RE.search(value))


def score_components(*, verification: float, difficulty: float, relevance: float, diversity: float) -> dict[str, float]:
    components = {
        "verification": round(max(0.0, min(1.0, verification)), 4),
        "difficulty": round(max(0.0, min(1.0, difficulty)), 4),
        "localpilot_relevance": round(max(0.0, min(1.0, relevance)), 4),
        "diversity": round(max(0.0, min(1.0, diversity)), 4),
    }
    components["weighted_total"] = round(
        40 * components["verification"]
        + 25 * components["difficulty"]
        + 20 * components["localpilot_relevance"]
        + 15 * components["diversity"],
        4,
    )
    return components


def xlam_trivial_one_call(call_count: int, tool_count: int) -> bool:
    return call_count == 1 and tool_count < 4


@dataclass
class Candidate:
    dataset: str
    original_id: str
    source_path: str
    messages: list[dict[str, str]]
    task_type: str
    difficulty: str
    family: str
    category: str
    subskill: str
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_signature: str | None = None
    repository: str | None = None
    raw_sha256: str | None = None
    license_override: str | None = None

    @property
    def rank_key(self) -> tuple[float, str]:
        return (-self.scores["weighted_total"], sha256_text(f"{SEED}:{self.dataset}:{self.original_id}"))

    @property
    def exact_key(self) -> str:
        return sha256_text("\n---TARGET---\n".join(normalized_pair(self.messages)))

    @property
    def near_text(self) -> str:
        # Injected system contracts can be much longer than the actual task and
        # would make unrelated xLAM/tool examples appear near-identical merely
        # because they share a large schema. Dataset-specific duplicate keys
        # already preserve canonical tool contracts where they are meaningful.
        return "\n".join(
            message["content"] for message in self.messages
            if message["role"] in {"user", "assistant", "tool"}
        )


class CandidatePool:
    """Bounded deterministic shortlist; overflow is a quality rejection."""

    def __init__(self, limits: dict[str, int]) -> None:
        self.limits = limits
        self.items: dict[str, list[Candidate]] = defaultdict(list)
        self.overflow = Counter()

    def add(self, bucket: str, candidate: Candidate) -> None:
        values = self.items[bucket]
        values.append(candidate)
        limit = self.limits.get(bucket, self.limits.get("*", 10000))
        if len(values) > limit * 2:
            values.sort(key=lambda item: item.rank_key)
            removed = len(values) - limit
            del values[limit:]
            self.overflow["not_shortlisted_quality"] += removed

    def finish(self) -> dict[str, list[Candidate]]:
        for bucket, values in self.items.items():
            limit = self.limits.get(bucket, self.limits.get("*", 10000))
            values.sort(key=lambda item: item.rank_key)
            removed = max(0, len(values) - limit)
            if removed:
                self.overflow["not_shortlisted_quality"] += removed
                del values[limit:]
        return dict(self.items)


class Deduplicator:
    def __init__(self, *, include_dataset_keys: bool = False) -> None:
        self.include_dataset_keys = include_dataset_keys
        self.exact: dict[str, str] = {}
        self.dataset_keys: dict[str, str] = {}
        self.family_skills: dict[str, dict[str, str]] = defaultdict(dict)
        self.near_signatures: list[set[str]] = []
        self.near_representatives: list[str] = []
        self.near_inverted: dict[str, list[int]] = defaultdict(list)
        self.code_signatures: list[set[str]] = []
        self.code_representatives: list[str] = []
        self.code_inverted: dict[str, list[int]] = defaultdict(list)

    @staticmethod
    def _representative(candidate: Candidate) -> str:
        return f"{candidate.dataset}:{candidate.exact_key}"

    @staticmethod
    def _principal_skill(candidate: Candidate) -> str:
        # The SWE collection genuinely offers distinct repair, test generation,
        # localization and agentic subskills for one issue.  Other dataset
        # strata (for example evol/self-instruct) are provenance variants, not
        # a justification for duplicating the underlying task.
        return candidate.subskill if candidate.dataset == "nemotron_swe_v2" else candidate.task_type

    @staticmethod
    def _problem_family(candidate: Candidate) -> str:
        value = candidate.metadata.get("problem_identity_sha256")
        return str(value) if value else candidate.family

    def match(self, candidate: Candidate) -> tuple[str | None, str | None]:
        if candidate.exact_key in self.exact:
            return "exact_duplicate", self.exact[candidate.exact_key]
        if self.include_dataset_keys:
            dataset_key = candidate.metadata.get("dataset_duplicate_key_sha256")
            scoped_key = f"{candidate.dataset}:{dataset_key}" if dataset_key else None
            if scoped_key and scoped_key in self.dataset_keys:
                return "dataset_key_duplicate", self.dataset_keys[scoped_key]
            problem_family = self._problem_family(candidate)
            prior_skills = self.family_skills[problem_family]
            principal_skill = self._principal_skill(candidate)
            if principal_skill in prior_skills:
                return "same_family_subskill_duplicate", prior_skills[principal_skill]
            if len(prior_skills) >= 2:
                representative = sorted(prior_skills.values())[0]
                return "same_family_example_cap", representative
        shingles = token_shingles(candidate.near_text)
        possible_near: set[int] = set()
        for shingle in shingles:
            postings = self.near_inverted.get(shingle, [])
            # Very common boilerplate shingles carry little retrieval value and
            # can otherwise make a global pass quadratic.
            if len(postings) <= 5000:
                possible_near.update(postings)
        for index in sorted(possible_near):
            prior = self.near_signatures[index]
            if min(len(shingles), len(prior)) / max(len(shingles), len(prior), 1) < NEAR_DUPLICATE_THRESHOLD:
                continue
            if jaccard(shingles, prior) >= NEAR_DUPLICATE_THRESHOLD:
                return "near_duplicate", self.near_representatives[index]
        signature = code_signature(candidate.messages)
        if signature:
            possible: set[int] = set()
            for shingle in signature:
                postings = self.code_inverted.get(shingle, [])
                if len(postings) <= 5000:
                    possible.update(postings)
            for index in sorted(possible):
                prior = self.code_signatures[index]
                if min(len(signature), len(prior)) / max(len(signature), len(prior)) < CODE_DUPLICATE_THRESHOLD:
                    continue
                if jaccard(signature, prior) >= CODE_DUPLICATE_THRESHOLD:
                    return "code_duplicate", self.code_representatives[index]
        return None, None

    def reason(self, candidate: Candidate) -> str | None:
        return self.match(candidate)[0]

    def add(self, candidate: Candidate) -> None:
        representative = self._representative(candidate)
        self.exact[candidate.exact_key] = representative
        if self.include_dataset_keys:
            dataset_key = candidate.metadata.get("dataset_duplicate_key_sha256")
            if dataset_key:
                self.dataset_keys[f"{candidate.dataset}:{dataset_key}"] = representative
            self.family_skills[self._problem_family(candidate)][self._principal_skill(candidate)] = representative
        shingles = token_shingles(candidate.near_text)
        if shingles:
            near_index = len(self.near_signatures)
            self.near_signatures.append(shingles)
            self.near_representatives.append(representative)
            for shingle in shingles:
                self.near_inverted[shingle].append(near_index)
        signature = code_signature(candidate.messages)
        if signature:
            index = len(self.code_signatures)
            self.code_signatures.append(signature)
            self.code_representatives.append(representative)
            for shingle in signature:
                self.code_inverted[shingle].append(index)


def token_length(tokenizer: Tokenizer, messages: list[dict[str, str]]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def candidate_record(candidate: Candidate, split: str, token_count: int) -> dict[str, Any]:
    spec = DATASETS[candidate.dataset]
    license_path = f"{spec.get('source_subdirectory', '').strip('/') + '/' if spec.get('source_subdirectory') else ''}README.md"
    identifier = f"ext-v1-{candidate.dataset.replace('_', '-')}-{sha256_text(candidate.original_id)[:16]}"
    provenance: dict[str, Any] = {
        "reference": f"hf:{spec['repo_id']}@{spec['revision']}#{candidate.original_id}",
        "repository": spec["repo_id"],
        "url": f"https://huggingface.co/datasets/{spec['repo_id']}/tree/{spec['revision']}",
        "license_reference": f"https://huggingface.co/datasets/{spec['repo_id']}/blob/{spec['revision']}/{license_path}",
        "dataset": candidate.dataset,
        "revision": spec["revision"],
        "original_id": candidate.original_id,
        "source_path": candidate.source_path,
        "transformation_version": TRANSFORMATION_VERSION,
        "acquisition_date": ACQUISITION_DATE,
    }
    if candidate.repository:
        provenance["source_repository"] = candidate.repository
    constituent = candidate.metadata.get("constituent_provenance")
    if isinstance(constituent, dict):
        provenance["constituent"] = constituent
    return {
        "id": identifier,
        "messages": candidate.messages,
        "task_type": candidate.task_type,
        "source": spec["repo_id"],
        "license": candidate.license_override or spec["license"],
        "quality_tier": "B",
        "split": split,
        "verification_status": "source_verified",
        "provenance": provenance,
        "tags": sorted({"external_corpus_v1", candidate.dataset, candidate.category, candidate.subskill}),
        "difficulty": candidate.difficulty,
        "created_at": f"{ACQUISITION_DATE}T00:00:00Z",
        "metadata": {
            **candidate.metadata,
            "family_id": candidate.family,
            "selection_score": candidate.scores,
            "token_count": token_count,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        },
    }


def source_record(record: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    provenance = record["provenance"]
    result = {
        "id": record["id"],
        "dataset": candidate.dataset,
        "dataset_name": DATASETS[candidate.dataset]["repo_id"],
        "pinned_revision": DATASETS[candidate.dataset]["revision"],
        "original_id": candidate.original_id,
        "license": candidate.license_override or DATASETS[candidate.dataset]["license"],
        "source_path": candidate.source_path,
        "transformation_version": TRANSFORMATION_VERSION,
        "acquisition_date": ACQUISITION_DATE,
        "family_id": candidate.family,
        "raw_record_sha256": candidate.raw_sha256,
        "reference": provenance["reference"],
        "license_reference": provenance["license_reference"],
    }
    constituent = candidate.metadata.get("constituent_provenance")
    if isinstance(constituent, dict):
        result["constituent_provenance"] = constituent
    return result


def grouped_split(candidates: Iterable[Candidate], validation_ratio: float = 0.10) -> dict[str, str]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        group = str(candidate.metadata.get("problem_identity_sha256") or candidate.family)
        groups[group].append(candidate)
    ordered = sorted(groups, key=lambda family: sha256_text(f"{SEED}:split:{family}"))
    target = round(sum(len(values) for values in groups.values()) * validation_ratio)
    selected: set[str] = set()
    current = 0
    for family in ordered:
        size = len(groups[family])
        if current < target and current + size <= target:
            selected.add(family)
            current += size
    remaining = sorted(
        (family for family in ordered if family not in selected),
        key=lambda family: (abs((current + len(groups[family])) - target), sha256_text(family)),
    )
    if remaining and abs((current + len(groups[remaining[0]])) - target) < abs(current - target):
        selected.add(remaining[0])
    result: dict[str, str] = {}
    for group, values in groups.items():
        split = "validation" if group in selected else "train"
        for candidate in values:
            existing = result.get(candidate.family)
            if existing is not None and existing != split:
                raise ValueError(f"Family {candidate.family} maps to multiple problem identities")
            result[candidate.family] = split
    return result


def jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield line_number, value, sha256_text(raw.rstrip("\r\n"))


def messages_without_hidden_reasoning(messages: Any, tools: Any = None) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    output: list[dict[str, str]] = []
    if isinstance(tools, list) and tools:
        output.append({"role": "system", "content": "Available tools:\n" + canonical_json(compact_tools(tools))})
    for item in messages:
        if not isinstance(item, dict):
            return []
        role = item.get("role")
        content = item.get("content") or ""
        calls = item.get("tool_calls")
        if role == "system" and (not str(content).strip() or (isinstance(tools, list) and tools)):
            continue
        if role == "assistant" and calls:
            call_payload = []
            for call in calls:
                function = call.get("function", call) if isinstance(call, dict) else call
                if isinstance(function, dict):
                    try:
                        arguments = parse_json(function.get("arguments", {}))
                    except (TypeError, ValueError):
                        arguments = function.get("arguments")
                    call_payload.append({"name": function.get("name"), "arguments": arguments})
            # Upstream assistant content beside a call may contain private
            # scratch reasoning.  The callable action is the supervised target;
            # keep only its canonical, schema-safe representation here.
            content = "<tool_calls>" + canonical_json(call_payload) + "</tool_calls>"
        if role in {"system", "user", "assistant", "tool"} and str(content).strip():
            # The pinned gpt-oss Harmony template does not accept a bare generic
            # `tool` role. Preserve the observation as an explicit user-side
            # environment message after validating the original call ordering.
            if role == "tool":
                name = str(item.get("name") or "tool")
                mapped = {"role": "user", "content": f"Tool result ({name}):\n{str(content).strip()}"}
            else:
                mapped = {"role": role, "content": str(content).strip()}
            output.append(mapped)
    return output


def classify_agentless(messages: list[dict[str, str]]) -> str | None:
    prompt = next((item["content"] for item in messages if item["role"] == "user"), "").casefold()
    target = next((item["content"] for item in reversed(messages) if item["role"] == "assistant"), "").casefold()
    patch_like = "diff --git" in target or "*** begin patch" in target or ("<<<<<<< search" in target and ">>>>>>> replace" in target)
    if patch_like and (
        re.search(r"(?:^|/)(?:tests?|test_[^/]*)[/_.]", target) or re.search(r"\b(?:def test_|assert|expect\()", target)
    ):
        return "test_generation"
    if patch_like:
        return "repair"
    if "list of files" in prompt or "localiz" in prompt or "ranked file" in prompt:
        return "localization"
    if "test generation" in prompt or "reproduction test" in prompt or "write a test" in prompt or "unit test" in prompt:
        return "test_generation"
    if "patch" in prompt or "repair" in prompt or "fix" in prompt:
        return "repair"
    return None


def verify_agentless(subskill: str, messages: list[dict[str, str]]) -> bool:
    raw_prompt = next((item["content"] for item in messages if item["role"] == "user"), "")
    raw_target = next((item["content"] for item in reversed(messages) if item["role"] == "assistant"), "")
    prompt, target = normalize_text(raw_prompt), normalize_text(raw_target)
    def clean_path(path: str) -> str:
        return path.removeprefix("a/").removeprefix("b/").rstrip(".,:;)]}>`'\"")

    prompt_paths = {clean_path(path) for path in PATH_RE.findall(prompt)}
    target_paths = {clean_path(path) for path in PATH_RE.findall(target)}
    search_replace_blocks = list(SEARCH_REPLACE_RE.finditer(raw_target))
    search_replace_valid = bool(search_replace_blocks) and all(
        normalize_text(match.group("search"))
        and normalize_text(match.group("search")) in prompt
        and normalize_text(match.group("search")) != normalize_text(match.group("replace"))
        and clean_path(match.group("path").strip()) in prompt_paths
        for match in search_replace_blocks
    )
    if subskill == "localization":
        # The released Agentless rows contain a repository tree but no changed-
        # file ground truth. Membership in that tree cannot verify relevance.
        return False
    if subskill == "repair":
        changed_lines = [line for line in raw_target.splitlines() if (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---"))]
        removed_lines = [normalize_text(line[1:]) for line in raw_target.splitlines() if line.startswith("-") and not line.startswith("---") and normalize_text(line[1:])]
        unified_diff_valid = (
            ("diff --git" in target or ("*** begin patch" in target and "*** end patch" in target))
            and "@@" in target and bool(changed_lines) and bool(removed_lines)
            and all(line in prompt for line in removed_lines)
            and bool(target_paths) and bool(target_paths & prompt_paths)
        )
        return unified_diff_valid or search_replace_valid
    if subskill == "test_generation":
        changed_lines = [line for line in raw_target.splitlines() if line.startswith("+") and not line.startswith("+++")]
        unified_diff_valid = (
            ("diff --git" in target or ("*** begin patch" in target and "*** end patch" in target))
            and "@@" in target and bool(changed_lines) and bool(target_paths & prompt_paths)
        )
        return (unified_diff_valid or search_replace_valid) and bool(TEST_RE.search(target)) and ("assert" in target or "expect(" in target or "should" in target)
    return False


def openhands_traits(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", [])
    actions: list[str] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", call) if isinstance(call, dict) else {}
            # ``think`` is provider scratch space and ``finish`` is the
            # OpenHands envelope for the final response.  Neither is a
            # meaningful repository action for the >=3-action quality gate.
            name = str(function.get("name") or "").casefold() if isinstance(function, dict) else ""
            if isinstance(function, dict) and name not in {"think", "finish"}:
                actions.append(str(function.get("arguments") or ""))
    normalized_actions = [normalize_text(action) for action in actions]
    first_edit = next((index for index, action in enumerate(actions) if EDIT_RE.search(action)), None)
    first_inspect = next((index for index, action in enumerate(actions) if INSPECT_RE.search(action)), None)
    last_edit = max((index for index, action in enumerate(actions) if EDIT_RE.search(action)), default=-1)
    later_validation = any(TEST_RE.search(action) for action in actions[last_edit + 1 :])
    failed_then_changed_then_passed = False
    text_messages = []
    for item in messages if isinstance(messages, list) else []:
        if not isinstance(item, dict):
            continue
        event = [str(item.get("content") or "")]
        for call in item.get("tool_calls") or []:
            function = call.get("function", call) if isinstance(call, dict) else {}
            name = str(function.get("name") or "").casefold() if isinstance(function, dict) else ""
            if isinstance(function, dict) and name not in {"think", "finish"}:
                event.append(str(function.get("arguments") or ""))
        text_messages.append("\n".join(event))
    failure_positions = [index for index, value in enumerate(text_messages) if FAIL_RE.search(value)]
    pass_positions = [index for index, value in enumerate(text_messages) if PASS_RE.search(value)]
    edit_positions = [index for index, value in enumerate(text_messages) if EDIT_RE.search(value)]
    if failure_positions and pass_positions and edit_positions:
        failed_then_changed_then_passed = any(fail < edit < passed for fail in failure_positions for edit in edit_positions for passed in pass_positions)
    return {
        "action_count": len(actions),
        "repeated_identical_action": any(left == right for left, right in zip(normalized_actions, normalized_actions[1:])),
        "inspect_before_edit": first_edit is not None and first_inspect is not None and first_inspect < first_edit,
        "validation_after_edit": later_validation,
        "failed_changed_passed": failed_then_changed_then_passed,
    }


def openhands_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str | None]:
    """Convert an OpenHands trace into a validated native gpt-oss trajectory.

    OpenHands stores the user-facing final response as an unresolved ``finish``
    tool call and may include ``think`` calls containing provider scratch
    reasoning.  gpt-oss requires each retained call to have one matching result,
    so internal ``think`` pairs are removed and terminal ``finish(message=...)``
    is converted back to a normal assistant response.  Repository calls and
    their observations remain structured in ``native_messages``.
    """
    tools, source_messages = row.get("tools"), row.get("messages")
    if not isinstance(tools, list) or not tools or not isinstance(source_messages, list) or not source_messages:
        return [], {}, "malformed_tool_schema"

    tool_names: list[str] = []
    operational_tools: list[dict[str, Any]] = []
    for wrapper in tools:
        tool = wrapper.get("function", wrapper) if isinstance(wrapper, dict) else None
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"].strip():
            return [], {}, "malformed_tool_schema"
        name = tool["name"].strip()
        if name in tool_names:
            return [], {}, "duplicate_tool_schema"
        tool_names.append(name)
        if name.casefold() not in {"think", "finish"}:
            operational_tools.append(wrapper)
    if not operational_tools:
        return [], {}, "missing_repository_tool"

    native_messages: list[dict[str, Any]] = []
    outstanding: dict[str, Any] | None = None
    seen_call_ids: set[str] = set()
    previous_operational_call: str | None = None
    repository_call_count = 0
    removed_think_calls = 0
    finish_converted = False
    final_answer = ""
    call_index = 0

    for message_index, message in enumerate(source_messages):
        if not isinstance(message, dict):
            return [], {}, "malformed_messages"
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        raw_calls = message.get("tool_calls")
        if outstanding is not None and role != "tool":
            return [], {}, "invalid_tool_result_order"

        if role == "assistant" and raw_calls:
            if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                return [], {}, "multiple_tool_calls_in_assistant_message"
            reason = validate_calls(tools, raw_calls)
            if reason:
                return [], {}, reason
            normalized_call = native_tool_call(raw_calls[0], call_index)
            call_index += 1
            function = normalized_call["function"]
            name = function["name"]
            call_id = normalized_call["id"]
            if call_id in seen_call_ids:
                return [], {}, "duplicate_tool_call_id"
            seen_call_ids.add(call_id)

            if name.casefold() == "finish":
                if message_index != len(source_messages) - 1 or finish_converted:
                    return [], {}, "invalid_finish_order"
                try:
                    arguments = parse_json(function["arguments"])
                except (TypeError, ValueError):
                    return [], {}, "malformed_finish_arguments"
                final_message = arguments.get("message") if isinstance(arguments, dict) else None
                if not isinstance(final_message, str) or not final_message.strip():
                    return [], {}, "empty_finish_message"
                final_answer = final_message.strip()
                native_messages.append({"role": "assistant", "content": final_answer})
                finish_converted = True
                continue

            call_key = canonical_json({"name": name, "arguments": parse_json(function["arguments"])})
            internal = name.casefold() == "think"
            if not internal:
                if call_key == previous_operational_call:
                    return [], {}, "repeated_identical_tool_call"
                previous_operational_call = call_key
                repository_call_count += 1
                # Adjacent assistant content in the source is provider scratch
                # reasoning; only the executable action is a training target.
                native_messages.append({"role": "assistant", "content": "", "tool_calls": [normalized_call]})
            else:
                removed_think_calls += 1
            outstanding = {"id": call_id, "name": name, "internal": internal}
        elif role == "tool":
            if outstanding is None:
                return [], {}, "invalid_tool_result_order"
            call_id = str(message.get("tool_call_id") or "")
            if call_id != outstanding["id"]:
                return [], {}, "invalid_tool_result_order"
            result_name = message.get("name")
            if result_name is not None and str(result_name) != outstanding["name"]:
                return [], {}, "invalid_tool_result_name"
            if not content:
                return [], {}, "unusable_tool_result"
            if not outstanding["internal"]:
                native_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": call_id,
                    "name": outstanding["name"],
                })
            outstanding = None
        elif role == "system":
            # Tool contracts are passed through the tokenizer's native ``tools``
            # argument.  The upstream provider/system scaffold is not a target.
            continue
        elif role == "user":
            if not content:
                return [], {}, "malformed_messages"
            native_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            # A content-only assistant message is valid only as the terminal
            # response.  Intermediate prose in this source is scratch planning.
            if message_index != len(source_messages) - 1 or not content:
                return [], {}, "unstructured_intermediate_assistant"
            final_answer = content
            native_messages.append({"role": "assistant", "content": content})
        else:
            return [], {}, "malformed_messages"

    if outstanding is not None:
        return [], {}, "missing_tool_result"
    if repository_call_count < 1:
        return [], {}, "missing_repository_tool_call"
    if not final_answer or not native_messages or native_messages[-1].get("role") != "assistant":
        return [], {}, "missing_final_answer"

    schema_messages = messages_without_hidden_reasoning(native_messages, operational_tools)
    traits = openhands_traits(row)
    traits.update({
        "native_tool_call_count": repository_call_count,
        "internal_think_call_count_removed": removed_think_calls,
        "finish_converted_to_final_answer": finish_converted,
        "native_tools": native_tools(operational_tools),
        "native_messages": native_messages,
    })
    return schema_messages, traits, None


def tool_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str | None]:
    """Validate a native tool trajectory and return its schema-safe rendering.

    A failure is only considered recovered after a *different* later call has
    observed the failed result and itself receives a successful result.  Tool
    outputs must also contribute evidence to a later call or the final answer;
    this rejects unused/unnecessary calls rather than trusting dataset labels.
    """
    tools = row.get("tools")
    messages = row.get("messages")
    if not isinstance(tools, list) or not isinstance(messages, list):
        return [], {}, "malformed_tool_schema"

    outstanding: dict[str, int] = {}
    seen_call_ids: set[str] = set()
    call_keys: set[str] = set()
    calls: list[dict[str, Any]] = []
    native_messages: list[dict[str, Any]] = []
    final_answer = ""
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            return [], {}, "malformed_messages"
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if outstanding and role != "tool":
            return [], {}, "invalid_tool_result_order"
        if role == "assistant" and message.get("tool_calls"):
            raw_calls = message["tool_calls"]
            if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                return [], {}, "multiple_tool_calls_in_assistant_message"
            if outstanding:
                return [], {}, "invalid_tool_result_order"
            reason = validate_calls(tools, raw_calls)
            if reason:
                return [], {}, reason
            raw_call = raw_calls[0]
            normalized_call = native_tool_call(raw_call, len(calls))
            function = normalized_call["function"]
            call_id = normalized_call["id"]
            arguments = parse_json(function["arguments"])
            call_key = canonical_json({"name": function["name"], "arguments": arguments})
            if call_key in call_keys:
                return [], {}, "repeated_identical_tool_call"
            if call_id in seen_call_ids:
                return [], {}, "duplicate_tool_call_id"
            call_keys.add(call_key)
            seen_call_ids.add(call_id)
            calls.append({
                "id": call_id,
                "name": function["name"],
                "key": call_key,
                "argument_tokens": meaningful_tokens(arguments),
                "call_message_index": message_index,
                "result_message_index": None,
                "result": "",
                "result_tokens": set(),
                "failed": None,
            })
            outstanding[call_id] = len(calls) - 1
            native_messages.append({
                # Do not retain any provider scratch reasoning adjacent to the
                # structured call.  It is neither required for execution nor a
                # permissible LocalPilot training target.
                "role": "assistant", "content": "",
                "tool_calls": [normalized_call],
            })
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in outstanding:
                return [], {}, "invalid_tool_result_order"
            call_index = outstanding.pop(call_id)
            call = calls[call_index]
            result_name = str(message.get("name") or "").strip()
            if result_name and normalize_text(result_name) != normalize_text(call["name"]):
                return [], {}, "invalid_tool_result_name"
            call["result_message_index"] = message_index
            call["result"] = content
            call["result_tokens"] = meaningful_tokens(content)
            call["failed"] = tool_result_failed(content)
            native_messages.append({
                "role": "tool", "content": content, "tool_call_id": call_id, "name": call["name"],
            })
        elif role in {"system", "user", "assistant"}:
            if role == "assistant" and content:
                final_answer = content
            if content:
                native_messages.append({"role": role, "content": content})
        else:
            return [], {}, "malformed_messages"
    if outstanding:
        return [], {}, "missing_tool_result"
    if not calls:
        return [], {}, "missing_tool_call"

    converted = messages_without_hidden_reasoning(messages, tools)
    if not converted or converted[-1]["role"] != "assistant" or converted[-1]["content"].startswith("<tool_calls>"):
        return [], {}, "missing_final_answer"
    if terminal_failure(final_answer):
        return [], {}, "terminal_failure"

    recovered_failures: set[int] = set()
    dependency_links: set[tuple[int, int]] = set()
    used_results: set[int] = set()
    final_tokens = meaningful_tokens(final_answer)
    for index, call in enumerate(calls):
        result_tokens = call["result_tokens"]
        if call["failed"]:
            recovery = next((
                later_index
                for later_index, later in enumerate(calls[index + 1 :], index + 1)
                if later["call_message_index"] > call["result_message_index"]
                and later["key"] != call["key"] and later["failed"] is False
            ), None)
            if recovery is None:
                return [], {}, "unrecovered_tool_error"
            recovered_failures.add(index)
            dependency_links.add((index, recovery))
            continue
        if not result_tokens:
            return [], {}, "unusable_tool_result"
        later_tokens = set(final_tokens)
        for later_index, later in enumerate(calls[index + 1 :], index + 1):
            later_tokens.update(later["argument_tokens"])
            if result_tokens & later["argument_tokens"]:
                dependency_links.add((index, later_index))
        if result_tokens & later_tokens:
            used_results.add(index)
        else:
            return [], {}, "unused_tool_result"

    if len(calls) > 1 and not dependency_links:
        return [], {}, "independent_tool_sequence"
    for index, call in enumerate(calls[1:], 1):
        if re.search(r"(?:search|find|lookup|query)", call["name"], re.IGNORECASE):
            prior_same = [prior for prior in range(index) if calls[prior]["name"] == call["name"]]
            if prior_same and not any((prior, index) in dependency_links for prior in prior_same):
                return [], {}, "repeated_search_without_dependency"

    names = [str(call["name"]) for call in calls]
    return converted, {
        "tool_call_count": len(names),
        "distinct_tool_count": len(set(names)),
        "ordered_tool_names": names,
        "ordered_tool_call_keys": [str(call["key"]) for call in calls],
        "error_recovered": bool(recovered_failures),
        "recovered_error_count": len(recovered_failures),
        "dependency_link_count": len(dependency_links),
        "used_tool_result_count": len(used_results),
        "native_tools": native_tools(tools),
        "native_messages": native_messages,
    }, None


def validate_openhands_sequence(row: dict[str, Any]) -> str | None:
    tools, messages = row.get("tools"), row.get("messages")
    if not isinstance(tools, list) or not isinstance(messages, list):
        return "malformed_tool_schema"
    outstanding: set[str] = set()
    previous_call: str | None = None
    generated_id = 0
    for message in messages:
        if not isinstance(message, dict):
            return "malformed_messages"
        if message.get("role") == "assistant" and message.get("tool_calls"):
            reason = validate_calls(tools, message["tool_calls"])
            if reason:
                return reason
            for call in message["tool_calls"]:
                function = call.get("function", call)
                try:
                    arguments = parse_json(function.get("arguments", {}))
                except (TypeError, ValueError):
                    arguments = function.get("arguments")
                call_key = canonical_json({"name": function.get("name"), "arguments": arguments})
                if call_key == previous_call:
                    return "repeated_identical_tool_call"
                previous_call = call_key
                call_id = str(call.get("id") or f"generated-{generated_id}")
                generated_id += 1
                if call_id in outstanding:
                    return "duplicate_tool_call_id"
                if str(function.get("name") or "").casefold() != "finish":
                    outstanding.add(call_id)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in outstanding:
                return "invalid_tool_result_order"
            outstanding.remove(call_id)
    if outstanding:
        return "missing_tool_result"
    return None


def codeact_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str | None]:
    content = row.get("content")
    if not isinstance(content, list) or not content:
        return [], {}, "malformed_messages"
    messages: list[dict[str, str]] = []
    actions: list[str] = []
    pairs: list[tuple[str, str]] = []
    pending_action: str | None = None
    pending_call_id: str | None = None
    native_messages: list[dict[str, Any]] = []
    execute_tools = native_tools([{
        "name": "execute",
        "description": "Execute Python code in the task environment and return its observation.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    }])
    for index, item in enumerate(content):
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            return [], {}, "malformed_messages"
        kind, text = item.get("class_"), item["content"].strip()
        if not text:
            return [], {}, "empty_output"
        if index == 0 and kind != "text_observation":
            return [], {}, "malformed_roles"
        if kind == "text_observation":
            role = "user" if index == 0 else "tool"
            if role == "tool":
                messages.append({"role": "user", "content": f"Execution observation:\n{text}"})
                if pending_call_id is None:
                    return [], {}, "invalid_tool_result_order"
                native_messages.append({
                    "role": "tool", "content": text, "tool_call_id": pending_call_id, "name": "execute",
                })
            else:
                messages.append({"role": role, "content": text})
                native_messages.append({"role": "user", "content": text})
            if index and pending_action is not None:
                if len(normalize_text(text).split()) < 2:
                    return [], {}, "no_execution_evidence"
                pairs.append((pending_action, text))
                pending_action = None
                pending_call_id = None
        elif kind == "code_action":
            if pending_action is not None:
                return [], {}, "unobserved_action_loop"
            pending_action = normalize_text(text)
            actions.append(pending_action)
            messages.append({"role": "assistant", "content": f"<execute>\n{text}\n</execute>"})
            pending_call_id = f"codeact_{len(actions)}"
            native_messages.append({
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": pending_call_id, "type": "function",
                    "function": {"name": "execute", "arguments": canonical_json({"code": text})},
                }],
            })
        elif kind == "message_action":
            messages.append({"role": "assistant", "content": text})
            embedded = re.search(r"<execute>(.*?)</execute>", text, re.IGNORECASE | re.DOTALL)
            if embedded:
                if pending_action is not None:
                    return [], {}, "unobserved_action_loop"
                code = embedded.group(1).strip()
                pending_action = normalize_text(code)
                actions.append(pending_action)
                pending_call_id = f"codeact_{len(actions)}"
                native_messages.append({
                    "role": "assistant", "content": "", "tool_calls": [{
                        "id": pending_call_id, "type": "function",
                        "function": {"name": "execute", "arguments": canonical_json({"code": code})},
                    }],
                })
            else:
                native_messages.append({"role": "assistant", "content": text})
        else:
            return [], {}, "malformed_roles"
    if len(actions) > 8:
        return [], {}, "too_many_action_cycles"
    if len(actions) != len(set(actions)):
        return [], {}, "repeated_identical_action"
    if pending_action is not None or not pairs:
        return [], {}, "no_execution_evidence"
    if messages[-1]["role"] != "assistant" or FAIL_RE.search(messages[-1]["content"]):
        return [], {}, "terminal_failure"
    error_positions = [index for index, (_, observation) in enumerate(pairs) if FAIL_RE.search(observation)]
    successful_positions = [index for index, (_, observation) in enumerate(pairs) if not FAIL_RE.search(observation)]
    error_recovered = any(
        failure < success and pairs[failure][0] != pairs[success][0]
        for failure in error_positions for success in successful_positions
    )
    if error_positions and not error_recovered:
        return [], {}, "terminal_failure"
    return messages, {
        "action_observation_cycles": len(pairs),
        "execution_error": bool(error_positions),
        "error_recovered": error_recovered,
        "successful_result": bool(successful_positions) and successful_positions[-1] == len(pairs) - 1,
        "native_tools": execute_tools,
        "native_messages": native_messages,
    }, None


def opencode_judgement(value: Any, test_count: int) -> tuple[float, str | None]:
    try:
        parsed = parse_json(value)
    except (TypeError, ValueError):
        return (1.0, None) if test_count >= 5 else (0.0, "unparseable_judgement")
    if not isinstance(parsed, dict) or not parsed:
        return 0.0, "unparseable_judgement"
    scores = []
    for dimension in parsed.values():
        if not isinstance(dimension, dict) or not isinstance(dimension.get("score"), (int, float)):
            return 0.0, "unparseable_judgement"
        scores.append(float(dimension["score"]))
    if min(scores, default=0) < 4:
        return 0.0, "llm_judgement_concern"
    return min(1.0, sum(scores) / (5 * len(scores))), None


def swecare_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str | None]:
    review = row.get("commit_to_review")
    comments = row.get("reference_review_comments")
    if not isinstance(review, dict) or not isinstance(comments, list) or not comments:
        return [], {}, "missing_review_evidence"
    patch = review.get("patch_to_review")
    if not isinstance(patch, str) or not patch.strip() or not ("diff --git" in patch or "@@" in patch):
        return [], {}, "missing_patch"
    merged_patch = row.get("merged_patch")
    if not isinstance(merged_patch, str) or not merged_patch.strip():
        return [], {}, "missing_verifier_patch"
    valid_comments = []
    reviewed_excerpts = []
    for comment in comments:
        if not isinstance(comment, dict) or not str(comment.get("text") or "").strip() or not str(comment.get("path") or "").strip():
            return [], {}, "invalid_review_location"
        line = comment.get("line") or comment.get("original_line") or comment.get("start_line") or comment.get("original_start_line")
        if not isinstance(line, int) or line < 1:
            return [], {}, "invalid_review_location"
        diff_hunk = comment.get("diff_hunk")
        if str(comment["path"]) not in patch or str(comment["path"]) not in merged_patch:
            return [], {}, "invalid_review_location"
        hunk_lines = diff_hunk.splitlines() if isinstance(diff_hunk, str) else []
        if not isinstance(diff_hunk, str) or "@@" not in diff_hunk or not any(
            line and not line.startswith("@@") for line in hunk_lines
        ):
            return [], {}, "insufficient_review_context"
        valid_comments.append({"path": comment["path"], "line": line, "comment": comment["text"]})
        reviewed_excerpts.append(
            f"File: {comment['path']}\nLine: {line}\n{diff_hunk.strip()}"
        )
    prompt = (
        "Review the following proposed change for correctness. Return only actionable review comments with file and line locations.\n\n"
        f"Title: {row.get('title') or ''}\n"
        f"Problem:\n{row.get('problem_statement') or row.get('body') or ''}\n\n"
        "Reviewed patch excerpts:\n" + "\n\n".join(reviewed_excerpts)
    )
    target = canonical_json(valid_comments)
    text = normalize_text(" ".join(item["comment"] for item in valid_comments))
    style_only = bool(re.search(r"\b(?:style|format(?:ting)?|naming|typo|whitespace)\b", text)) and not bool(
        re.search(r"\b(?:bug|incorrect|fail|regression|unsafe|security|race|performance|compatib|state|test)\b", text)
    )
    if style_only:
        return [], {}, "style_only_review"
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}], {
        "human_review_comment_count": len(valid_comments),
        "merged_patch_used_as_verifier_only": True,
    }, None


def raw_record_digest(row: dict[str, Any]) -> str:
    return sha256_text(canonical_json(row))


def simple_difficulty(value: float) -> str:
    if value >= 0.85:
        return "expert"
    if value >= 0.60:
        return "hard"
    if value >= 0.35:
        return "medium"
    return "easy"
