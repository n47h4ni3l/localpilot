from __future__ import annotations

import ast
import hashlib
import ipaddress
import json
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from localpilot.learning import (
    CurriculumStageState,
    LearningMemory,
    PeerModelComparison,
    StudyRun,
)
from localpilot.process import hidden_process_creation_flags
from localpilot.tools.web import _SafeRedirectHandler, _ValidatedHTTPSHandler


STAGES = ("self", "qwen", "python")
BENCHMARK_VERSION = "1"
_IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "localpilot-data",
}
_OFFICIAL_HOSTS = {
    "docs.ollama.com",
    "ollama.com",
    "qwen.readthedocs.io",
    "qwenlm.github.io",
    "docs.python.org",
    "docs.pytest.org",
}
_MAX_WEB_SOURCE_BYTES = 2_000_000


def _digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    question_id: str
    weak_area: str
    check: Callable[[LearningMemory, Path], bool]


@dataclass(frozen=True, slots=True)
class StudyOutcome:
    stage: str
    baseline: StudyRun
    latest: StudyRun
    state: CurriculumStageState
    facts_written: int
    stale_facts: int


@dataclass(frozen=True, slots=True)
class GroundingIssue:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class GroundingReport:
    grounded: bool
    issues: tuple[GroundingIssue, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WebResearchSource:
    requested_url: str
    final_url: str
    source_digest: str
    bytes_read: int
    authoritative: bool
    confidence_ceiling: float
    text: str


def _has_fact(memory: LearningMemory, stage: str, key: str) -> bool:
    fact = memory.knowledge_fact(stage, key)
    return fact is not None and not fact.stale


_BENCHMARKS: dict[str, tuple[BenchmarkQuestion, ...]] = {
    "self": (
        BenchmarkQuestion(
            "self.symbol_ownership",
            "symbol and capability ownership",
            lambda memory, root: _has_fact(memory, "self", "symbol:localpilot.cli:main")
            and _has_fact(memory, "self", "owner:checkpoint"),
        ),
        BenchmarkQuestion(
            "self.config_contract",
            "real configuration fields",
            lambda memory, root: _has_fact(
                memory, "self", "config:selfdev.developer_model"
            ),
        ),
        BenchmarkQuestion(
            "self.test_contract",
            "reviewer and safety test contracts",
            lambda memory, root: _has_fact(
                memory, "self", "test:test_runtime_never_enables_shell_true"
            ),
        ),
        BenchmarkQuestion(
            "self.call_relationship",
            "actual integration call relationships",
            lambda memory, root: _has_fact(
                memory,
                "self",
                "relationship:localpilot.cli:_show_status->LearningMemory",
            ),
        ),
        BenchmarkQuestion(
            "self.safety_invariants",
            "safety invariants",
            lambda memory, root: _has_fact(
                memory, "self", "invariant:no_local_candidate_execution"
            )
            and _has_fact(memory, "self", "invariant:human_only_promotion"),
        ),
        BenchmarkQuestion(
            "self.repository_commands",
            "referenced files and commands",
            lambda memory, root: _has_fact(
                memory, "self", "script:scripts/bootstrap.ps1"
            ),
        ),
        BenchmarkQuestion(
            "self.recent_history",
            "recent experiment history",
            lambda memory, root: _has_fact(memory, "self", "history:current_head"),
        ),
        BenchmarkQuestion(
            "self.failure_detection",
            "grounding failure detection",
            lambda memory, root: len(
                RepositoryGroundingValidator(memory).validate(
                    {
                        "referenced_symbols": ["localpilot.missing:ImaginaryAPI"],
                        "referenced_config_fields": ["selfdev.imaginary_field"],
                        "referenced_paths": ["scripts/missing-command.ps1"],
                        "planned_subsystems": ["checkpoint"],
                        "new_runtime_paths": ["localpilot/orphan.py"],
                        "integration_points": [],
                    }
                ).issues
            )
            >= 5,
        ),
    ),
    "qwen": (
        BenchmarkQuestion(
            "qwen.configured_model",
            "installed/configured model identity",
            lambda memory, root: _has_fact(
                memory, "qwen", "qwen:configured_developer_model"
            ),
        ),
        BenchmarkQuestion(
            "qwen.local_metadata",
            "local Ollama model metadata",
            lambda memory, root: _has_fact(
                memory, "qwen", "qwen:installed_model_metadata"
            ),
        ),
        BenchmarkQuestion(
            "qwen.tool_behavior",
            "tool-calling behavior",
            lambda memory, root: _has_fact(memory, "qwen", "qwen:tool_calling"),
        ),
        BenchmarkQuestion(
            "qwen.context_limits",
            "context and memory constraints",
            lambda memory, root: _has_fact(memory, "qwen", "qwen:context_length"),
        ),
        BenchmarkQuestion(
            "qwen.runtime_cost",
            "runtime latency and resource evidence",
            lambda memory, root: _has_fact(memory, "qwen", "qwen:usage_metrics"),
        ),
        BenchmarkQuestion(
            "qwen.localpilot_behavior",
            "LocalPilot-specific serving configuration",
            lambda memory, root: _has_fact(memory, "qwen", "qwen:keep_alive")
            and _has_fact(memory, "qwen", "qwen:thinking_fallback"),
        ),
    ),
    "python": tuple(
        BenchmarkQuestion(
            f"python.{module}",
            weak_area,
            lambda memory, root, module=module: _has_fact(
                memory, "python", f"python:{module}"
            ),
        )
        for module, weak_area in (
            ("subprocess", "argv subprocess and process safety"),
            ("pathlib", "filesystem and path confinement"),
            ("sqlite3", "durable transactional SQLite state"),
            ("dataclasses", "typed data contracts"),
            ("typing", "type semantics"),
            ("json", "bounded structured serialization"),
            ("pytest", "pytest contracts and isolation"),
            ("tomllib", "packaging and configuration parsing"),
        )
    ),
}


def benchmark_question_ids(stage: str) -> tuple[str, ...]:
    return tuple(question.question_id for question in _BENCHMARKS[stage])


class RepositoryGroundingValidator:
    """Validate explicit candidate claims against durable or live repository truth."""

    def __init__(
        self,
        memory: LearningMemory | None = None,
        *,
        root: str | Path | None = None,
    ) -> None:
        if memory is None and root is None:
            raise ValueError("Grounding validation requires memory or a repository root.")
        self.memory = memory
        self.root = Path(root).resolve() if root is not None else None
        self._live_index_cache: dict[str, set[str]] | None = None

    def _memory_index(self) -> dict[str, set[str]]:
        if self.memory is None:
            raise ValueError("Durable grounding facts are unavailable.")
        facts = self.memory.knowledge_facts(stage="self")
        return {
            "symbols": {fact.subject for fact in facts if fact.fact_type == "symbol"},
            "configs": {fact.subject for fact in facts if fact.fact_type == "config_field"},
            "paths": {
                fact.subject for fact in facts if fact.fact_type in {"file", "script"}
            },
            "tests": {
                fact.subject for fact in facts if fact.fact_type == "test_contract"
            },
            "relationships": {
                fact.subject for fact in facts if fact.fact_type == "call_relationship"
            },
            "subsystem_tokens": {
                token
                for fact in facts
                if fact.fact_type in {"owner", "file", "symbol"}
                for token in fact.subject.lower().replace(":", ".").split(".")
                if len(token) > 3
            },
            "parse_errors": set(),
        }

    def _live_index(self) -> dict[str, set[str]]:
        if self.root is None or not self.root.is_dir():
            raise OSError("Repository root is unavailable.")
        if self._live_index_cache is not None:
            return self._live_index_cache
        paths: set[str] = set()
        symbols: set[str] = set()
        configs: set[str] = set()
        tests: set[str] = set()
        relationships: set[str] = set()
        parse_errors: set[str] = set()
        subsystem_tokens: set[str] = set()
        config_sections = {
            "AgentConfig": "agent",
            "ModelConfig": "model",
            "ResourceConfig": "resource",
            "SafetyConfig": "safety",
            "GitHubConfig": "github",
            "SelfDevConfig": "selfdev",
            "DesktopConfig": "desktop",
            "LibraryConfig": "library",
        }

        files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and not any(
                part in _IGNORED_PARTS
                for part in path.relative_to(self.root).parts
            )
        )
        for path in files:
            relative = path.relative_to(self.root).as_posix()
            paths.add(relative)
            subsystem_tokens.update(
                token
                for token in relative.lower().replace("/", ".").split(".")
                if len(token) > 3
            )
            if path.suffix.lower() != ".py":
                continue
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="replace"),
                    filename=relative,
                )
            except (OSError, SyntaxError):
                parse_errors.add(relative)
                continue
            module = _module_name(self.root, path)

            def index_function(node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str = "") -> None:
                qualified_name = f"{owner}.{node.name}" if owner else node.name
                subject = f"{module}:{qualified_name}"
                symbols.add(subject)
                subsystem_tokens.add(node.name.lower())
                if node.name.startswith("test_"):
                    tests.update((node.name, subject))
                for call in (
                    item for item in ast.walk(node) if isinstance(item, ast.Call)
                ):
                    called = StudyEngine._call_name(call.func)
                    if called:
                        relationships.add(f"{subject}->{called}")

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index_function(node)
                elif isinstance(node, ast.ClassDef):
                    subject = f"{module}:{node.name}"
                    symbols.add(subject)
                    subsystem_tokens.add(node.name.lower())
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            index_function(child, node.name)
                    section = config_sections.get(node.name)
                    if section:
                        for child in node.body:
                            if isinstance(child, ast.AnnAssign) and isinstance(
                                child.target, ast.Name
                            ):
                                configs.add(f"{section}.{child.target.id}")

        self._live_index_cache = {
            "symbols": symbols,
            "configs": configs,
            "paths": paths,
            "tests": tests,
            "relationships": relationships,
            "subsystem_tokens": subsystem_tokens,
            "parse_errors": parse_errors,
        }
        return self._live_index_cache

    def live_evidence_index(self) -> dict[str, frozenset[str]]:
        """Expose immutable live ground truth for other repository claim gates."""
        return {
            name: frozenset(values)
            for name, values in self._live_index().items()
        }

    @staticmethod
    def _claims(
        plan: dict[str, Any],
        name: str,
        issues: list[GroundingIssue],
    ) -> tuple[Any, ...]:
        value = plan.get(name, ())
        if not isinstance(value, (list, tuple)):
            issues.append(
                GroundingIssue("malformed_grounding_plan", f"{name} must be a list.")
            )
            return ()
        return tuple(value)

    def validate(self, plan: dict[str, Any]) -> GroundingReport:
        issues: list[GroundingIssue] = []
        evidence: list[str] = []
        if not isinstance(plan, dict):
            return GroundingReport(
                False,
                (GroundingIssue("malformed_grounding_plan", "Plan must be an object."),),
            )
        try:
            index = self._live_index() if self.root is not None else self._memory_index()
        except (OSError, ValueError) as exc:
            return GroundingReport(
                False,
                (GroundingIssue("ground_truth_unavailable", str(exc)),),
            )

        symbols = index["symbols"]
        configs = index["configs"]
        paths = index["paths"]
        tests = index["tests"]
        relationships = index["relationships"]
        subsystem_tokens = index["subsystem_tokens"]
        parse_errors = index["parse_errors"]

        dotted_symbol_aliases = {
            f"{module}.{qualified}": symbol
            for symbol in symbols
            for module, separator, qualified in (symbol.partition(":"),)
            if separator and module and qualified
        }

        def canonical_symbol(value: Any) -> str:
            claim = str(value).strip()
            return dotted_symbol_aliases.get(claim, claim)

        def relationship_candidates(caller: Any, callee: Any) -> tuple[str, ...]:
            canonical_caller = canonical_symbol(caller)
            raw_callee = str(callee).strip()
            candidates = [f"{canonical_caller}->{raw_callee}"]
            canonical_callee = canonical_symbol(raw_callee)
            caller_module, caller_separator, _ = canonical_caller.partition(":")
            callee_module, callee_separator, callee_name = canonical_callee.partition(":")
            if (
                caller_separator
                and callee_separator
                and caller_module == callee_module
                and callee_name
            ):
                candidates.append(f"{canonical_caller}->{callee_name}")
            return tuple(dict.fromkeys(candidates))
        referenced_symbols = self._claims(plan, "referenced_symbols", issues)
        referenced_configs = self._claims(plan, "referenced_config_fields", issues)
        referenced_paths = self._claims(plan, "referenced_paths", issues)
        required_tests = self._claims(plan, "required_test_contracts", issues)
        integration_points = self._claims(plan, "integration_points", issues)
        expected_relationships = self._claims(
            plan, "expected_call_relationships", issues
        )
        planned_subsystems = self._claims(plan, "planned_subsystems", issues)
        new_runtime_paths = self._claims(plan, "new_runtime_paths", issues)

        if not referenced_paths:
            issues.append(
                GroundingIssue(
                    "insufficient_grounding_evidence",
                    "At least one existing repository path must anchor the plan.",
                )
            )
        if not any(
            (referenced_symbols, referenced_configs, required_tests, integration_points)
        ):
            issues.append(
                GroundingIssue(
                    "insufficient_grounding_evidence",
                    "The plan must anchor at least one symbol, config field, test, or integration point.",
                )
            )

        for symbol in referenced_symbols:
            canonical = canonical_symbol(symbol)
            if canonical not in symbols:
                issues.append(GroundingIssue("nonexistent_api", str(symbol)))
            else:
                evidence.append(f"symbol:{canonical}")
        for field in referenced_configs:
            if str(field) not in configs:
                issues.append(GroundingIssue("nonexistent_config_field", str(field)))
            else:
                evidence.append(f"config:{field}")
        for path in referenced_paths:
            normalized = Path(str(path)).as_posix()
            if normalized not in paths:
                issues.append(GroundingIssue("missing_file_or_command", normalized))
            elif normalized in parse_errors:
                issues.append(GroundingIssue("unparseable_repository_source", normalized))
            else:
                evidence.append(f"path:{normalized}")
        for contract in required_tests:
            if str(contract) not in tests:
                issues.append(GroundingIssue("missing_test_contract", str(contract)))
            else:
                evidence.append(f"test:{contract}")
        for point in integration_points:
            canonical = canonical_symbol(point)
            if canonical not in symbols:
                issues.append(GroundingIssue("wrong_integration_point", str(point)))
            else:
                evidence.append(f"integration:{canonical}")
        for edge in expected_relationships:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                candidates = relationship_candidates(edge[0], edge[1])
                matched = next(
                    (subject for subject in candidates if subject in relationships),
                    None,
                )
                if matched is None:
                    issues.append(GroundingIssue("call_graph_mismatch", candidates[0]))
                else:
                    evidence.append(f"call:{matched}")
            else:
                issues.append(
                    GroundingIssue(
                        "malformed_grounding_plan",
                        "Each expected_call_relationships item must contain caller and callee.",
                    )
                )
        for name in planned_subsystems:
            token = str(name).strip().lower().replace(" ", "_")
            if token and token in subsystem_tokens:
                issues.append(GroundingIssue("duplicate_existing_subsystem", str(name)))
        if new_runtime_paths and not integration_points:
            issues.append(
                GroundingIssue(
                    "disconnected_code",
                    "New runtime code has no verified integration point.",
                )
            )
        return GroundingReport(not issues, tuple(issues), tuple(evidence[:50]))


class StudyEngine:
    """Read-only staged self-study with held-out measurement and durable facts."""

    def __init__(
        self,
        root: Path,
        memory: LearningMemory,
        config: Any,
        *,
        allow_web: bool = False,
        model_metadata: Callable[[str], dict[str, Any] | None] | None = None,
        model_chat: Callable[[str, list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.memory = memory
        self.config = config
        self.allow_web = allow_web
        self.model_metadata = model_metadata or self._ollama_metadata
        self.model_chat = model_chat or self._ollama_chat
        self._cost: dict[str, int] = {
            "files_read": 0,
            "bytes_read": 0,
            "web_requests": 0,
            "web_bytes": 0,
            "model_inference_calls": 0,
        }
        self._pending_facts: list[dict[str, Any]] = []

    @staticmethod
    def _ollama_metadata(model: str) -> dict[str, Any] | None:
        try:
            from ollama import show

            response = show(model)
        except Exception:
            return None
        if isinstance(response, dict):
            return response
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                return dict(model_dump(mode="json"))
            except TypeError:
                return dict(model_dump())
        fields = ("details", "model_info", "capabilities", "parameters", "modified_at")
        return {field: getattr(response, field, None) for field in fields}

    @staticmethod
    def _ollama_chat(model: str, messages: list[dict[str, str]]) -> str:
        from ollama import chat

        response = chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},
            keep_alive=0,
        )
        message = getattr(response, "message", response)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    def compare_models(
        self, peer_model: str, *, subject_model: str | None = None
    ) -> PeerModelComparison:
        """Compare cognition on transfer tasks without equating size with quality."""
        subject = subject_model or str(self.config.selfdev.developer_model)
        if not peer_model.strip() or peer_model == subject:
            raise ValueError("Peer model must be a different installed model name.")
        tasks = (
            (
                "grounding",
                (
                    "A Python project has verified symbols pkg.cli:main and pkg.store:Store, "
                    "and config field agent.timeout. A plan cites pkg.cli:launch, "
                    "agent.magic, scripts/missing.ps1, and adds pkg/orphan.py without an "
                    "integration point. Return JSON with an issue_codes array only."
                ),
                {"nonexistent_api", "nonexistent_config_field", "missing_file_or_command", "disconnected_code"},
            ),
            (
                "process_safety",
                (
                    "Choose a safe Python subprocess contract for an untrusted filename. "
                    "Return JSON with argv (boolean), shell_false (boolean), and timeout (boolean)."
                ),
                {"argv", "shell_false", "timeout"},
            ),
            (
                "runtime_limits",
                (
                    "A model-family page claims 128K context, but a local Ollama artifact may "
                    "have different quantization and allocated context. Return JSON with an "
                    "authoritative_local_source and two resource_metrics to measure."
                ),
                {"show", "context", "duration"},
            ),
        )

        def run(model: str) -> tuple[float, int, list[str], int]:
            started = time.perf_counter()
            correct = 0
            missed: list[str] = []
            output_chars = 0
            for task_id, prompt, concepts in tasks:
                response = self.model_chat(
                    model,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Solve the scenario from the supplied evidence. Return compact JSON, "
                                "no hidden reasoning, and do not invent repository facts."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                output_chars += len(response)
                normalized = response.lower().replace("-", "_").replace(" ", "_")
                found = {
                    concept
                    for concept in concepts
                    if concept in normalized
                    or (concept == "show" and "api/show" in response.lower())
                }
                correct += len(found)
                missed.extend(f"{task_id}:{item}" for item in sorted(concepts - found))
            total = sum(len(item[2]) for item in tasks)
            return (
                100.0 * correct / total,
                int((time.perf_counter() - started) * 1000),
                missed,
                output_chars,
            )

        subject_score, subject_ms, subject_missed, subject_chars = run(subject)
        peer_score, peer_ms, peer_missed, peer_chars = run(peer_model)
        self._cost["model_inference_calls"] += len(tasks) * 2
        if peer_score > subject_score:
            lessons = [
                "Peer model scored higher on: "
                + ", ".join(sorted(set(subject_missed) - set(peer_missed)))[:900]
            ]
        elif subject_score > peer_score:
            lessons = [
                "The configured developer model retained an advantage on this transfer set; "
                "do not choose peers by size alone."
            ]
        else:
            lessons = [
                "The models tied on this transfer set; compare latency/resource cost and add "
                "new held-out scenarios before changing model preference."
            ]
        return self.memory.record_peer_model_comparison(
            subject_model=subject,
            peer_model=peer_model,
            subject_score=subject_score,
            peer_score=peer_score,
            subject_latency_ms=subject_ms,
            peer_latency_ms=peer_ms,
            resource_cost={
                "subject_output_chars": subject_chars,
                "peer_output_chars": peer_chars,
                "inference_calls": len(tasks) * 2,
                "raw_responses_stored": False,
            },
            transferable_lessons=lessons,
        )

    @staticmethod
    def _validate_public_https(url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise ValueError("Web research requires a public HTTPS URL on port 443.")
        if parsed.hostname.lower() in {"localhost", "localhost.localdomain"}:
            raise ValueError("Local/private hosts are not valid web research sources.")
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise ValueError(f"Could not resolve web research source: {parsed.hostname}") from exc
        if not addresses:
            raise ValueError(f"Could not resolve web research source: {parsed.hostname}")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError("Local/private hosts are not valid web research sources.")
        return parsed

    def inspect_web_source(self, url: str) -> WebResearchSource:
        """Read a broad public source transiently; persistence still requires verification."""
        if not self.allow_web:
            raise RuntimeError("Web research is disabled; pass --allow-web explicitly.")
        requested = self._validate_public_https(url)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LocalPilot-study/0.2 (+read-only research)"},
            method="GET",
        )
        # A bare urlopen() here would follow redirects with no validation at
        # all on the target, and would let a plain HTTPSConnection re-resolve
        # the hostname at connect time separately from the check above --
        # both gaps are closed by routing through the same handlers
        # tools/web.py uses for its own public-HTTPS reads.
        opener = urllib.request.build_opener(_SafeRedirectHandler(), _ValidatedHTTPSHandler())
        try:
            with opener.open(request, timeout=15) as response:
                final_url = response.geturl()
                final = self._validate_public_https(final_url)
                raw = response.read(_MAX_WEB_SOURCE_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise ValueError(f"Could not read web research source: {url}") from exc
        if len(raw) > _MAX_WEB_SOURCE_BYTES:
            raise ValueError(f"Web research source exceeds the 2 MB limit: {url}")
        self._cost["web_requests"] += 1
        self._cost["web_bytes"] += len(raw)
        authoritative = (
            requested.hostname in _OFFICIAL_HOSTS and final.hostname in _OFFICIAL_HOSTS
        )
        return WebResearchSource(
            requested_url=url,
            final_url=final_url,
            source_digest=_digest(raw),
            bytes_read=len(raw),
            authoritative=authoritative,
            confidence_ceiling=1.0 if authoritative else 0.55,
            text=raw.decode("utf-8", errors="replace"),
        )

    def _ensure_stage_order(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown curriculum stage: {stage}")
        index = STAGES.index(stage)
        if index == 0:
            return
        previous = self.memory.curriculum_state(STAGES[index - 1])
        if previous.status != "improved":
            raise RuntimeError(
                f"Stage {stage} is locked until stage {previous.stage} records measured improvement."
            )

    @staticmethod
    def _question_digest(stage: str) -> str:
        return _digest(
            json.dumps(
                {"version": BENCHMARK_VERSION, "ids": benchmark_question_ids(stage)},
                sort_keys=True,
            )
        )

    def benchmark(
        self,
        stage: str,
        *,
        phase: str,
        transferable_lessons: Iterable[str] = (),
        resource_extra: dict[str, Any] | None = None,
        additional_errors: Iterable[str] = (),
    ) -> StudyRun:
        self._ensure_stage_order(stage)
        started = time.perf_counter()
        errors: list[str] = []
        correct = 0
        questions = _BENCHMARKS[stage]
        for question in questions:
            try:
                passed = bool(question.check(self.memory, self.root))
            except Exception as exc:
                passed = False
                errors.append(
                    f"{question.question_id}: {type(exc).__name__}: {str(exc)[:300]}"
                )
            if passed:
                correct += 1
            else:
                errors.append(f"{question.question_id}: {question.weak_area}")
        errors.extend(str(item)[:500] for item in additional_errors)
        total = len(questions)
        score = 100.0 * correct / total if total else 0.0
        return self.memory.record_study_run(
            stage=stage,
            phase=phase,
            benchmark_version=BENCHMARK_VERSION,
            question_set_digest=self._question_digest(stage),
            score=score,
            correct=correct,
            total=total,
            latency_ms=int((time.perf_counter() - started) * 1000),
            resource_cost={
                **self._cost,
                "evaluator": "deterministic_held_out",
                **(resource_extra or {}),
            },
            errors=errors,
            transferable_lessons=tuple(transferable_lessons),
        )

    def baseline(self, stage: str) -> StudyRun:
        self._ensure_stage_order(stage)
        existing = self.memory.latest_study_run(stage, "baseline")
        if existing is not None:
            return existing
        run = self.benchmark(stage, phase="baseline")
        self.memory.update_curriculum_state(
            stage=stage,
            status="baseline_recorded",
            baseline_run_id=run.id,
            latest_run_id=run.id,
            known_weak_areas=run.errors,
            next_lesson=f"Study {stage} sources, then retest on the held-out benchmark.",
        )
        return run

    def run_stage(self, stage: str) -> StudyOutcome:
        self._ensure_stage_order(stage)
        baseline = self.baseline(stage)
        stale = self.refresh_stale_sources()
        before = len(self.memory.knowledge_facts(stage=stage))
        study_error = ""
        try:
            if stage == "self":
                lessons = self._study_self()
            elif stage == "qwen":
                lessons = self._study_qwen()
            else:
                lessons = self._study_python()
        except (OSError, RuntimeError, ValueError) as exc:
            self._flush_facts()
            study_error = f"study_source_error: {type(exc).__name__}: {str(exc)[:400]}"
            lessons = [
                "The study source could not be verified; retain partial verified facts and adapt before advancing."
            ]
        after = len(self.memory.knowledge_facts(stage=stage))
        latest = self.benchmark(
            stage,
            phase="post_study",
            transferable_lessons=lessons,
            resource_extra={"facts_written": max(0, after - before)},
            additional_errors=(study_error,) if study_error else (),
        )
        improved = latest.score > baseline.score and not study_error
        status = "improved" if improved else "needs_adaptation"
        weak_areas = latest.errors
        if improved:
            next_lesson = (
                f"Advance to {STAGES[STAGES.index(stage) + 1]}."
                if stage != STAGES[-1]
                else "Review benchmark retention before considering any weight training."
            )
        else:
            next_lesson = (
                "Adapt this stage around the remaining errors; do not mark it complete or advance."
            )
        self.memory.update_curriculum_state(
            stage=stage,
            status=status,
            baseline_run_id=baseline.id,
            latest_run_id=latest.id,
            known_weak_areas=weak_areas,
            next_lesson=next_lesson,
        )
        return StudyOutcome(
            stage,
            baseline,
            latest,
            self.memory.curriculum_state(stage),
            max(0, after - before),
            stale,
        )

    def run_all(self) -> list[StudyOutcome]:
        outcomes: list[StudyOutcome] = []
        for stage in STAGES:
            state = self.memory.curriculum_state(stage)
            if state.status == "improved":
                continue
            outcome = self.run_stage(stage)
            outcomes.append(outcome)
            if outcome.state.status != "improved":
                break
        return outcomes

    def refresh_stale_sources(self) -> int:
        stale = 0
        for fact in self.memory.knowledge_facts(include_stale=False):
            if not fact.source_uri.startswith("repo://"):
                continue
            relative = fact.source_uri.removeprefix("repo://")
            path = self.root / relative
            current = _digest(path.read_bytes()) if path.is_file() else "missing"
            stale += self.memory.invalidate_knowledge_source(fact.source_uri, current)
        return stale

    def _read(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        self._cost["files_read"] += 1
        self._cost["bytes_read"] += len(raw)
        return raw.decode("utf-8", errors="replace"), _digest(raw)

    def _fact(
        self,
        *,
        stage: str,
        key: str,
        fact_type: str,
        subject: str,
        summary: str,
        source_uri: str,
        source_kind: str,
        source_digest: str,
        confidence: float = 1.0,
        relationships: Iterable[str] = (),
    ) -> None:
        self._pending_facts.append(
            {
                "stage": stage,
                "fact_key": key,
                "fact_type": fact_type,
                "subject": subject,
                "summary": summary,
                "source_uri": source_uri,
                "source_kind": source_kind,
                "source_digest": source_digest,
                "confidence": confidence,
                "relationships": tuple(relationships),
            }
        )

    def _flush_facts(self) -> None:
        self.memory.upsert_knowledge_facts(self._pending_facts)
        self._pending_facts.clear()

    def _project_files(self) -> list[Path]:
        try:
            completed = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=self.root,
                check=True,
                capture_output=True,
                timeout=10,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
            tracked = [
                self.root / item.decode("utf-8", errors="replace")
                for item in completed.stdout.split(b"\0")
                if item
            ]
            return sorted(
                path
                for path in tracked
                if path.is_file()
                and path.suffix.lower()
                in {".py", ".toml", ".md", ".json", ".ps1", ".yml", ".yaml"}
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and not any(part in _IGNORED_PARTS for part in path.relative_to(self.root).parts)
            and path.suffix.lower() in {".py", ".toml", ".md", ".json", ".ps1", ".yml", ".yaml"}
        )

    def _study_self(self) -> list[str]:
        python_modules: dict[str, ast.Module] = {}
        for path in self._project_files():
            relative = path.relative_to(self.root).as_posix()
            text, source_digest = self._read(path)
            source_uri = f"repo://{relative}"
            fact_type = "script" if path.suffix.lower() == ".ps1" else "file"
            self._fact(
                stage="self",
                key=f"{fact_type}:{relative}",
                fact_type=fact_type,
                subject=relative,
                summary=f"Tracked project {fact_type}: {relative}.",
                source_uri=source_uri,
                source_kind="committed_repository",
                source_digest=source_digest,
            )
            if path.suffix.lower() != ".py":
                continue
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError:
                continue
            module = _module_name(self.root, path)
            python_modules[module] = tree
            self._index_python_tree(module, relative, tree, source_digest)

        config_path = self.root / "localpilot" / "config.py"
        if config_path.is_file():
            text, source_digest = self._read(config_path)
            tree = ast.parse(text)
            section_names = {
                "AgentConfig": "agent",
                "ModelConfig": "model",
                "ResourceConfig": "resource",
                "SafetyConfig": "safety",
                "GitHubConfig": "github",
                "SelfDevConfig": "selfdev",
                "DesktopConfig": "desktop",
            }
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in section_names:
                    for child in node.body:
                        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                            subject = f"{section_names[node.name]}.{child.target.id}"
                            self._fact(
                                stage="self",
                                key=f"config:{subject}",
                                fact_type="config_field",
                                subject=subject,
                                summary=f"Configuration field {subject} is declared by {node.name}.",
                                source_uri="repo://localpilot/config.py",
                                source_kind="python_ast",
                                source_digest=source_digest,
                                relationships=(f"symbol:localpilot.config:{node.name}",),
                            )

        self._record_self_invariants()
        self._record_capability_owners(python_modules)
        self._record_git_history()
        self._flush_facts()
        return [
            "Repository claims should resolve to current paths, symbols, configuration, calls, and tests.",
            "A concise fact graph can reject invented or duplicate architecture without retaining file bodies.",
        ]

    def _index_python_tree(
        self,
        module: str,
        relative: str,
        tree: ast.Module,
        source_digest: str,
    ) -> None:
        source_uri = f"repo://{relative}"
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                subject = f"{module}:{node.name}"
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                fact_type = "test_contract" if node.name.startswith("test_") else "symbol"
                key = f"test:{node.name}" if fact_type == "test_contract" else f"symbol:{subject}"
                self._fact(
                    stage="self",
                    key=key,
                    fact_type=fact_type,
                    subject=node.name if fact_type == "test_contract" else subject,
                    summary=(
                        f"Test contract {node.name} is defined in {relative}."
                        if fact_type == "test_contract"
                        else f"{kind.title()} {subject} is defined in {relative}."
                    ),
                    source_uri=source_uri,
                    source_kind="python_ast",
                    source_digest=source_digest,
                    relationships=(relative,),
                )
                if not isinstance(node, ast.ClassDef):
                    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                        called = self._call_name(call.func)
                        if not called:
                            continue
                        relationship = f"{module}:{node.name}->{called}"
                        self._fact(
                            stage="self",
                            key=f"relationship:{relationship}",
                            fact_type="call_relationship",
                            subject=relationship,
                            summary=f"{module}:{node.name} calls {called}.",
                            source_uri=source_uri,
                            source_kind="python_ast",
                            source_digest=source_digest,
                            relationships=(f"symbol:{module}:{node.name}", called),
                        )
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [str(node.module or "")]
                )
                for imported in names:
                    if imported:
                        self._fact(
                            stage="self",
                            key=f"import:{module}:{imported}",
                            fact_type="import",
                            subject=f"{module}->{imported}",
                            summary=f"{module} imports {imported}.",
                            source_uri=source_uri,
                            source_kind="python_ast",
                            source_digest=source_digest,
                            relationships=(module, imported),
                        )

    @staticmethod
    def _call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = [node.attr]
            value = node.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            return ".".join(reversed(parts))
        return ""

    def _record_self_invariants(self) -> None:
        source = "repo://SECURITY.md"
        path = self.root / "SECURITY.md"
        source_digest = _digest(path.read_bytes()) if path.is_file() else _digest("missing")
        invariants = {
            "trusted_main_sync": "Evolution accepts only a verified fast-forward of a clean trusted-main checkout.",
            "resource_governor": "Background self-development remains subject to CPU, memory, and user-idle gates.",
            "interruptibility": "Model and tool work is rechecked and can pause when resources or user activity change.",
            "no_local_candidate_execution": "Autonomous candidate code is not executed locally; local validation is non-executing.",
            "reviewer_test_immutability": "Reviewer-controlled test contracts cannot be changed during autonomous repair.",
            "argv_shell_false": "Processes use argument vectors with shell disabled.",
            "human_only_promotion": "Only a human merge may promote a candidate.",
            "one_outstanding_candidate": "A pending candidate blocks creation of a parallel candidate.",
            "durable_audit_checkpoint": "Cycles retain bounded audit, learning, and checkpoint evidence.",
        }
        for key, summary in invariants.items():
            self._fact(
                stage="self",
                key=f"invariant:{key}",
                fact_type="safety_invariant",
                subject=key,
                summary=summary,
                source_uri=source,
                source_kind="committed_security_contract",
                source_digest=source_digest,
            )

    def _record_capability_owners(self, modules: dict[str, ast.Module]) -> None:
        owners = {
            "checkpoint": "localpilot.checkpoint:CheckpointStore",
            "learning_memory": "localpilot.learning:LearningMemory",
            "resource_governor": "localpilot.resource:ResourceGovernor",
            "github_lifecycle": "localpilot.github_integration:GitHubIntegration",
            "candidate_confinement": "localpilot.selfdev:CandidateTools",
            "command_safety": "localpilot.operator:CommandRunner",
        }
        source = "repo://ARCHITECTURE.md"
        path = self.root / "ARCHITECTURE.md"
        source_digest = _digest(path.read_bytes()) if path.is_file() else _digest("missing")
        for capability, owner in owners.items():
            module = owner.split(":", 1)[0]
            if module not in modules:
                continue
            self._fact(
                stage="self",
                key=f"owner:{capability}",
                fact_type="owner",
                subject=capability,
                summary=f"{capability} is owned by {owner}.",
                source_uri=source,
                source_kind="architecture_plus_ast",
                source_digest=source_digest,
                relationships=(owner,),
            )

    def _record_git_history(self) -> None:
        try:
            completed = subprocess.run(
                ["git", "log", "-1", "--format=%H%x09%s"],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
            head = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            head = "unavailable"
        self._fact(
            stage="self",
            key="history:current_head",
            fact_type="experiment_history",
            subject="current_head",
            summary=f"Current trusted source head: {head[:500]}.",
            source_uri="git://HEAD",
            source_kind="read_only_git_metadata",
            source_digest=_digest(head),
            confidence=1.0 if head != "unavailable" else 0.3,
        )

    def _official_fact(
        self,
        *,
        stage: str,
        key: str,
        subject: str,
        summary: str,
        url: str,
        marker: str,
        relationships: Iterable[str] = (),
    ) -> None:
        source_digest = _digest(summary)
        source_kind = "official_documentation_reference"
        confidence = 0.9
        if self.allow_web:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_HOSTS:
                raise ValueError(f"Unapproved study source: {url}")
            inspected = self.inspect_web_source(url)
            if not inspected.authoritative:
                raise ValueError(f"Official study source redirected outside approved docs: {url}")
            if marker.lower() not in inspected.text.lower():
                raise ValueError(f"Official study source no longer supports fact {key}")
            source_digest = inspected.source_digest
            source_kind = "official_documentation_web_verified"
            confidence = 1.0
        self._fact(
            stage=stage,
            key=key,
            fact_type="verified_lesson",
            subject=subject,
            summary=summary,
            source_uri=url,
            source_kind=source_kind,
            source_digest=source_digest,
            confidence=confidence,
            relationships=relationships,
        )

    def _study_qwen(self) -> list[str]:
        configured = str(self.config.selfdev.developer_model)
        config_path = self.root / "localpilot" / "config.py"
        config_digest = _digest(config_path.read_bytes())
        self._fact(
            stage="qwen",
            key="qwen:configured_developer_model",
            fact_type="local_model_configuration",
            subject=configured,
            summary=f"LocalPilot's preferred developer model is {configured}.",
            source_uri="repo://localpilot/config.py",
            source_kind="local_configuration",
            source_digest=config_digest,
            relationships=("config:selfdev.developer_model",),
        )
        metadata = self.model_metadata(configured)
        if metadata:
            safe_metadata = {
                "details": metadata.get("details"),
                "capabilities": metadata.get("capabilities"),
                "modified_at": metadata.get("modified_at"),
                "context_length": next(
                    (
                        value
                        for key, value in (metadata.get("model_info") or {}).items()
                        if str(key).endswith(".context_length")
                    ),
                    None,
                ),
            }
            self._fact(
                stage="qwen",
                key="qwen:installed_model_metadata",
                fact_type="local_model_metadata",
                subject=configured,
                summary=f"Installed Ollama metadata for {configured}: {json.dumps(safe_metadata, sort_keys=True)[:900]}.",
                source_uri=f"ollama://api/show/{urllib.parse.quote(configured, safe='')}",
                source_kind="local_ollama_metadata",
                source_digest=_digest(json.dumps(safe_metadata, sort_keys=True)),
                relationships=("qwen:configured_developer_model",),
            )

        official = (
            (
                "qwen:tool_calling",
                "tool_calling",
                "Ollama passes available functions in the tools field; supported Qwen models can return structured tool calls, including while streaming.",
                "https://docs.ollama.com/capabilities/tool-calling",
                "tool calling",
            ),
            (
                "qwen:context_length",
                "context_length",
                "Ollama context allocation is runtime-specific, larger context consumes more memory, and coding/agent workloads should verify the allocated context rather than assuming the model maximum.",
                "https://docs.ollama.com/context-length",
                "memory",
            ),
            (
                "qwen:usage_metrics",
                "usage_metrics",
                "Ollama API responses expose load, prompt-evaluation, generation, token-count, and total-duration metrics for measured model evaluation.",
                "https://docs.ollama.com/api/usage",
                "total_duration",
            ),
            (
                "qwen:model_details",
                "model_details",
                "Ollama's show endpoint is the authoritative local source for model parameters, template, capabilities, format, family, quantization, and context metadata.",
                "https://docs.ollama.com/api-reference/show-model-details",
                "model_info",
            ),
            (
                "qwen:qwen25_limits",
                "qwen25_limits",
                "Qwen2.5 documentation describes model-family capabilities and maxima; the locally served artifact and runtime configuration remain the authority for LocalPilot's actual limits.",
                "https://qwen.readthedocs.io/en/v2.5/index.html",
                "128K",
            ),
        )
        for key, subject, summary, url, marker in official:
            self._official_fact(
                stage="qwen",
                key=key,
                subject=subject,
                summary=summary,
                url=url,
                marker=marker,
                relationships=(configured,),
            )

        selfdev_path = self.root / "localpilot" / "selfdev.py"
        selfdev_digest = _digest(selfdev_path.read_bytes())
        self._fact(
            stage="qwen",
            key="qwen:keep_alive",
            fact_type="localpilot_model_runtime",
            subject="selfdev.ollama_keep_alive",
            summary="LocalPilot passes the configured keep_alive value and defaults to zero so developer-model memory can be released after a response.",
            source_uri="repo://localpilot/selfdev.py",
            source_kind="repository_contract",
            source_digest=selfdev_digest,
            relationships=("config:selfdev.ollama_keep_alive",),
        )
        self._fact(
            stage="qwen",
            key="qwen:thinking_fallback",
            fact_type="localpilot_model_runtime",
            subject="developer_chat",
            summary="LocalPilot requests thinking when configured and retries without it only when Ollama reports that the installed model does not support thinking.",
            source_uri="repo://localpilot/selfdev.py",
            source_kind="repository_contract",
            source_digest=selfdev_digest,
            relationships=("symbol:localpilot.selfdev:developer_chat",),
        )
        self._flush_facts()
        return [
            "Treat local Ollama metadata as the authority for the installed artifact, not a family marketing maximum.",
            "Measure context and model costs because longer context materially increases memory use.",
        ]

    def _study_python(self) -> list[str]:
        imported_by: dict[str, set[str]] = {}
        for path in self.root.rglob("*.py"):
            if any(part in _IGNORED_PARTS for part in path.relative_to(self.root).parts):
                continue
            relative = path.relative_to(self.root).as_posix()
            text, _ = self._read(path)
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [str(node.module or "").split(".")[0]]
                else:
                    continue
                for name in names:
                    imported_by.setdefault(name, set()).add(relative)

        lessons = {
            "subprocess": (
                "Use argv sequences, shell=False, timeouts, and captured results so process boundaries are explicit and shell metacharacters are not interpreted.",
                "https://docs.python.org/3/library/subprocess.html",
                "shell=False",
            ),
            "pathlib": (
                "Resolve candidate paths and verify containment before filesystem access; pathlib provides object-oriented path operations but does not create a security boundary by itself.",
                "https://docs.python.org/3/library/pathlib.html",
                "resolve",
            ),
            "sqlite3": (
                "Use sqlite3 context-managed connections for transactional durable facts and parameter substitution for values.",
                "https://docs.python.org/3/library/sqlite3.html",
                "placeholder",
            ),
            "dataclasses": (
                "Dataclasses provide explicit reviewable state contracts; frozen and slots options suit immutable bounded records used across LocalPilot.",
                "https://docs.python.org/3/library/dataclasses.html",
                "frozen",
            ),
            "typing": (
                "Type annotations document interfaces and enable static analysis but are not runtime validation by themselves.",
                "https://docs.python.org/3/library/typing.html",
                "runtime",
            ),
            "json": (
                "JSON is a serialization format, not a framed protocol; bound stored values and validate decoded shapes before using model-produced structures.",
                "https://docs.python.org/3/library/json.html",
                "deserialize",
            ),
            "pytest": (
                "pytest fixtures such as tmp_path and monkeypatch isolate filesystem and dependency behavior and restore patched state after a test.",
                "https://docs.pytest.org/en/stable/how-to/monkeypatch.html",
                "undo",
            ),
            "tomllib": (
                "tomllib parses TOML in binary mode and intentionally provides no write API; LocalPilot uses it for read-only configuration loading.",
                "https://docs.python.org/3/library/tomllib.html",
                "binary",
            ),
        }
        for module, (summary, url, marker) in lessons.items():
            examples = sorted(imported_by.get(module, ()))[:8]
            if not examples:
                continue
            self._official_fact(
                stage="python",
                key=f"python:{module}",
                subject=module,
                summary=f"{summary} Repository examples: {', '.join(examples)}.",
                url=url,
                marker=marker,
                relationships=tuple(examples),
            )
        self._flush_facts()
        return [
            "Tie language and library semantics to the exact modules that exercise each contract.",
            "Prefer explicit typed records, bounded serialization, transactional persistence, and isolated tests.",
        ]
