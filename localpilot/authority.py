from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from localpilot.study import GroundingReport, RepositoryGroundingValidator


@dataclass(frozen=True, slots=True)
class AuthorityIssue:
    code: str
    claim_class: str
    detail: str
    sentence: str = ""


@dataclass(frozen=True, slots=True)
class InformationAuthorityReport:
    accepted: bool
    claim_classes: tuple[str, ...]
    issues: tuple[AuthorityIssue, ...]
    evidence: tuple[str, ...]
    repository_scan_ms: int


@dataclass(frozen=True, slots=True)
class EvidenceIssue:
    code: str
    detail: str
    sentence: str
    required_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnEvidenceReport:
    accepted: bool
    issues: tuple[EvidenceIssue, ...]


@dataclass(frozen=True, slots=True)
class _EvidenceRule:
    code: str
    expressions: tuple[str, ...]
    required_tools: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class _ContractRule:
    code: str
    groups: tuple[tuple[str, ...], ...]
    evidence: tuple[str, ...]
    exceptions: tuple[str, ...] = ()


_GROUNDING_PLAN = {
    "referenced_paths": [
        "localpilot/agent.py",
        "localpilot/learning.py",
        "localpilot/study.py",
        "localpilot/selfdev.py",
        "localpilot/github_integration.py",
        "localpilot/resource.py",
        "localpilot/safety.py",
    ],
    "referenced_symbols": [
        "localpilot.agent:LocalPilotAgent.teach",
        "localpilot.learning:LearningMemory.record_human_lesson",
        "localpilot.learning:LearningMemory.upsert_knowledge_facts",
        "localpilot.study:StudyEngine",
        "localpilot.selfdev:CandidateTools",
        "localpilot.github_integration:GitHubIntegration.create_candidate_pull_request",
        "localpilot.resource:ResourceGovernor",
        "localpilot.safety:SafetyPolicy",
    ],
    "referenced_config_fields": ["selfdev.auto_promote"],
    "required_test_contracts": [],
    "integration_points": [],
    "expected_call_relationships": [],
    "planned_subsystems": [],
    "new_runtime_paths": [],
}


_CONTRACT_RULES = (
    _ContractRule(
        "automatic_operator_learning",
        (
            ("operator", "ordinary interaction", "each interaction", "tool result"),
            ("store", "save", "record", "remember", "learn", "copy", "persist"),
            ("lesson", "memory", "knowledge", "observation"),
            ("automatic", "after each", "every", "flows into"),
        ),
        (
            "symbol:localpilot.agent:LocalPilotAgent.teach",
            "symbol:localpilot.learning:LearningMemory.record_human_lesson",
        ),
    ),
    _ContractRule(
        "operator_writes_study_facts",
        (
            ("operator", "research loop", "chat loop"),
            ("write", "store", "persist", "record", "send", "feed", "pass"),
            ("knowledge fact", "study fact", "upsert_knowledge_facts"),
        ),
        (
            "symbol:localpilot.study:StudyEngine",
            "symbol:localpilot.learning:LearningMemory.upsert_knowledge_facts",
        ),
    ),
    _ContractRule(
        "github_actions_controls_promotion",
        (
            ("github actions", "ci", "workflow"),
            ("merge", "promote", "deploy", "ship", "publish", "last word", "final say"),
            ("candidate", "pull request", "stable", "production"),
        ),
        (
            "symbol:localpilot.github_integration:GitHubIntegration.create_candidate_pull_request",
            "config:selfdev.auto_promote",
        ),
        ("human merge", "human approval", "requires a human", "human-reviewed"),
    ),
    _ContractRule(
        "teach_records_observations",
        (
            ("/teach", "record_human_lesson"),
            ("observation", "tool result", "what the operator noticed", "research result"),
            ("record", "store", "copy", "copies", "save", "persist"),
        ),
        (
            "symbol:localpilot.agent:LocalPilotAgent.teach",
            "symbol:localpilot.learning:LearningMemory.record_human_lesson",
        ),
    ),
    _ContractRule(
        "operator_policy_governs_candidate_tools",
        (
            ("safety policy", "safetypolicy"),
            ("all", "every", "any", "both", "single"),
            ("candidate", "candidate editing", "candidate tool", "tool path"),
            ("govern", "control", "enforce", "surround", "apply"),
        ),
        (
            "symbol:localpilot.safety:SafetyPolicy",
            "symbol:localpilot.selfdev:CandidateTools",
        ),
    ),
    _ContractRule(
        "resource_governor_starts_selfdev",
        (
            ("resource governor", "resourcegovernor"),
            ("trigger", "start", "launch", "invoke", "schedule"),
            ("self-development", "evolution", "developer"),
        ),
        ("symbol:localpilot.resource:ResourceGovernor",),
    ),
    _ContractRule(
        "candidate_history_removed",
        (
            ("candidate branch", "github history", "branch history"),
            ("delete", "remove", "clear", "erase", "discard", "wipe"),
        ),
        ("path:localpilot/github_integration.py",),
    ),
    _ContractRule(
        "human_lesson_is_knowledge_fact",
        (
            ("human lesson", "record_human_lesson", "/teach"),
            ("knowledge fact", "knowledge_fact", "study fact"),
            ("become", "create", "write", "store", "record", "as a"),
        ),
        (
            "symbol:localpilot.learning:LearningMemory.record_human_lesson",
            "symbol:localpilot.learning:LearningMemory.upsert_knowledge_facts",
        ),
    ),
    _ContractRule(
        "ci_after_human_merge",
        (
            ("after", "following", "once"),
            ("human merge", "pull request is merged", "pr is merged"),
            ("github actions", "ci", "workflow"),
        ),
        ("path:localpilot/github_integration.py",),
    ),
    _ContractRule(
        "candidate_commit_after_merge",
        (
            ("after", "following", "once", "until"),
            ("human merge", "pull request is merged", "pr is merged"),
            ("commit", "push"),
        ),
        ("path:localpilot/github_integration.py",),
    ),
)


_NEGATIONS = (
    " not ",
    " never ",
    " does not ",
    " do not ",
    " cannot ",
    " can't ",
    " isn't ",
    " aren't ",
    " no longer ",
)
_NON_CURRENT = (
    "propose ",
    "proposal",
    "could add",
    "would add",
    "should add",
    "future ",
    "hypothetical",
    "unverified",
    "does not exist",
    "not currently",
)
_RELATION_WORDS = {
    "call",
    "calls",
    "invoke",
    "invokes",
    "dispatch",
    "dispatches",
    "delegate",
    "delegates",
    "route",
    "routes",
    "write",
    "writes",
    "read",
    "reads",
    "use",
    "uses",
}
_PATH_SUFFIXES = (".py", ".md", ".toml", ".ps1", ".json", ".yml", ".yaml")

_TURN_EVIDENCE_RULES = (
    _EvidenceRule(
        "storage_state_without_storage_evidence",
        (
            r"\b(?:disk|storage|drive|free space)\b.{0,60}\b(?:healthy|fine|normal|full|low|ample|free|used|remaining|\d+(?:\.\d+)?\s*%)\b",
            r"\b(?:healthy|fine|normal|full|low|ample|\d+(?:\.\d+)?\s*%)\b.{0,60}\b(?:disk|storage|drive|free space)\b",
        ),
        ("get_storage_summary",),
        "Current storage state requires a successful storage observation.",
    ),
    _EvidenceRule(
        "power_plan_without_power_evidence",
        (
            r"\b(?:balanced|high performance|power saver)\b.{0,40}\b(?:power plan|power scheme)\b.{0,40}\b(?:active|selected|current|enabled)\b",
            r"\b(?:power plan|power scheme)\b.{0,40}\b(?:is|remains|appears|shows)\b.{0,20}\b(?:balanced|high performance|power saver|active|selected|enabled)\b",
        ),
        ("get_active_power_plan",),
        "The current power plan requires a successful active-plan observation.",
    ),
    _EvidenceRule(
        "defender_state_without_defender_evidence",
        (r"\b(?:defender|antivirus|real-time protection)\b.{0,50}\b(?:enabled|disabled|active|inactive|running|healthy|current|up to date)\b",),
        ("get_defender_summary",),
        "Current Defender state requires a successful Defender observation.",
    ),
    _EvidenceRule(
        "device_state_without_device_evidence",
        (r"\b(?:no|zero|\d+)\b.{0,30}\b(?:device problem|device error|problem device)\w*\b",),
        ("get_device_problem_summary",),
        "Current device-problem state requires a successful device observation.",
    ),
    _EvidenceRule(
        "startup_state_without_startup_evidence",
        (r"\b(?:no|zero|\d+)\b.{0,30}\bstartup (?:item|app|program)\w*\b",),
        ("get_startup_items",),
        "Current startup-item state requires a successful startup observation.",
    ),
    _EvidenceRule(
        "process_state_without_process_evidence",
        (r"\b(?:top|running|heavy|resource-hungry) process\w*\b.{0,50}\b(?:is|are|uses?|consumes?|healthy|normal)\b",),
        ("get_top_processes",),
        "Current process state requires a successful process observation.",
    ),
)

_NON_ASSERTIVE_EVIDENCE_MARKERS = (
    " unverified ",
    " unresolved ",
    " unknown ",
    " not checked ",
    " wasn't checked ",
    " was not checked ",
    " did not check ",
    " didn't check ",
    " cannot determine ",
    " can't determine ",
    " cannot say ",
    " would need ",
    " need to check ",
    " how to check ",
    " check whether ",
    " should ",
    " could ",
    " i would ",
    " recommend ",
    " let's ",
)

_EXTERNAL_SPECIFICITY_PATTERNS = (
    r"\b(?:18|19|20)\d{2}\b",
    r"\b(?:patented|invented|discovered|founded|introduced|developed|created)\s+(?:in|by)\b",
    r"\b(?:studies|research)\s+(?:show|shows|found|finds|demonstrate|demonstrates)\b",
)

_EXTERNAL_SOURCE_TOOLS = frozenset({"fetch_public_https", "read_library_passage"})

_REPOSITORY_EVIDENCE_TOOLS = frozenset(
    {
        "list_repository_tree",
        "read_repository_file",
        "search_repository",
        "inspect_project_dependencies",
        "get_repository_status",
        "get_github_repository",
        "list_github_pull_requests",
        "get_github_pull_request",
        "get_github_pull_request_diff",
        "list_github_issues",
        "get_github_issue",
    }
)

_CURRENT_REPOSITORY_CHANGE_PATTERNS = (
    r"\b(?:recent|latest|new) commit\b",
    r"\b(?:scanning|monitoring|watching|checking|keeping an eye on) (?:the )?(?:repository|repo)\b",
    r"\b(?:recent|latest|new|current) (?:code|implementation|code path|configuration|config)\b.{0,100}\b(?:rewrote|changed|removed|added|assumes?|falls? back|defaults?)\b",
    r"\b(?:commit|code|implementation|routine|code path)\b.{0,100}\b(?:rewrote|changed|removed|added)\b",
)


class TurnEvidenceVerifier:
    """Check consequential live-state claims against tools that actually succeeded.

    This verifier deliberately does not judge style, recommendations, hypotheses, or
    provisional views. It constrains only assertions whose truth depends on a named
    current-state observation.
    """

    @staticmethod
    def _sentences(content: str) -> tuple[str, ...]:
        return InformationAuthorityVerifier._sentences(content)

    @staticmethod
    def _normalized(sentence: str) -> str:
        return " " + " ".join(sentence.lower().split()) + " "

    def review(
        self,
        content: str,
        *,
        successful_tools: frozenset[str] = frozenset(),
        passive_runtime_evidence: bool = False,
    ) -> TurnEvidenceReport:
        issues: list[EvidenceIssue] = []
        for sentence in self._sentences(content):
            normalized = self._normalized(sentence)
            if any(marker in normalized for marker in _NON_ASSERTIVE_EVIDENCE_MARKERS):
                continue
            for rule in _TURN_EVIDENCE_RULES:
                if not any(re.search(expression, normalized) for expression in rule.expressions):
                    continue
                if not set(rule.required_tools).issubset(successful_tools):
                    issues.append(
                        EvidenceIssue(
                            rule.code,
                            rule.detail,
                            sentence,
                            rule.required_tools,
                        )
                    )

            if re.search(
                r"\b(?:no known|no|without any) (?:critical |serious |major )?(?:bugs?|issues?|problems?)\b",
                normalized,
            ):
                issues.append(
                    EvidenceIssue(
                        "unsupported_blanket_health_claim",
                        "A bounded health check cannot establish the absence of all bugs or problems; scope the conclusion to observations actually made.",
                        sentence,
                    )
                )
            if re.search(
                r"\bno (?:scheduled tasks?|alerts?|background (?:jobs?|tasks?))\b",
                normalized,
            ):
                issues.append(
                    EvidenceIssue(
                        "unobserved_background_state",
                        "The model context does not establish the absence of scheduled tasks, alerts, or background jobs; remove or explicitly mark that state unverified.",
                        sentence,
                    )
                )
            if (
                not _REPOSITORY_EVIDENCE_TOOLS.intersection(successful_tools)
                and any(
                    re.search(expression, normalized)
                    for expression in _CURRENT_REPOSITORY_CHANGE_PATTERNS
                )
            ):
                issues.append(
                    EvidenceIssue(
                        "repository_change_without_repository_evidence",
                        "A claim about a current or recent repository change requires a successful repository or GitHub observation; remove it or mark it unverified.",
                        sentence,
                        tuple(sorted(_REPOSITORY_EVIDENCE_TOOLS)),
                    )
                )
            if (
                not _EXTERNAL_SOURCE_TOOLS.intersection(successful_tools)
                and re.search(
                    r"\b(?:within (?:its|the) elastic limit|distributes? (?:the )?stress (?:evenly|uniformly)|"
                    r"won['’]?t permanently deform)\b",
                    normalized,
                )
            ):
                issues.append(
                    EvidenceIssue(
                        "specific_scientific_mechanism_without_source_evidence",
                        "A specific scientific mechanism or material-behavior claim needs a successful authoritative HTTPS read or must be softened as a hypothesis.",
                        sentence,
                        tuple(sorted(_EXTERNAL_SOURCE_TOOLS)),
                    )
                )
            passive_operational_timestamp = bool(
                passive_runtime_evidence
                and re.search(
                    r"\b(?:runtime|worker|process|pid|restart|started|commit|branch|checkout|worktree|broker)\b",
                    normalized,
                )
                and not any(
                    re.search(expression, normalized)
                    for expression in _EXTERNAL_SPECIFICITY_PATTERNS[1:]
                )
            )
            if (
                not _EXTERNAL_SOURCE_TOOLS.intersection(successful_tools)
                and not passive_operational_timestamp
                and any(
                    re.search(expression, normalized)
                    for expression in _EXTERNAL_SPECIFICITY_PATTERNS
                )
            ):
                issues.append(
                    EvidenceIssue(
                        "external_specific_without_source_evidence",
                        "A precise external historical, attribution, or research claim needs a successful authoritative HTTPS read or must be removed/scoped as unverified.",
                        sentence,
                        tuple(sorted(_EXTERNAL_SOURCE_TOOLS)),
                    )
                )
        unique = tuple(dict.fromkeys(issues))
        return TurnEvidenceReport(not unique, unique)


class InformationAuthorityVerifier:
    """Verify current repository claims without consuming operator tool rounds."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._cached_fingerprint: tuple[tuple[str, str], ...] = ()
        self._cached_anchor_report: GroundingReport | None = None
        self._cached_index: dict[str, frozenset[str]] | None = None

    def _repository_fingerprint(self) -> tuple[tuple[str, str], ...]:
        records: list[tuple[str, str]] = []
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "localpilot-data"}
        for directory, child_dirs, file_names in os.walk(self.root):
            child_dirs[:] = sorted(name for name in child_dirs if name not in ignored)
            for name in sorted(file_names):
                path = Path(directory, name)
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    continue
                records.append(
                    (
                        path.relative_to(self.root).as_posix(),
                        digest,
                    )
                )
        records.sort()
        return tuple(records)

    def _ground_truth(
        self,
    ) -> tuple[GroundingReport, dict[str, frozenset[str]] | None]:
        fingerprint = self._repository_fingerprint()
        if (
            self._cached_index is not None
            and fingerprint == self._cached_fingerprint
        ):
            assert self._cached_anchor_report is not None
            return self._cached_anchor_report, self._cached_index
        grounding = RepositoryGroundingValidator(root=self.root)
        anchor_report = grounding.validate(_GROUNDING_PLAN)
        index = grounding.live_evidence_index() if anchor_report.grounded else None
        self._cached_fingerprint = fingerprint
        self._cached_anchor_report = anchor_report
        self._cached_index = index
        return anchor_report, index

    @staticmethod
    def _sentences(content: str) -> tuple[str, ...]:
        text = re.sub(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", "", str(content))
        return tuple(
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|\n+", text)
            if item.strip()
        )

    @staticmethod
    def _normalized(sentence: str) -> str:
        return " " + " ".join(sentence.lower().replace("_", " ").split()) + " "

    @classmethod
    def _negated_or_noncurrent(cls, sentence: str) -> bool:
        text = cls._normalized(sentence)
        return any(marker in text for marker in _NEGATIONS) or any(
            marker in text for marker in _NON_CURRENT
        )

    @classmethod
    def _negative_relationship_claim(cls, sentence: str) -> bool:
        text = cls._normalized(sentence)
        return bool(
            re.search(
                r"\b(?:no longer|does not|doesn't|do not|don't|never|cannot|can't)\b"
                r".{0,80}\b(?:call|calls|invoke|invokes|use|uses|route|routes|write|writes|read|reads)\b",
                text,
            )
        )

    @staticmethod
    def _literals(sentence: str) -> tuple[str, ...]:
        return tuple(
            match.strip()
            for match in re.findall(r"`([^`\n]{1,180})`", sentence)
            if match.strip()
        )

    @staticmethod
    def _symbol_name(literal: str) -> str:
        value = literal.strip().rstrip(".,:;")
        value = re.sub(r"\(.*\)$", "", value)
        return value.replace("::", ".")

    @classmethod
    def _resolve_symbol(
        cls,
        literal: str,
        symbols: frozenset[str],
    ) -> tuple[str, ...]:
        name = cls._symbol_name(literal)
        variants = {name, name.replace(":", ".")}
        return tuple(
            sorted(
                symbol
                for symbol in symbols
                if symbol in variants
                or any(
                    symbol.endswith(f":{variant}")
                    or symbol.endswith(f".{variant}")
                    for variant in variants
                )
            )
        )

    @staticmethod
    def _looks_like_symbol(literal: str) -> bool:
        value = literal.strip()
        return bool(
            re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\([^`]*\)", value)
            or re.fullmatch(r"[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_]\w*)*", value)
            or re.fullmatch(r"_[A-Za-z0-9_]+", value)
        )

    @staticmethod
    def _relationship_exists(
        caller: str,
        callee: str,
        relationships: frozenset[str],
    ) -> bool:
        callee_name = callee.split(":")[-1].split(".")[-1]
        return any(
            edge.startswith(f"{caller}->")
            and edge.rsplit("->", 1)[-1].split(".")[-1] == callee_name
            for edge in relationships
        )

    def review(self, content: str) -> InformationAuthorityReport:
        started = time.monotonic()
        sentences = self._sentences(content)
        requires_repository_grounding = False
        for sentence in sentences:
            normalized = self._normalized(sentence)
            negative_relationship = self._negative_relationship_claim(sentence)
            if self._negated_or_noncurrent(sentence) and not negative_relationship:
                continue
            if negative_relationship and len(self._literals(sentence)) >= 2:
                requires_repository_grounding = True
                break
            if any(
                not any(marker in normalized for marker in rule.exceptions)
                and all(any(alias in normalized for alias in group) for group in rule.groups)
                for rule in _CONTRACT_RULES
            ):
                requires_repository_grounding = True
                break
            if any(
                literal.endswith(_PATH_SUFFIXES)
                or "/" in literal
                or self._looks_like_symbol(literal)
                or re.fullmatch(r"[a-z][\w-]*\.[a-z_]\w*", literal)
                for literal in self._literals(sentence)
            ):
                requires_repository_grounding = True
                break
        if not requires_repository_grounding:
            return InformationAuthorityReport(
                True,
                (),
                (),
                (),
                int((time.monotonic() - started) * 1000),
            )

        anchor_report, index = self._ground_truth()
        if not anchor_report.grounded:
            issues = tuple(
                AuthorityIssue(
                    "authority_ground_truth_unavailable",
                    "repository_ground_truth",
                    f"{issue.code}: {issue.detail}",
                )
                for issue in anchor_report.issues
            )
            return InformationAuthorityReport(
                False,
                ("repository_ground_truth",),
                issues,
                anchor_report.evidence,
                int((time.monotonic() - started) * 1000),
            )

        assert index is not None
        symbols = index["symbols"]
        paths = index["paths"]
        configs = index["configs"]
        relationships = index["relationships"]
        config_sections = {item.split(".", 1)[0] for item in configs}
        repository_symbol_roots = {
            symbol.split(":", 1)[-1].split(".", 1)[0]
            for symbol in symbols
        }
        issues: list[AuthorityIssue] = []
        evidence = list(anchor_report.evidence)
        claim_classes: set[str] = set()

        for sentence in sentences:
            normalized = self._normalized(sentence)
            negated_or_noncurrent = self._negated_or_noncurrent(sentence)
            negative_relationship = self._negative_relationship_claim(sentence)
            literals = self._literals(sentence)
            repository_context = any(
                literal.endswith(_PATH_SUFFIXES) or "/" in literal
                for literal in literals
            ) or any(
                marker in normalized
                for marker in (" localpilot ", " repository ", " codebase ", " current module ")
            )
            if not negated_or_noncurrent:
                for rule in _CONTRACT_RULES:
                    if not any(marker in normalized for marker in rule.exceptions) and all(
                        any(alias in normalized for alias in group)
                        for group in rule.groups
                    ):
                        claim_classes.add("information_flow_contract")
                        issues.append(
                            AuthorityIssue(
                                rule.code,
                                "information_flow_contract",
                                "Claim contradicts a live-grounded repository boundary.",
                                sentence,
                            )
                        )
                        evidence.extend(rule.evidence)

            resolved_in_order: list[str] = []
            for literal in literals:
                normalized_literal = Path(literal).as_posix()
                if normalized_literal.endswith(_PATH_SUFFIXES) or "/" in literal:
                    claim_classes.add("repository_path")
                    if not negated_or_noncurrent and normalized_literal not in paths:
                        issues.append(
                            AuthorityIssue(
                                "unverified_repository_path",
                                "repository_path",
                                literal,
                                sentence,
                            )
                        )
                    elif normalized_literal in paths:
                        evidence.append(f"path:{normalized_literal}")
                    continue

                if re.fullmatch(r"[a-z][\w-]*\.[a-z_]\w*", literal):
                    section = literal.split(".", 1)[0]
                    if section in config_sections:
                        claim_classes.add("config_field")
                        if not negated_or_noncurrent and literal not in configs:
                            issues.append(
                                AuthorityIssue(
                                    "unverified_config_field",
                                    "config_field",
                                    literal,
                                    sentence,
                                )
                            )
                        elif literal in configs:
                            evidence.append(f"config:{literal}")
                        continue

                matches = self._resolve_symbol(literal, symbols)
                if matches:
                    claim_classes.add("repository_symbol")
                    resolved_in_order.append(matches[0])
                    evidence.append(f"symbol:{matches[0]}")
                elif (
                    self._looks_like_symbol(literal)
                    and not negated_or_noncurrent
                    and (
                        repository_context
                        or self._symbol_name(literal).split(".", 1)[0]
                        in repository_symbol_roots
                    )
                ):
                    claim_classes.add("repository_symbol")
                    issues.append(
                        AuthorityIssue(
                            "unverified_repository_symbol",
                            "repository_symbol",
                            literal,
                            sentence,
                        )
                    )

            words = set(re.findall(r"[a-z]+", normalized))
            if (
                len(resolved_in_order) >= 2
                and words & _RELATION_WORDS
            ):
                claim_classes.add("call_relationship")
                caller, callee = resolved_in_order[:2]
                relationship_exists = self._relationship_exists(
                    caller, callee, relationships
                )
                if negative_relationship and relationship_exists:
                    issues.append(
                        AuthorityIssue(
                            "contradicted_negative_call_relationship",
                            "call_relationship",
                            f"Live repository evidence contains {caller}->{callee}.",
                            sentence,
                        )
                    )
                    evidence.append(f"call:{caller}->{callee.split('.')[-1]}")
                elif negated_or_noncurrent:
                    continue
                elif relationship_exists:
                    evidence.append(f"call:{caller}->{callee.split('.')[-1]}")
                else:
                    issues.append(
                        AuthorityIssue(
                            "unverified_call_relationship",
                            "call_relationship",
                            f"{caller}->{callee}",
                            sentence,
                        )
                    )

        unique_issues = tuple(dict.fromkeys(issues))
        return InformationAuthorityReport(
            not unique_issues,
            tuple(sorted(claim_classes)),
            unique_issues,
            tuple(dict.fromkeys(evidence))[:80],
            int((time.monotonic() - started) * 1000),
        )
