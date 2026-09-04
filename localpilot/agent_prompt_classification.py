"""Stateless prompt-classification helpers for LocalPilotAgent.

Extracted verbatim (no logic changes) from agent.py as part of the low-risk
mechanical decomposition. These answer 'what kind of prompt is this' --
temporal/web-dependent, practical-troubleshooting, bounded-conversational,
operational-self-status -- and are used to decide what evidence a response
needs. Left behind on LocalPilotAgent as staticmethod(...) shims so both
self._name(...) and LocalPilotAgent._name(...) call sites elsewhere in
agent.py keep resolving unchanged. Three of these call each other; those
internal calls were rewritten from LocalPilotAgent._name(...) to a plain
_name(...) call, since the target now lives in this same module and the
class name doesn't resolve here -- see the commit message for specifics.
ask() and _continue_high_reasoning_answer() were not touched."""

import re

def _evidence_requirements(prompt: str) -> set[str]:
    """Identify explicit evidence sources the owner asked LocalPilot to inspect."""
    text = " ".join(str(prompt).lower().split())
    requirements: set[str] = set()

    def mentions(*phrases: str) -> bool:
        return any(
            re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
            for phrase in phrases
        )

    forbids_public_web = bool(
        re.search(
            r"\b(?:do not|don['’]?t|without) (?:use|using|search|browse|consult|access)?\s*"
            r"(?:the )?(?:public )?(?:web|internet|online sources?)\b",
            text,
        )
    )

    if re.search(r"https://\S+", text) and not forbids_public_web:
        requirements.add("public HTTPS")
    if (
        _is_practical_troubleshooting_prompt(prompt)
        and not forbids_public_web
    ):
        requirements.update({"public web discovery", "public HTTPS"})

    action_terms = (
        "inspect", "review", "check", "verify", "read", "search", "look at",
        "examine", "consult", "use", "list", "show", "find", "open", "current", "actual", "status", "latest",
    )
    asks_for_evidence = mentions(*action_terms)
    if not forbids_public_web and asks_for_evidence and mentions(
        "public web", "public internet", "the web", "online", "primary source"
    ):
        requirements.add("public HTTPS")
    if _is_temporal_web_prompt(prompt) and not forbids_public_web:
        requirements.update({"public web discovery", "public HTTPS"})
    if asks_for_evidence and mentions(
        "library", "local library", "my books", "the books", "manuals"
    ):
        requirements.add("local library")

    pr_number = re.search(r"\bpr\s*#?\s*\d+\b", text) is not None
    if pr_number or mentions("github", "pull request"):
        requirements.add("private GitHub")
    repo_context = mentions(
        "repository", "repo", "local repository", "trusted repository",
        "source code", "codebase", "localpilot", "github",
    )
    if asks_for_evidence and repo_context and mentions("issue", "ci", "commit", "branch"):
        requirements.add("private GitHub")

    local_repo_explicit = mentions(
        "local repository", "trusted repository", "source code", "codebase"
    )
    generic_repo = mentions("repository", "repo") and not mentions(
        "github repository", "github repo"
    )
    if asks_for_evidence and (local_repo_explicit or generic_repo):
        requirements.add("trusted repository")
    self_structure_terms = (
        "module", "class", "function", "dependency", "configuration", "config",
        "integration point", "architecture", "file", "command",
    )
    if mentions("localpilot") and mentions(*self_structure_terms):
        requirements.add("trusted repository")

    pc_specific = mentions(
        "windows", "process", "storage", "disk", "startup", "defender", "device",
        "power plan", "health check", "system health", "my pc", "this pc", "your pc",
        "my computer", "this computer", "your computer",
    )
    if asks_for_evidence and pc_specific:
        requirements.add("Windows/PC state")
    return requirements


def _is_temporal_web_prompt(prompt: str) -> bool:
    """Recognize claims whose truth depends on both current discovery and a live source."""
    text = " ".join(str(prompt).lower().split())
    temporal = bool(
        re.search(
            r"\b(?:latest|newest|current|today|as of (?:today|now|\d{4}))\b",
            text,
        )
    )
    web_research = bool(
        re.search(
            r"\b(?:public (?:web|internet)|the (?:web|internet)|online|primary sources?|"
            r"fact[- ]check|look (?:it|this) up|browse|search the web)\b",
            text,
        )
    )
    return temporal and web_research


def _is_practical_troubleshooting_prompt(prompt: str) -> bool:
    """Recognize recurring product or device faults that benefit from live support evidence."""
    text = " ".join(str(prompt).lower().split())
    device = bool(
        re.search(
            r"\b(?:3d printer|printer|nozzle|extruder|filament|device|machine|appliance|"
            r"router|camera|phone|laptop|computer|motor|battery|engine|vehicle)\b",
            text,
        )
    )
    fault = bool(
        re.search(
            r"\b(?:keeps?|repeatedly|again|straight away|won['’]?t|not working|fails?|"
            r"clogg?(?:ed|ing)?|jam(?:med|ming)?|error|fault|broken|overheat(?:ing)?|"
            r"leak(?:ing)?|disconnect(?:ing)?|stuck)\b",
            text,
        )
    )
    return device and fault


def _practical_troubleshooting_fallback(
    prompt: str, issues: tuple[str, ...]
) -> str | None:
    """Return a conservative correction when model-only safety recovery is exhausted."""
    issue_set = set(issues)
    if "unsafe_pla_temperature_example" in issue_set and re.search(
        r"\bh2s\b", str(prompt), re.IGNORECASE
    ):
        return (
            "I need to correct the previous advice: do not use 240°C or higher as a general PLA "
            "recommendation, and I will not invent a replacement temperature. Use the temperature in your "
            "filament profile and follow Bambu Lab's official H2S nozzle/hotend unclogging procedure: "
            "https://wiki.bambulab.com/en/h2s/troubleshooting/nozzle-clog. Because the clog returns "
            "immediately, work through that procedure and inspect the filament path, extruder, and hotend "
            "for retained debris; if it still recurs, stop replacing settings blindly and contact Bambu Lab "
            "support with logs and photos."
        )
    if {
        "practical_troubleshooting_source_unattributed",
        "unsafe_pla_temperature_example",
    }.intersection(issue_set):
        return (
            "[LocalPilot withheld a practical-troubleshooting draft because its source attribution or a "
            "numeric safety detail remained unreliable after bounded correction.]"
        )
    return None


def _requires_information_authority_review(prompt: str) -> bool:
    """Limit the extra review pass to LocalPilot's own current architecture."""
    text = " ".join(str(prompt).lower().split())
    if "localpilot" not in text:
        return False
    if _is_operational_self_status_prompt(prompt):
        return False
    return any(
        term in text
        for term in (
            "architecture",
            "module",
            "class",
            "function",
            "dependency",
            "configuration",
            "config",
            "integration",
            "learning",
            "memory",
            "study",
            "self-development",
            "candidate",
            "promotion",
            "merge",
            "safety",
            "github actions",
            "ci",
        )
    )


def _looks_like_generic_reset(content: str) -> bool:
    text = " ".join(str(content).strip().lower().split())
    if len(text) > 300:
        return False
    return any(
        phrase in text
        for phrase in (
            "hello! how can i help",
            "hello, how can i help",
            "hi! how can i help",
            "how may i assist you",
            "what can i help you with",
        )
    )


def _is_bounded_conversational_prompt(prompt: str) -> bool:
    """Use proportionate reasoning for turns whose value is judgment, not research."""
    text = " ".join(str(prompt).lower().split())
    explicit_conversational_style = bool(
        re.search(
            r"\b(?:keep it conversational|just (?:chat|talk)|casual question|"
            r"talk to me like a (?:friend|colleague))\b",
            text,
        )
    )
    explicit_research_request = bool(
        re.search(
            r"\b(?:search|research|look (?:it|this) up|verify|fact[- ]check|cite|citation|"
            r"sources?|latest|current events?|public web|local library)\b",
            text,
        )
    )
    if explicit_conversational_style and not explicit_research_request:
        return True
    bounded_invitation = bool(
        re.search(
            r"\b(?:room to think|what has your attention|pick something ordinary|"
            r"felt curiosity|topic-selection story|agree with me|just agree|say you agree|"
            r"how are you|what(?:'s| is) interesting|help me plan|realistic plan|"
            r"what (?:kind of )?conversation would you (?:enjoy|like)|"
            r"what would you (?:enjoy|like) (?:talking|chatting) about|what should we talk about|"
            r"plan my|prioriti(?:es|se|ze)|weekend plan|talk to me like a friend|"
            r"help me (?:switch off|wind down|unwind|relax)|"
            r"suggest (?:one|a) (?:small|simple|ordinary|relaxing) thing)\b",
            text,
        )
    )
    personal_media_choice = bool(
        not explicit_research_request
        and re.search(
            r"\b(?:what would you (?:put on|listen to|watch|read|choose)|"
            r"what (?:music|film|movie|show|book|background audio) would you (?:pick|choose)|"
            r"if you wanted .{0,80}\bwhat would you (?:put on|listen to|watch|read))\b",
            text,
        )
    )
    workplace_judgment = bool(
        not explicit_research_request
        and re.search(
            r"\b(?:supplier|client|customer|order|meeting|workday|starting work|scattered)\b",
            text,
        )
        and re.search(
            r"\b(?:what would you do first|what would you actually say|what should i do first|"
            r"choose one useful way to help|help me get (?:clear|started)|how would you handle)\b",
            text,
        )
    )
    return bounded_invitation or personal_media_choice or workplace_judgment


def _is_operational_self_status_prompt(prompt: str) -> bool:
    """Recognize questions answered by passive lifecycle and self-dev evidence."""
    text = " ".join(str(prompt).lower().split())
    self_reference = bool(
        re.search(r"\b(?:localpilot|you|your|yourself)\b", text)
        or re.search(
            r"\b(?:the |current |new )?(?:runtime|background[- ]worker|"
            r"evolution orchestrator|learning[_ ]memory)\b",
            text,
        )
    )
    status_topic = bool(
        re.search(
            r"\b(?:restart(?:ed|s|ing)?|runtime|pid|branch|commit|checkout|up[- ]to[- ]date|"
            r"background[- ]worker|self[- ]development|evolution|learning progress|"
            r"what (?:have you|you've) learned|what (?:have you|you've) changed|"
            r"learning[_ ]memory|durable memory|do you have (?:a )?memory|"
            r"store or retrieve (?:new )?learning|"
            r"public (?:web|internet)|internet access|web access|"
            r"search(?:ed|ing)? (?:the )?(?:web|internet)|"
            r"browse(?:d|ing)? (?:the )?(?:web|internet)|"
            r"candidate (?:pr|pull request|experiment)|autonomous work|autonom(?:y|ous|ously)|"
            r"becom(?:e|ing) more capable|what (?:is|isn't|'s) (?:stable|blocked)|"
            r"what (?:is )?blocking (?:you|localpilot)|what still requires me|handover)\b",
            text,
        )
    )
    return self_reference and status_topic


def _is_historical_autonomy_status_prompt(prompt: str) -> bool:
    request = " ".join(str(prompt).lower().split())
    return bool(
        re.search(
            r"\b(?:while i was away|what did .{0,40} accomplish|waste(?:d)? time|"
            r"stay out of my way|since i (?:left|was away))\b",
            request,
        )
    )
