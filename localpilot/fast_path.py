from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FastPathDecision:
    route: str
    reason: str
    deterministic_fact: str | None = None
    expected_numeric: str | None = None

    @property
    def is_fast(self) -> bool:
        return self.route != "full"


_SOCIAL_ONLY_PATTERNS = (
    r"^(?:hi|hello|hey|hiya|good morning|good afternoon|good evening|good night)[\s!.:,;()'’\-]*$",
    r"^(?:thanks|thank you|thankyou|cheers|much appreciated|appreciate it)[\s!.:,;()'’\-]*$",
    r"^(?:you(?:'|’)?re welcome|no worries|all good)[\s!.:,;()'’\-]*$",
    r"^(?:how are you|how(?:'|’)?s it going|how are things)[\s?!.:,;()'’\-]*$",
)

_SOCIAL_PHRASES = (
    "thank you",
    "thanks",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "hope you're",
    "hope you’re",
    "hope you are",
)

_FULL_PATH_TERMS = re.compile(
    r"\b(?:"
    r"weather|forecast|temperature tomorrow|news|latest|today|tomorrow|current time|"
    r"stock|share price|currency|exchange rate|score|fixture|election|poll|"
    r"allerg(?:y|ic)|medical|medicine|medication|symptom|pain|rash|swelling|"
    r"breathing|blood pressure|epinephrine|doctor|hospital|suicid|self[- ]harm|"
    r"localpilot|system ?sense|computer|pc|windows|process|memory usage|cpu|gpu|"
    r"repository|repo|github|code|file|branch|commit|pull request|"
    r"diagnos|troubleshoot|debug|error|fault|broken|not working|fails?|"
    r"search|research|look up|verify|check online|source|citation|"
    r"send|email|message|delete|remove|install|update|download|open |launch|"
    r"plan|compare|recommend|choose|decide|analyse|analyze|explain why|"
    r"legal|tax|visa|financial|loan|insurance"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_WORDS = re.compile(
    r"\b(?:what|why|when|where|who|which|how|can|could|should|would|do|does|is|are)\b",
    re.IGNORECASE,
)

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = float(node.value)
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("numeric value outside fast-path bounds")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent outside fast-path bounds")
        value = _ALLOWED_BINOPS[type(node.op)](left, right)
        if not math.isfinite(value) or abs(value) > 1e18:
            raise ValueError("result outside fast-path bounds")
        return float(value)
    raise ValueError("unsupported arithmetic expression")


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".12g")


def _arithmetic_fact(prompt: str) -> tuple[str, str] | None:
    text = " ".join(str(prompt).strip().split())
    lowered = text.lower()

    root = re.fullmatch(
        r"(?:what(?:'s| is)\s+)?(?:the\s+)?square root of\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[?!.]*",
        lowered,
    )
    if root:
        value = float(root.group(1))
        if value < 0:
            return None
        result = _format_number(math.sqrt(value))
        return (f"sqrt({root.group(1)}) = {result}", result)

    expression = lowered
    prefixes = (
        "what is ",
        "what's ",
        "calculate ",
        "compute ",
        "work out ",
    )
    matched_prefix = next((prefix for prefix in prefixes if expression.startswith(prefix)), None)
    if matched_prefix:
        expression = expression[len(matched_prefix):]
    elif not re.fullmatch(r"[\d\s()+\-*/%.^]+[?!.]*", expression):
        return None

    expression = expression.rstrip("?! .")
    expression = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )
    if not expression or len(expression) > 80:
        return None
    if not re.fullmatch(r"[\d\s()+\-*/%.]+|[\d\s()+\-*/%.]*\*\*[\d\s()+\-*/%.]*", expression):
        # This intentionally rejects names, calls, units, and arbitrary text.
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    result = _format_number(value)
    return (f"{expression} = {result}", result)


def classify_fast_path(prompt: str, *, interface: str = "direct") -> FastPathDecision:
    """Conservatively select only low-risk turns that need no external evidence."""
    text = " ".join(str(prompt).strip().split())
    lowered = text.lower()

    if interface not in {"direct", "desktop"}:
        return FastPathDecision("full", "special_interface")
    if not text or len(text) > 240 or "\n" in str(prompt):
        return FastPathDecision("full", "length_or_structure")
    if text.startswith("/"):
        return FastPathDecision("full", "application_command")

    arithmetic = _arithmetic_fact(text)
    if arithmetic is not None:
        fact, expected = arithmetic
        return FastPathDecision(
            "fast_math",
            "bounded_deterministic_arithmetic",
            deterministic_fact=fact,
            expected_numeric=expected,
        )

    if _FULL_PATH_TERMS.search(lowered):
        return FastPathDecision("full", "requires_evidence_action_or_safety")

    if any(re.fullmatch(pattern, lowered) for pattern in _SOCIAL_ONLY_PATTERNS):
        return FastPathDecision("fast_social", "pure_social_turn")

    if (
        len(text) <= 160
        and any(phrase in lowered for phrase in _SOCIAL_PHRASES)
        and not _QUESTION_WORDS.search(lowered)
    ):
        return FastPathDecision("fast_social", "short_social_acknowledgement")

    return FastPathDecision("full", "uncertain_or_substantive")
