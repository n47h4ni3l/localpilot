from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: RiskLevel
    fn: Callable[..., Any]


class SafetyPolicy:
    """Policy scaffold for PC-control tools.

    v0.1 ships only read-only PC tools. Self-development writes are separately
    sandboxed to candidate workspaces and therefore do not use this policy.
    """

    def __init__(
        self,
        auto_allow_read_only: bool = True,
        auto_allow_reversible: bool = True,
        require_confirmation_for_destructive: bool = True,
    ) -> None:
        self.auto_allow_read_only = auto_allow_read_only
        self.auto_allow_reversible = auto_allow_reversible
        self.require_confirmation_for_destructive = require_confirmation_for_destructive

    def permits_without_confirmation(self, risk: RiskLevel) -> bool:
        if risk is RiskLevel.READ_ONLY:
            return self.auto_allow_read_only
        if risk is RiskLevel.REVERSIBLE:
            return self.auto_allow_reversible
        return not self.require_confirmation_for_destructive
