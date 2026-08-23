from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentConfig:
    name: str = "LocalPilot"
    data_dir: str = "localpilot-data"
    # Legacy compatibility input. New operator research uses an advisory soft
    # budget and a separate hard safety ceiling so the model can decide when
    # another observation is genuinely necessary.
    max_tool_rounds: int = 12
    research_soft_tool_rounds: int = 12
    research_hard_tool_rounds: int = 24


@dataclass(slots=True)
class ModelConfig:
    """The everyday operator model. Self-development has its own model."""

    provider: str = "ollama"
    name: str = "gpt-oss:20b"
    # GPT-OSS requires an explicit low/medium/high reasoning level. High is the
    # default because LocalPilot's mission favors careful reasoning over speed.
    think: bool | str = "high"
    temperature: float = 0.1
    # Ollama may otherwise allocate only a small runtime context window even
    # when the model supports much more. Tool-driven agent work needs enough
    # room to retain the owner request plus repository/GitHub observations.
    context_tokens: int = 32768


@dataclass(slots=True)
class ResourceConfig:
    active_priority: str = "below_normal"
    idle_priority: str = "normal"
    background_idle_seconds: int = 600
    max_cpu_percent_for_background: float = 65.0
    max_memory_percent_for_background: float = 82.0


@dataclass(slots=True)
class SafetyConfig:
    auto_allow_read_only: bool = True
    auto_allow_reversible: bool = True
    require_confirmation_for_destructive: bool = True


@dataclass(slots=True)
class GitHubConfig:
    enabled: bool = True
    remote: str = "origin"
    main_branch: str = "main"
    auto_push_candidates: bool = True


@dataclass(slots=True)
class SelfDevConfig:
    enabled: bool = True
    # This is deliberately distinct from model.name. If it is unavailable,
    # LocalPilot falls back to the everyday model for that cycle.
    developer_model: str = "qwen2.5:32b"
    # Ordered fallbacks are considered only when the preferred/everyday model
    # would exceed the background memory ceiling on the current machine.
    developer_model_fallbacks: list[str] = field(default_factory=lambda: ["qwen2.5:14b"])
    # Give repository/tool loops a deliberate context allocation instead of
    # inheriting Ollama's runtime default. 16K is conservative for background
    # Qwen work; owners with more headroom can raise it up to the validated cap.
    context_tokens: int = 16384
    # Model file size is a useful lower-bound estimate for resident memory.
    # Reserve additional space for context/KV cache before starting inference.
    model_memory_overhead_gb: float = 1.0
    # Do not leave a self-development model resident after a response. This
    # lets a paused cycle return RAM/VRAM to the owner immediately.
    ollama_keep_alive: float | str = 0
    auto_promote: bool = False
    research_tool_rounds: int = 6
    max_tool_rounds: int = 14
    local_repair_tool_rounds: int = 6
    max_local_repair_attempts: int = 3
    # Candidate complexity is reported after the soft budget, but only the
    # hard ceiling blocks writes. Directories never consume either budget.
    candidate_file_soft_budget: int = 100
    candidate_file_hard_ceiling: int = 500
    # Deprecated compatibility input. A legacy value of 8 is migrated to the
    # new defaults so an old localpilot.toml cannot silently preserve the
    # framework failure this policy change corrects.
    max_files_per_cycle: int | None = None
    max_zip_members: int = 2000
    max_zip_size_mb: int = 1024
    # Resources live outside the repository under agent.data_dir. Eight GiB is
    # ample for ordinary research datasets/model metadata while remaining a
    # bounded, operator-configurable allocation.
    candidate_resource_quota_gb: float = 8.0
    max_resource_file_mb: int = 512
    run_static_checks: bool = True
    allow_local_candidate_execution: bool = False
    learning_database: str = "learning.sqlite3"
    lesson_limit: int = 6


@dataclass(slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    selfdev: SelfDevConfig = field(default_factory=SelfDevConfig)
    source_path: Path | None = None


def _apply(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def _normalize_model_thinking(cfg: Config) -> None:
    think = cfg.model.think
    model_name = cfg.model.name.lower()
    if "gpt-oss" in model_name:
        # Ollama ignores boolean think values for GPT-OSS. Migrate old configs
        # rather than silently leaving the model at an undefined effort level.
        if think is True:
            cfg.model.think = "high"
        elif think is False:
            cfg.model.think = "low"
        elif isinstance(think, str):
            normalized = think.strip().lower()
            if normalized not in {"low", "medium", "high"}:
                raise ValueError("GPT-OSS model.think must be low, medium, or high")
            cfg.model.think = normalized
        else:
            raise ValueError("GPT-OSS model.think must be low, medium, or high")
        return

    if isinstance(think, str):
        normalized = think.strip().lower()
        if normalized not in {"low", "medium", "high", "max"}:
            raise ValueError("model.think string must be low, medium, high, or max")
        cfg.model.think = normalized
    elif not isinstance(think, bool):
        raise ValueError("model.think must be a boolean or supported reasoning level")


def _validate_context_tokens(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer token count")
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer token count") from exc
    if tokens < 4096 or tokens > 131072:
        raise ValueError(f"{name} must be between 4096 and 131072 tokens")
    return tokens


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    chosen = Path(path) if path else Path(os.environ.get("LOCALPILOT_CONFIG", "localpilot.toml"))
    if chosen.exists():
        with chosen.open("rb") as handle:
            raw = tomllib.load(handle)
        agent_raw = raw.get("agent", {})
        _apply(cfg.agent, agent_raw)
        # Preserve the intent of older configs that only set max_tool_rounds,
        # while giving them a distinct hard ceiling instead of silently using
        # the old value as both advice and termination.
        if "max_tool_rounds" in agent_raw and "research_soft_tool_rounds" not in agent_raw:
            cfg.agent.research_soft_tool_rounds = int(cfg.agent.max_tool_rounds)
        if "max_tool_rounds" in agent_raw and "research_hard_tool_rounds" not in agent_raw:
            soft = max(1, int(cfg.agent.research_soft_tool_rounds))
            cfg.agent.research_hard_tool_rounds = max(soft + 4, soft * 2)
        _apply(cfg.model, raw.get("model", {}))
        _apply(cfg.resource, raw.get("resource", {}))
        _apply(cfg.safety, raw.get("safety", {}))
        _apply(cfg.github, raw.get("github", {}))
        selfdev_raw = raw.get("selfdev", {})
        _apply(cfg.selfdev, selfdev_raw)
        legacy_limit = selfdev_raw.get("max_files_per_cycle")
        if legacy_limit is not None:
            if "candidate_file_hard_ceiling" in selfdev_raw:
                raise ValueError(
                    "Configure candidate_file_hard_ceiling, not both it and legacy max_files_per_cycle"
                )
            legacy_limit = int(legacy_limit)
            cfg.selfdev.candidate_file_hard_ceiling = (
                500 if legacy_limit <= 8 else legacy_limit
            )
        cfg.source_path = chosen.resolve()

    _normalize_model_thinking(cfg)
    cfg.model.context_tokens = _validate_context_tokens(
        "model.context_tokens", cfg.model.context_tokens
    )
    cfg.selfdev.context_tokens = _validate_context_tokens(
        "selfdev.context_tokens", cfg.selfdev.context_tokens
    )
    cfg.agent.research_soft_tool_rounds = int(cfg.agent.research_soft_tool_rounds)
    cfg.agent.research_hard_tool_rounds = int(cfg.agent.research_hard_tool_rounds)
    if cfg.agent.research_soft_tool_rounds < 1:
        raise ValueError("agent.research_soft_tool_rounds must be positive")
    if cfg.agent.research_hard_tool_rounds < cfg.agent.research_soft_tool_rounds:
        raise ValueError(
            "agent.research_hard_tool_rounds must be at least agent.research_soft_tool_rounds"
        )

    # Promotion is a human/repository action, never an autonomous config knob.
    if cfg.selfdev.auto_promote:
        raise ValueError("selfdev.auto_promote cannot be enabled; candidates require review and merge")
    if cfg.selfdev.candidate_file_soft_budget < 1:
        raise ValueError("selfdev.candidate_file_soft_budget must be positive")
    if cfg.selfdev.candidate_file_hard_ceiling < cfg.selfdev.candidate_file_soft_budget:
        raise ValueError(
            "selfdev.candidate_file_hard_ceiling must be at least candidate_file_soft_budget"
        )
    if cfg.selfdev.max_zip_members < 1 or cfg.selfdev.max_zip_size_mb < 1:
        raise ValueError("candidate ZIP limits must be positive")
    if cfg.selfdev.candidate_resource_quota_gb <= 0 or cfg.selfdev.max_resource_file_mb < 1:
        raise ValueError("candidate resource limits must be positive")
    return cfg
