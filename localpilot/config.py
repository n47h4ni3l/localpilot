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
    max_tool_rounds: int = 12


@dataclass(slots=True)
class ModelConfig:
    """The everyday operator model. Self-development has its own model."""

    provider: str = "ollama"
    name: str = "gpt-oss:20b"
    think: bool = True
    temperature: float = 0.1


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
    auto_promote: bool = False
    research_tool_rounds: int = 6
    max_tool_rounds: int = 14
    local_repair_tool_rounds: int = 6
    max_local_repair_attempts: int = 3
    max_files_per_cycle: int = 8
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


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    chosen = Path(path) if path else Path(os.environ.get("LOCALPILOT_CONFIG", "localpilot.toml"))
    if chosen.exists():
        with chosen.open("rb") as handle:
            raw = tomllib.load(handle)
        _apply(cfg.agent, raw.get("agent", {}))
        _apply(cfg.model, raw.get("model", {}))
        _apply(cfg.resource, raw.get("resource", {}))
        _apply(cfg.safety, raw.get("safety", {}))
        _apply(cfg.github, raw.get("github", {}))
        _apply(cfg.selfdev, raw.get("selfdev", {}))
        cfg.source_path = chosen.resolve()

    # Promotion is a human/repository action, never an autonomous config knob.
    if cfg.selfdev.auto_promote:
        raise ValueError("selfdev.auto_promote cannot be enabled; candidates require review and merge")
    return cfg

