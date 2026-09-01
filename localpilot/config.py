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
    # Keep the everyday model warm through an ordinary work session. Loading a
    # large local model can take minutes; immediate unload turns every short
    # colleague-style exchange into a cold start.
    ollama_keep_alive: float | str = "30m"
    # Optional hybrid durable-memory retrieval. LocalPilot never pulls this
    # model automatically; unavailable embeddings fall back to lexical search.
    memory_embeddings_enabled: bool = False
    memory_embedding_model: str = "embeddinggemma"
    memory_embedding_keep_alive: float | str = "5m"
    memory_semantic_weight: float = 12.0
    memory_semantic_min_similarity: float = 0.2
    memory_embedding_batch_size: int = 64
    memory_embedding_migration_limit: int = 512


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
class DesktopConfig:
    """Loopback-only desktop chat and broker settings."""

    host: str = "127.0.0.1"
    port: int = 8765
    chat_database: str = "chat.sqlite3"
    runtime_restart_limit: int = 5
    request_timeout_seconds: float = 600.0


@dataclass(slots=True)
class LibraryConfig:
    """Owner-managed, read-only source library with a disposable local index."""

    enabled: bool = False
    root: str = r"E:\LLM_HOME\library"
    index_database: str = "library-index.sqlite3"
    max_documents: int = 2000
    max_refresh_files: int = 50
    max_file_size_mb: int = 256
    max_pages_per_document: int = 2000
    max_chars_per_page: int = 30_000
    max_search_results: int = 8


@dataclass(slots=True)
class SystemSenseConfig:
    """Passive, read-only environmental telemetry settings."""

    enabled: bool = True
    database: str = "systemsense.sqlite3"
    sample_interval_seconds: float = 5.0
    inventory_interval_seconds: float = 900.0
    retention_days: int = 30
    baseline_window_hours: int = 24
    correlation_window_days: int = 14
    max_processes: int = 12
    compact_context_enabled: bool = True


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
    # One evolution invocation shares these budgets across discovery, research,
    # implementation, repair, and delivery. Per-stage round limits remain useful
    # but cannot by themselves bound a multi-stage or multi-call response.
    cycle_wall_clock_seconds: float = 900.0
    max_tool_calls_per_cycle: int = 32
    max_web_calls_per_cycle: int = 8
    opportunity_similarity_threshold: float = 0.82
    max_queued_opportunities: int = 48
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
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    systemsense: SystemSenseConfig = field(default_factory=SystemSenseConfig)
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
        _apply(cfg.desktop, raw.get("desktop", {}))
        _apply(cfg.library, raw.get("library", {}))
        _apply(cfg.systemsense, raw.get("systemsense", {}))
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
    if not isinstance(cfg.model.ollama_keep_alive, (int, float, str)) or isinstance(
        cfg.model.ollama_keep_alive, bool
    ):
        raise ValueError("model.ollama_keep_alive must be seconds or an Ollama duration string")
    cfg.selfdev.context_tokens = _validate_context_tokens(
        "selfdev.context_tokens", cfg.selfdev.context_tokens
    )
    if not isinstance(cfg.model.memory_embeddings_enabled, bool):
        raise ValueError("model.memory_embeddings_enabled must be a boolean")
    cfg.model.memory_embedding_model = str(cfg.model.memory_embedding_model).strip()
    if cfg.model.memory_embeddings_enabled and not cfg.model.memory_embedding_model:
        raise ValueError(
            "model.memory_embedding_model is required when memory embeddings are enabled"
        )
    cfg.model.memory_semantic_weight = float(cfg.model.memory_semantic_weight)
    cfg.model.memory_semantic_min_similarity = float(
        cfg.model.memory_semantic_min_similarity
    )
    cfg.model.memory_embedding_batch_size = int(
        cfg.model.memory_embedding_batch_size
    )
    cfg.model.memory_embedding_migration_limit = int(
        cfg.model.memory_embedding_migration_limit
    )
    if cfg.model.memory_semantic_weight < 0:
        raise ValueError("model.memory_semantic_weight cannot be negative")
    if not -1 <= cfg.model.memory_semantic_min_similarity <= 1:
        raise ValueError(
            "model.memory_semantic_min_similarity must be between -1 and 1"
        )
    if not 1 <= cfg.model.memory_embedding_batch_size <= 256:
        raise ValueError(
            "model.memory_embedding_batch_size must be between 1 and 256"
        )
    if not 1 <= cfg.model.memory_embedding_migration_limit <= 5000:
        raise ValueError(
            "model.memory_embedding_migration_limit must be between 1 and 5000"
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
    cfg.selfdev.cycle_wall_clock_seconds = float(cfg.selfdev.cycle_wall_clock_seconds)
    cfg.selfdev.max_tool_calls_per_cycle = int(cfg.selfdev.max_tool_calls_per_cycle)
    cfg.selfdev.max_web_calls_per_cycle = int(cfg.selfdev.max_web_calls_per_cycle)
    cfg.selfdev.opportunity_similarity_threshold = float(
        cfg.selfdev.opportunity_similarity_threshold
    )
    cfg.selfdev.max_queued_opportunities = int(cfg.selfdev.max_queued_opportunities)
    if cfg.selfdev.cycle_wall_clock_seconds < 30:
        raise ValueError("selfdev.cycle_wall_clock_seconds must be at least 30")
    if cfg.selfdev.max_tool_calls_per_cycle < 1:
        raise ValueError("selfdev.max_tool_calls_per_cycle must be positive")
    if not 0 <= cfg.selfdev.max_web_calls_per_cycle <= cfg.selfdev.max_tool_calls_per_cycle:
        raise ValueError(
            "selfdev.max_web_calls_per_cycle must be between zero and max_tool_calls_per_cycle"
        )
    if not 0.5 <= cfg.selfdev.opportunity_similarity_threshold <= 1:
        raise ValueError(
            "selfdev.opportunity_similarity_threshold must be between 0.5 and 1"
        )
    if cfg.selfdev.max_queued_opportunities < 1:
        raise ValueError("selfdev.max_queued_opportunities must be positive")
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
    if cfg.desktop.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("desktop.host must remain loopback-only")
    cfg.desktop.port = int(cfg.desktop.port)
    if not 1 <= cfg.desktop.port <= 65535:
        raise ValueError("desktop.port must be between 1 and 65535")
    desktop_database = Path(str(cfg.desktop.chat_database).strip())
    if (
        not desktop_database.name
        or desktop_database.is_absolute()
        or len(desktop_database.parts) != 1
    ):
        raise ValueError("desktop.chat_database must be one local filename")
    if desktop_database.name.casefold() == Path(cfg.selfdev.learning_database).name.casefold():
        raise ValueError("desktop.chat_database must remain separate from learning_database")
    cfg.desktop.chat_database = desktop_database.name
    cfg.desktop.runtime_restart_limit = int(cfg.desktop.runtime_restart_limit)
    if cfg.desktop.runtime_restart_limit < 1:
        raise ValueError("desktop.runtime_restart_limit must be positive")
    cfg.desktop.request_timeout_seconds = float(cfg.desktop.request_timeout_seconds)
    if not 1 <= cfg.desktop.request_timeout_seconds <= 3600:
        raise ValueError("desktop.request_timeout_seconds must be between 1 and 3600")
    if not isinstance(cfg.library.enabled, bool):
        raise ValueError("library.enabled must be a boolean")
    cfg.library.root = str(cfg.library.root).strip()
    if cfg.library.enabled and not cfg.library.root:
        raise ValueError("library.root is required when the local library is enabled")
    library_database = Path(str(cfg.library.index_database).strip())
    if (
        not library_database.name
        or library_database.is_absolute()
        or len(library_database.parts) != 1
    ):
        raise ValueError("library.index_database must be one local filename")
    reserved_databases = {
        Path(cfg.desktop.chat_database).name.casefold(),
        Path(cfg.selfdev.learning_database).name.casefold(),
    }
    if library_database.name.casefold() in reserved_databases:
        raise ValueError("library.index_database must remain separate from chat and learning databases")
    cfg.library.index_database = library_database.name
    library_bounds = {
        "max_documents": 100_000,
        "max_refresh_files": 2_000,
        "max_file_size_mb": 1_024,
        "max_pages_per_document": 10_000,
        "max_chars_per_page": 200_000,
        "max_search_results": 50,
    }
    for field_name, maximum in library_bounds.items():
        value = int(getattr(cfg.library, field_name))
        if not 1 <= value <= maximum:
            raise ValueError(
                f"library.{field_name} must be between 1 and {maximum}"
            )
        setattr(cfg.library, field_name, value)

    if not isinstance(cfg.systemsense.enabled, bool):
        raise ValueError("systemsense.enabled must be a boolean")
    if not isinstance(cfg.systemsense.compact_context_enabled, bool):
        raise ValueError("systemsense.compact_context_enabled must be a boolean")
    systemsense_database = Path(str(cfg.systemsense.database).strip())
    if (
        not systemsense_database.name
        or systemsense_database.is_absolute()
        or len(systemsense_database.parts) != 1
    ):
        raise ValueError("systemsense.database must be one local filename")
    if systemsense_database.name.casefold() in reserved_databases | {
        library_database.name.casefold()
    }:
        raise ValueError("systemsense.database must remain separate from other databases")
    cfg.systemsense.database = systemsense_database.name
    cfg.systemsense.sample_interval_seconds = float(cfg.systemsense.sample_interval_seconds)
    cfg.systemsense.inventory_interval_seconds = float(
        cfg.systemsense.inventory_interval_seconds
    )
    cfg.systemsense.retention_days = int(cfg.systemsense.retention_days)
    cfg.systemsense.baseline_window_hours = int(cfg.systemsense.baseline_window_hours)
    cfg.systemsense.correlation_window_days = int(
        cfg.systemsense.correlation_window_days
    )
    cfg.systemsense.max_processes = int(cfg.systemsense.max_processes)
    if not 1 <= cfg.systemsense.sample_interval_seconds <= 300:
        raise ValueError("systemsense.sample_interval_seconds must be between 1 and 300")
    if not 60 <= cfg.systemsense.inventory_interval_seconds <= 86_400:
        raise ValueError(
            "systemsense.inventory_interval_seconds must be between 60 and 86400"
        )
    if not 1 <= cfg.systemsense.retention_days <= 3650:
        raise ValueError("systemsense.retention_days must be between 1 and 3650")
    if not 1 <= cfg.systemsense.baseline_window_hours <= 720:
        raise ValueError(
            "systemsense.baseline_window_hours must be between 1 and 720"
        )
    if not 1 <= cfg.systemsense.correlation_window_days <= 3650:
        raise ValueError(
            "systemsense.correlation_window_days must be between 1 and 3650"
        )
    if not 1 <= cfg.systemsense.max_processes <= 50:
        raise ValueError("systemsense.max_processes must be between 1 and 50")
    return cfg
