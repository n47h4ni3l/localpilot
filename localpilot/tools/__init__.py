from __future__ import annotations

from pathlib import Path

from localpilot.safety import RiskLevel, ToolSpec
from localpilot.tools.repository import RepositoryReader
from localpilot.tools.windows import (
    get_active_power_plan,
    get_defender_summary,
    get_device_problem_summary,
    get_startup_items,
    get_storage_summary,
    get_system_summary,
    get_top_processes,
)


def registry(project_root: str | Path | None = None) -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("get_system_summary", "Read Windows, CPU and RAM summary.", RiskLevel.READ_ONLY, get_system_summary),
        ToolSpec("get_storage_summary", "Read local disk capacity and free space.", RiskLevel.READ_ONLY, get_storage_summary),
        ToolSpec("get_top_processes", "Read top processes by CPU and memory.", RiskLevel.READ_ONLY, get_top_processes),
        ToolSpec("get_startup_items", "Read Windows startup entries.", RiskLevel.READ_ONLY, get_startup_items),
        ToolSpec("get_active_power_plan", "Read the active Windows power plan.", RiskLevel.READ_ONLY, get_active_power_plan),
        ToolSpec("get_defender_summary", "Read basic Microsoft Defender protection state.", RiskLevel.READ_ONLY, get_defender_summary),
        ToolSpec("get_device_problem_summary", "Read connected devices with problem states.", RiskLevel.READ_ONLY, get_device_problem_summary),
    ]
    if project_root is not None:
        repository = RepositoryReader(project_root)
        specs.extend(
            [
                ToolSpec(
                    "list_repository_tree",
                    "List a bounded tree of the trusted LocalPilot repository without following symlinks.",
                    RiskLevel.READ_ONLY,
                    repository.list_repository_tree,
                ),
                ToolSpec(
                    "read_repository_file",
                    "Read a bounded line range from a verified UTF-8 file in the trusted LocalPilot repository.",
                    RiskLevel.READ_ONLY,
                    repository.read_repository_file,
                ),
                ToolSpec(
                    "search_repository",
                    "Search trusted repository text for a literal string and return bounded path/line matches.",
                    RiskLevel.READ_ONLY,
                    repository.search_repository,
                ),
                ToolSpec(
                    "inspect_project_dependencies",
                    "Inspect declared Python/build dependencies from the trusted repository's pyproject.toml.",
                    RiskLevel.READ_ONLY,
                    repository.inspect_project_dependencies,
                ),
                ToolSpec(
                    "get_repository_status",
                    "Read current Git branch, HEAD, and working-tree status for the trusted LocalPilot checkout.",
                    RiskLevel.READ_ONLY,
                    repository.get_repository_status,
                ),
            ]
        )
    return {spec.name: spec for spec in specs}
