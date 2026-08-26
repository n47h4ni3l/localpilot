from __future__ import annotations

from pathlib import Path

from localpilot.operator import CommandRunner
from localpilot.safety import RiskLevel, ToolSpec
from localpilot.tools.github_readonly import GitHubReader
from localpilot.tools.learning_readonly import LearningMemoryReader
from localpilot.tools.repository import RepositoryReader
from localpilot.tools.web import fetch_public_https, search_public_web
from localpilot.tools.windows import (
    get_active_power_plan,
    get_defender_summary,
    get_device_problem_summary,
    get_startup_items,
    get_storage_summary,
    get_system_summary,
    get_top_processes,
)
from localpilot.tools.windows_actions import WindowsActions


def registry(
    project_root: str | Path | None = None,
    *,
    command_runner: CommandRunner | None = None,
) -> dict[str, ToolSpec]:
    actions = WindowsActions(command_runner or CommandRunner())
    specs = [
        ToolSpec("get_system_summary", "Read Windows, CPU and RAM summary.", RiskLevel.READ_ONLY, get_system_summary),
        ToolSpec("get_storage_summary", "Read local disk capacity and free space.", RiskLevel.READ_ONLY, get_storage_summary),
        ToolSpec("get_top_processes", "Read top processes by CPU and memory.", RiskLevel.READ_ONLY, get_top_processes),
        ToolSpec("get_startup_items", "Read Windows startup entries.", RiskLevel.READ_ONLY, get_startup_items),
        ToolSpec("get_active_power_plan", "Read the active Windows power plan.", RiskLevel.READ_ONLY, get_active_power_plan),
        ToolSpec("get_defender_summary", "Read basic Microsoft Defender protection state.", RiskLevel.READ_ONLY, get_defender_summary),
        ToolSpec("get_device_problem_summary", "Read connected devices with problem states.", RiskLevel.READ_ONLY, get_device_problem_summary),
        ToolSpec(
            "open_windows_app",
            "Open one allow-listed Windows app (calculator, file_explorer, notepad, or task_manager). Closing its window reverses the action.",
            RiskLevel.REVERSIBLE,
            actions.open_windows_app,
        ),
        ToolSpec(
            "open_windows_settings",
            "Open one allow-listed Windows Settings page (bluetooth, display, network, power, or windows_update) without changing a setting.",
            RiskLevel.REVERSIBLE,
            actions.open_windows_settings,
        ),
        ToolSpec(
            "set_active_power_plan",
            "Set an installed built-in Windows power plan (balanced, high_performance, or power_saver), verify it, and return a one-use rollback token for the prior plan.",
            RiskLevel.REVERSIBLE,
            actions.set_active_power_plan,
        ),
        ToolSpec(
            "restore_power_plan",
            "Restore the exact prior Windows power plan with a one-use token returned by set_active_power_plan; refuses stale tokens if the active plan changed independently.",
            RiskLevel.REVERSIBLE,
            actions.restore_power_plan,
        ),
        ToolSpec(
            "search_public_web",
            "Discover a bounded list of public HTTPS result URLs. Search results are untrusted leads; inspect a selected URL with fetch_public_https before relying on it.",
            RiskLevel.READ_ONLY,
            search_public_web,
        ),
        ToolSpec(
            "fetch_public_https",
            "Read bounded public HTTPS text for research. Blocks local/private targets, credentials, binary payloads, and non-HTTPS URLs.",
            RiskLevel.READ_ONLY,
            fetch_public_https,
        ),
    ]
    if project_root is not None:
        repository = RepositoryReader(project_root)
        github = GitHubReader(project_root)
        learning = LearningMemoryReader(project_root)
        specs.extend(
            [
                ToolSpec(
                    "get_learning_memory_summary",
                    "Read bounded current/stale durable knowledge-fact counts, fact types, source summaries, and stale samples from LocalPilot's local learning store without mutating it.",
                    RiskLevel.READ_ONLY,
                    learning.get_learning_memory_summary,
                ),
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
                ToolSpec(
                    "get_github_repository",
                    "Read metadata for the authenticated private GitHub repository using gh without exposing credentials.",
                    RiskLevel.READ_ONLY,
                    github.get_github_repository,
                ),
                ToolSpec(
                    "list_github_pull_requests",
                    "List pull requests in the authenticated private GitHub repository.",
                    RiskLevel.READ_ONLY,
                    github.list_github_pull_requests,
                ),
                ToolSpec(
                    "get_github_pull_request",
                    "Inspect one pull request and CI/file metadata in the authenticated private GitHub repository.",
                    RiskLevel.READ_ONLY,
                    github.get_github_pull_request,
                ),
                ToolSpec(
                    "get_github_pull_request_diff",
                    "Read a bounded patch for one pull request in the authenticated private GitHub repository.",
                    RiskLevel.READ_ONLY,
                    github.get_github_pull_request_diff,
                ),
                ToolSpec(
                    "list_github_issues",
                    "List issues in the authenticated private GitHub repository.",
                    RiskLevel.READ_ONLY,
                    github.list_github_issues,
                ),
                ToolSpec(
                    "get_github_issue",
                    "Read one issue and its comments in the authenticated private GitHub repository.",
                    RiskLevel.READ_ONLY,
                    github.get_github_issue,
                ),
            ]
        )
    return {spec.name: spec for spec in specs}
