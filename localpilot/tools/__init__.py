from __future__ import annotations

from localpilot.safety import RiskLevel, ToolSpec
from localpilot.tools.windows import (
    get_active_power_plan,
    get_defender_summary,
    get_device_problem_summary,
    get_startup_items,
    get_storage_summary,
    get_system_summary,
    get_top_processes,
)


def registry() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("get_system_summary", "Read Windows, CPU and RAM summary.", RiskLevel.READ_ONLY, get_system_summary),
        ToolSpec("get_storage_summary", "Read local disk capacity and free space.", RiskLevel.READ_ONLY, get_storage_summary),
        ToolSpec("get_top_processes", "Read top processes by CPU and memory.", RiskLevel.READ_ONLY, get_top_processes),
        ToolSpec("get_startup_items", "Read Windows startup entries.", RiskLevel.READ_ONLY, get_startup_items),
        ToolSpec("get_active_power_plan", "Read the active Windows power plan.", RiskLevel.READ_ONLY, get_active_power_plan),
        ToolSpec("get_defender_summary", "Read basic Microsoft Defender protection state.", RiskLevel.READ_ONLY, get_defender_summary),
        ToolSpec("get_device_problem_summary", "Read connected devices with problem states.", RiskLevel.READ_ONLY, get_device_problem_summary),
    ]
    return {spec.name: spec for spec in specs}
