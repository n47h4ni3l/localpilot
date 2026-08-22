from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass

import psutil

from localpilot.config import ResourceConfig


@dataclass(slots=True)
class ResourceState:
    idle_seconds: float
    cpu_percent: float
    memory_percent: float
    background_allowed: bool
    reason: str
    idle_allowed: bool = True
    capacity_allowed: bool = True
    idle_reason: str = ""
    capacity_reason: str = ""

    def allows_selfdev(self, *, ignore_idle: bool = False) -> bool:
        """Manual force may skip the idle wait, never hardware protection."""
        return self.capacity_allowed and (ignore_idle or self.idle_allowed)

    def blocking_reason(self, *, ignore_idle: bool = False) -> str:
        reasons: list[str] = []
        if not ignore_idle and self.idle_reason:
            reasons.append(self.idle_reason)
        if self.capacity_reason:
            reasons.append(self.capacity_reason)
        return "; ".join(reasons) or "idle capacity available"


def windows_idle_seconds() -> float:
    """Return seconds since last keyboard/mouse input on Windows."""
    if os.name != "nt":
        return float("inf")

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    tick = ctypes.windll.kernel32.GetTickCount()
    return max(0.0, (tick - info.dwTime) / 1000.0)


class ResourceGovernor:
    """Keeps LocalPilot polite when the PC is in active use."""

    def __init__(self, config: ResourceConfig) -> None:
        self.config = config

    def sample(self, *, interval: float = 0.15) -> ResourceState:
        idle = windows_idle_seconds()
        cpu = psutil.cpu_percent(interval=interval)
        memory = psutil.virtual_memory().percent
        idle_allowed = idle >= self.config.background_idle_seconds
        idle_reason = ""
        capacity_reasons: list[str] = []
        if not idle_allowed:
            idle_reason = f"user idle {idle:.0f}s < {self.config.background_idle_seconds}s"
        if cpu > self.config.max_cpu_percent_for_background:
            capacity_reasons.append(f"CPU {cpu:.0f}% > {self.config.max_cpu_percent_for_background:.0f}%")
        if memory > self.config.max_memory_percent_for_background:
            capacity_reasons.append(f"memory {memory:.0f}% > {self.config.max_memory_percent_for_background:.0f}%")
        capacity_reason = "; ".join(capacity_reasons)
        capacity_allowed = not capacity_reasons
        allowed = idle_allowed and capacity_allowed
        reasons = "; ".join(item for item in (idle_reason, capacity_reason) if item)
        return ResourceState(
            idle,
            cpu,
            memory,
            allowed,
            reasons or "idle capacity available",
            idle_allowed,
            capacity_allowed,
            idle_reason,
            capacity_reason,
        )

    def apply_process_priority(self, idle: bool) -> None:
        if os.name != "nt":
            return
        process = psutil.Process()
        requested = self.config.idle_priority if idle else self.config.active_priority
        mapping = {
            "idle": psutil.IDLE_PRIORITY_CLASS,
            "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        }
        priority = mapping.get(requested.lower())
        if priority is not None:
            try:
                process.nice(priority)
            except (psutil.AccessDenied, OSError):
                pass

    def wait_until_background_allowed(self, poll_seconds: float = 5.0) -> ResourceState:
        while True:
            state = self.sample()
            self.apply_process_priority(idle=state.background_allowed)
            if state.background_allowed:
                return state
            time.sleep(poll_seconds)
