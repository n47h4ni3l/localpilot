import os
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Literal

from localpilot.operator import CommandRunner, CommandSpec, OperationRisk


WindowsApp = Literal["calculator", "file_explorer", "notepad", "task_manager"]
WindowsSettingsPage = Literal[
    "bluetooth",
    "display",
    "network",
    "power",
    "windows_update",
]
PowerPlan = Literal["balanced", "high_performance", "power_saver"]

_APPS: dict[str, tuple[str, ...]] = {
    "calculator": ("calc.exe",),
    "file_explorer": ("explorer.exe",),
    "notepad": ("notepad.exe",),
    "task_manager": ("taskmgr.exe",),
}
_SETTINGS: dict[str, str] = {
    "bluetooth": "ms-settings:bluetooth",
    "display": "ms-settings:display",
    "network": "ms-settings:network-status",
    "power": "ms-settings:powersleep",
    "windows_update": "ms-settings:windowsupdate",
}
_POWER_PLANS: dict[str, str] = {
    "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "high_performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
    "power_saver": "a1841308-3541-4fab-bc81-f71556f20b4a",
}
_GUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


@dataclass(frozen=True, slots=True)
class _PowerRollback:
    previous_guid: str
    target_guid: str
    target_plan: str


class WindowsActions:
    """Small reversible Windows UI actions with fixed executables and arguments."""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner
        self._power_rollbacks: dict[str, _PowerRollback] = {}
        self._power_lock = threading.Lock()

    @staticmethod
    def _require_windows() -> None:
        if os.name != "nt":
            raise RuntimeError("Windows operator actions are available only on Windows.")

    def open_windows_app(self, app: WindowsApp) -> dict:
        """Open one allow-listed Windows app; close its window to reverse the action."""
        self._require_windows()
        argv = _APPS.get(str(app))
        if argv is None:
            raise ValueError(f"Unsupported Windows app: {app}")
        return self.runner.run(
            CommandSpec(
                argv=list(argv),
                risk=OperationRisk.REVERSIBLE,
                timeout=10,
                action=f"open_windows_app:{app}",
                wait=False,
            )
        )

    def open_windows_settings(self, page: WindowsSettingsPage) -> dict:
        """Open one allow-listed Windows Settings page without changing a setting."""
        self._require_windows()
        uri = _SETTINGS.get(str(page))
        if uri is None:
            raise ValueError(f"Unsupported Windows Settings page: {page}")
        return self.runner.run(
            CommandSpec(
                argv=["explorer.exe", uri],
                risk=OperationRisk.REVERSIBLE,
                timeout=10,
                action=f"open_windows_settings:{page}",
                wait=False,
            )
        )

    def _powercfg(self, *arguments: str, risk: OperationRisk, action: str) -> str:
        result = self.runner.run(
            CommandSpec(
                argv=["powercfg.exe", *arguments],
                risk=risk,
                timeout=15,
                action=action,
            )
        )
        if result.get("status") != "succeeded" or result.get("returncode") != 0:
            raise RuntimeError(f"Windows power action failed: {action}")
        return str(result.get("stdout") or "")

    def _active_power_guid(self) -> str:
        output = self._powercfg(
            "/GETACTIVESCHEME",
            risk=OperationRisk.READ_ONLY,
            action="get_active_power_plan_for_change",
        )
        match = _GUID.search(output)
        if match is None:
            raise RuntimeError("Windows did not return an active power plan GUID.")
        return match.group(0).lower()

    def _available_power_guids(self) -> set[str]:
        output = self._powercfg(
            "/LIST",
            risk=OperationRisk.READ_ONLY,
            action="list_power_plans_for_change",
        )
        return {match.group(0).lower() for match in _GUID.finditer(output)}

    def _set_power_guid(self, guid: str, *, action: str) -> None:
        if _GUID.fullmatch(guid) is None:
            raise ValueError("Power plan GUID did not pass strict validation.")
        self._powercfg(
            "/SETACTIVE",
            guid,
            risk=OperationRisk.REVERSIBLE,
            action=action,
        )

    def set_active_power_plan(self, plan: PowerPlan) -> dict:
        """Set an installed built-in plan and return a verified one-use rollback token."""
        self._require_windows()
        target_guid = _POWER_PLANS.get(str(plan))
        if target_guid is None:
            raise ValueError(f"Unsupported Windows power plan: {plan}")
        previous_guid = self._active_power_guid()
        available = self._available_power_guids()
        if target_guid not in available:
            raise RuntimeError(f"The Windows power plan is not installed: {plan}")
        if previous_guid == target_guid:
            return {
                "action": "set_active_power_plan",
                "risk": OperationRisk.REVERSIBLE.value,
                "status": "no_change",
                "active_plan": plan,
                "active_guid": target_guid,
                "rollback_token": None,
            }

        try:
            self._set_power_guid(
                target_guid,
                action=f"set_active_power_plan:{plan}",
            )
            verified_guid = self._active_power_guid()
            if verified_guid != target_guid:
                raise RuntimeError("Windows did not activate the requested power plan.")
        except Exception as exc:
            try:
                current_guid = self._active_power_guid()
                if current_guid != previous_guid and previous_guid in available:
                    self._set_power_guid(
                        previous_guid,
                        action="automatic_power_plan_rollback",
                    )
                if self._active_power_guid() != previous_guid:
                    raise RuntimeError("Automatic power-plan rollback verification failed.")
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Power-plan change failed and automatic rollback could not be verified."
                ) from rollback_exc
            raise RuntimeError(
                "Power-plan change failed; the prior plan was restored and verified."
            ) from exc

        token = secrets.token_urlsafe(24)
        with self._power_lock:
            # Any older token is stale after a later verified plan transition.
            self._power_rollbacks.clear()
            self._power_rollbacks[token] = _PowerRollback(
                previous_guid=previous_guid,
                target_guid=target_guid,
                target_plan=str(plan),
            )
        return {
            "action": "set_active_power_plan",
            "risk": OperationRisk.REVERSIBLE.value,
            "status": "changed",
            "active_plan": plan,
            "active_guid": target_guid,
            "previous_guid": previous_guid,
            "rollback_token": token,
        }

    def restore_power_plan(self, rollback_token: str) -> dict:
        """Use a one-time session token to restore the exact prior active plan."""
        self._require_windows()
        with self._power_lock:
            rollback = self._power_rollbacks.get(str(rollback_token))
        if rollback is None:
            raise ValueError("Unknown or already-used power-plan rollback token.")

        current_guid = self._active_power_guid()
        if current_guid == rollback.previous_guid:
            with self._power_lock:
                self._power_rollbacks.pop(str(rollback_token), None)
            return {
                "action": "restore_power_plan",
                "risk": OperationRisk.REVERSIBLE.value,
                "status": "already_restored",
                "active_guid": rollback.previous_guid,
            }
        if current_guid != rollback.target_guid:
            raise RuntimeError(
                "Rollback refused because the active plan changed after this token was issued."
            )
        if rollback.previous_guid not in self._available_power_guids():
            raise RuntimeError("Rollback refused because the prior plan is no longer installed.")

        self._set_power_guid(
            rollback.previous_guid,
            action=f"restore_power_plan:{rollback.target_plan}",
        )
        if self._active_power_guid() != rollback.previous_guid:
            raise RuntimeError("Windows did not restore the prior power plan.")
        with self._power_lock:
            self._power_rollbacks.pop(str(rollback_token), None)
        return {
            "action": "restore_power_plan",
            "risk": OperationRisk.REVERSIBLE.value,
            "status": "restored",
            "active_guid": rollback.previous_guid,
        }
