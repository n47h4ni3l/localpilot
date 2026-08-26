from __future__ import annotations

import inspect
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class OperationRisk(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: list[str]
    risk: OperationRisk
    timeout: float
    action: str = "command"
    wait: bool = True


class CommandRunner:
    """Execute validated argv commands and emit presentation-safe audit events."""

    def __init__(
        self,
        approval_callback: Callable[..., bool] | None = None,
        audit_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.approval_callback = approval_callback
        self.audit_callback = audit_callback

    @staticmethod
    def _validated_argv(command_spec: CommandSpec) -> list[str]:
        if not isinstance(command_spec, CommandSpec):
            raise TypeError("CommandRunner requires a CommandSpec.")
        if not isinstance(command_spec.risk, OperationRisk):
            raise ValueError("CommandSpec risk must be an OperationRisk.")
        if not command_spec.argv or any(
            not isinstance(item, str) or not item for item in command_spec.argv
        ):
            raise ValueError("CommandSpec argv must contain non-empty strings.")
        if not 0.01 <= float(command_spec.timeout) <= 60:
            raise ValueError("CommandSpec timeout must be between 0.01 and 60 seconds.")
        return list(command_spec.argv)

    def _approved(self, command_spec: CommandSpec) -> bool:
        if command_spec.risk is not OperationRisk.DESTRUCTIVE:
            return True
        if self.approval_callback is None:
            return False
        try:
            parameters = inspect.signature(self.approval_callback).parameters
        except (TypeError, ValueError):
            parameters = {"command_spec": None}
        return bool(
            self.approval_callback(command_spec)
            if parameters
            else self.approval_callback()
        )

    def _audit(self, command_spec: CommandSpec, **payload: Any) -> None:
        if self.audit_callback is None:
            return
        self.audit_callback(
            {
                "action": command_spec.action,
                "risk": command_spec.risk.value,
                "executable": command_spec.argv[0],
                "argument_count": max(0, len(command_spec.argv) - 1),
                **payload,
            }
        )

    @staticmethod
    def _text(value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def run(self, command_spec: CommandSpec) -> dict[str, Any]:
        argv = self._validated_argv(command_spec)
        if not self._approved(command_spec):
            self._audit(command_spec, status="denied", returncode=None, duration_ms=0)
            raise RuntimeError("Destructive operation not approved")

        started = time.monotonic()
        if not command_spec.wait:
            try:
                process = subprocess.Popen(
                    argv,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                self._audit(
                    command_spec,
                    status="failed_to_start",
                    returncode=None,
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                )
                raise RuntimeError(
                    f"Could not start approved action {command_spec.action}."
                ) from exc
            duration_ms = int((time.monotonic() - started) * 1000)
            self._audit(
                command_spec,
                status="started",
                returncode=None,
                duration_ms=duration_ms,
                process_id=process.pid,
            )
            return {
                "action": command_spec.action,
                "risk": command_spec.risk.value,
                "status": "started",
                "process_id": process.pid,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        try:
            result = subprocess.run(
                argv,
                timeout=command_spec.timeout,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            self._audit(
                command_spec,
                status="timed_out",
                returncode=None,
                duration_ms=duration_ms,
            )
            return {
                "action": command_spec.action,
                "risk": command_spec.risk.value,
                "status": "timed_out",
                "stdout": self._text(exc.stdout),
                "stderr": self._text(exc.stderr),
                "returncode": None,
            }

        duration_ms = int((time.monotonic() - started) * 1000)
        status = "succeeded" if result.returncode == 0 else "failed"
        self._audit(
            command_spec,
            status=status,
            returncode=result.returncode,
            duration_ms=duration_ms,
        )
        return {
            "action": command_spec.action,
            "risk": command_spec.risk.value,
            "status": status,
            "stdout": self._text(result.stdout),
            "stderr": self._text(result.stderr),
            "returncode": result.returncode,
        }
