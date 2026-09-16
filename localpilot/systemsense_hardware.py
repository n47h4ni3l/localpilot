from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from localpilot.process import hidden_process_creation_flags


_HELPER_NAME = "LocalPilot.SystemSense.HardwareProvider.exe"
_DEFAULT_TIMEOUT_SECONDS = 5.0
_legacy_collector_factory: Callable[..., Any] | None = None


def _runtime_id() -> str:
    machine = platform.machine().casefold()
    if machine in {"arm64", "aarch64"}:
        return "win-arm64"
    if machine in {"x86", "i386", "i686"}:
        return "win-x86"
    return "win-x64"


def bundled_helper_path() -> Path:
    override = os.environ.get("LOCALPILOT_HARDWARE_PROVIDER", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (
        Path(__file__).resolve().parent
        / "_hardware"
        / _runtime_id()
        / _HELPER_NAME
    )


class BundledHardwareMonitorCollector:
    """Read sensors from LocalPilot's bundled LibreHardwareMonitorLib helper.

    The helper is a separate read-only process so low-level hardware probing is
    isolated from the LocalPilot runtime. One helper stays alive across samples;
    the process exits automatically when its stdin closes with the parent.
    """

    def __init__(
        self,
        helper_path: str | Path | None = None,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.helper_path = (
            Path(helper_path).expanduser().resolve()
            if helper_path is not None
            else bundled_helper_path()
        )
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[tuple[int, str]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def _reader(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self._responses.put((int(process.pid), line))
        except (OSError, ValueError):
            return

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                    process.wait(timeout=1.0)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        with self._lock:
            self._stop_process()

    def _start_process(self) -> tuple[subprocess.Popen[str] | None, list[str]]:
        process = self._process
        if process is not None and process.poll() is None:
            return process, []
        self._stop_process()

        if os.name != "nt":
            return None, ["bundled-provider:windows-only"]
        if not self.helper_path.is_file():
            return None, [f"bundled-provider:not-installed:{self.helper_path}"]

        try:
            process = self._popen_factory(
                [str(self.helper_path), "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, [f"bundled-provider:start:{type(exc).__name__}"]

        self._process = process
        self._reader_thread = threading.Thread(
            target=self._reader,
            args=(process,),
            name="localpilot-hardware-provider-reader",
            daemon=True,
        )
        self._reader_thread.start()
        return process, []

    @staticmethod
    def _normalize_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "source": "LibreHardwareMonitorLib",
                "available": False,
                "sensors": [],
                "errors": ["bundled-provider:invalid-payload"],
            }
        sensors = payload.get("sensors")
        if not isinstance(sensors, list):
            sensors = []
        errors = [str(item)[:300] for item in (payload.get("errors") or [])]
        if not payload.get("ok") and payload.get("error"):
            errors.append(
                "bundled-provider:"
                + str(payload.get("stage") or "runtime")
                + ":"
                + str(payload.get("error"))[:200]
            )
        return {
            "source": str(payload.get("source") or "LibreHardwareMonitorLib"),
            "provider_version": payload.get("provider_version"),
            "available": bool(payload.get("ok")) and bool(sensors),
            "sensors": sensors,
            "errors": errors,
        }

    def collect(self) -> dict[str, Any]:
        with self._lock:
            process, errors = self._start_process()
            if process is None:
                return {
                    "source": "LibreHardwareMonitorLib",
                    "available": False,
                    "sensors": [],
                    "errors": errors,
                }
            if process.stdin is None:
                self._stop_process()
                return {
                    "source": "LibreHardwareMonitorLib",
                    "available": False,
                    "sensors": [],
                    "errors": ["bundled-provider:no-stdin"],
                }

            try:
                process.stdin.write("snapshot\n")
                process.stdin.flush()
            except (OSError, ValueError):
                self._stop_process()
                return {
                    "source": "LibreHardwareMonitorLib",
                    "available": False,
                    "sensors": [],
                    "errors": ["bundled-provider:write-failed"],
                }

            deadline = time.monotonic() + self.timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_process()
                    return {
                        "source": "LibreHardwareMonitorLib",
                        "available": False,
                        "sensors": [],
                        "errors": ["bundled-provider:timeout"],
                    }
                try:
                    pid, line = self._responses.get(timeout=remaining)
                except queue.Empty:
                    continue
                if pid != int(process.pid):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    self._stop_process()
                    return {
                        "source": "LibreHardwareMonitorLib",
                        "available": False,
                        "sensors": [],
                        "errors": ["bundled-provider:invalid-json"],
                    }
                return self._normalize_payload(payload)


class SystemSenseHardwareCollector:
    """Prefer LocalPilot's bundled provider and preserve legacy WMI fallbacks."""

    def __init__(
        self,
        wmi: Any = None,
        *,
        bundled_collector: BundledHardwareMonitorCollector | Any | None = None,
    ) -> None:
        if _legacy_collector_factory is None:
            raise RuntimeError("SystemSense hardware provider was not installed")
        self.bundled = bundled_collector or BundledHardwareMonitorCollector()
        self.fallback = _legacy_collector_factory(wmi)

    def collect(self) -> dict[str, Any]:
        primary = self.bundled.collect()
        if primary.get("available"):
            return primary

        fallback = self.fallback.collect()
        combined_errors = list(primary.get("errors") or []) + list(
            fallback.get("errors") or []
        )
        if fallback.get("available"):
            result = dict(fallback)
            result["errors"] = combined_errors
            result["fallback_from"] = primary.get("source") or "LibreHardwareMonitorLib"
            return result
        return {
            "source": primary.get("source") or fallback.get("source") or "hardware-sensors",
            "available": False,
            "sensors": [],
            "errors": combined_errors,
        }

    def close(self) -> None:
        close = getattr(self.bundled, "close", None)
        if callable(close):
            close()


def install_bundled_hardware_provider() -> None:
    """Install the bundled-first collector without changing existing call sites."""

    global _legacy_collector_factory
    from localpilot import systemsense_collectors

    current = systemsense_collectors.LibreHardwareMonitorCollector
    if current is SystemSenseHardwareCollector:
        return
    _legacy_collector_factory = current
    systemsense_collectors.LibreHardwareMonitorCollector = SystemSenseHardwareCollector
