"""Opt-in elevated Windows login startup, scoped to this user and checkout.

Registration requires an already elevated caller. It never prompts for UAC,
starts a desktop, or stops an existing one. Legacy shortcuts are removed only
after the replacement task has been read back and verified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from localpilot.config import load_config
from localpilot.process import current_process_elevated, hidden_process_creation_flags


_TASK_FUNCTIONS = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
function Get-IdentitySid {
    return [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
}
function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Get-StartupTask {
    return Get-ScheduledTask -TaskPath '\' -ErrorAction Stop |
        Where-Object { $_.TaskName -ceq $taskName }
}
function Assert-OwnedTask($task) {
    if (-not $task) { return }
    $actions = @($task.Actions)
    if ($task.Description -cne $description -or $actions.Count -ne 1 -or
        $actions[0].WorkingDirectory -ine $root -or
        $actions[0].Arguments -notmatch '^-m localpilot\.cli --config .+ desktop$' -or
        $task.Principal.UserId -ine $sid) {
        throw 'A different task occupies this LocalPilot startup name; it was preserved.'
    }
}
function Assert-RegisteredTask($task) {
    Assert-OwnedTask $task
    $actions = @($task.Actions)
    $triggers = @($task.Triggers)
    if (-not $task -or $actions[0].Execute -ine $executable -or
        $actions[0].Arguments -cne $arguments -or
        $task.Principal.RunLevel -ne 'Highest' -or
        $task.Principal.LogonType -ne 'Interactive' -or
        -not $task.Settings.Enabled -or $task.Settings.MultipleInstances -ne 'IgnoreNew' -or
        $triggers.Count -ne 1 -or $triggers[0].CimClass.CimClassName -ne 'MSFT_TaskLogonTrigger' -or
        $triggers[0].UserId -ine $sid) {
        throw 'The replacement login task did not verify; the previous setup will be restored.'
    }
}
function Get-LegacyShortcut {
    if (-not (Test-Path -LiteralPath $shortcut -PathType Leaf)) { return $null }
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($shortcut)
    return @{ target = [string]$link.TargetPath; arguments = [string]$link.Arguments;
              root = [string]$link.WorkingDirectory }
}
function Remove-VerifiedLegacyShortcut {
    if (-not $legacyExpected) { return }
    $link = Get-LegacyShortcut
    if (-not $link) { return }
    if ($link.target -ine $legacyExpected.target -or
        $link.root -ine $legacyExpected.root -or
        $link.arguments -cne $legacyExpected.arguments) {
        throw 'The legacy startup shortcut changed during setup; it was preserved.'
    }
    Remove-Item -LiteralPath $shortcut -ErrorAction Stop
}
"""

_TASK_ACTION = r"""
$operation = $env:LOCALPILOT_STARTUP_OPERATION
$root = $env:LOCALPILOT_STARTUP_ROOT
$executable = $env:LOCALPILOT_STARTUP_EXECUTABLE
$arguments = $env:LOCALPILOT_STARTUP_ARGUMENTS
$shortcut = $env:LOCALPILOT_STARTUP_SHORTCUT
$legacyExpected = $null
if ($env:LOCALPILOT_STARTUP_LEGACY) {
    $legacyExpected = $env:LOCALPILOT_STARTUP_LEGACY | ConvertFrom-Json
}
$sid = Get-IdentitySid
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $userHash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($sid)))).Replace('-', '').Substring(0, 12)
} finally { $sha.Dispose() }
$taskName = 'LocalPilot Desktop - ' + $env:LOCALPILOT_STARTUP_ROOT_HASH + '-' + $userHash
$description = 'LocalPilot desktop login v1; root=' + $env:LOCALPILOT_STARTUP_ROOT_HASH + '; user=' + $sid
$task = Get-StartupTask
Assert-OwnedTask $task
if ($operation -eq 'status') {
    @{ enabled = [bool]($task -and $task.Settings.Enabled); legacy = (Get-LegacyShortcut) } |
        ConvertTo-Json -Compress -Depth 4
    exit 0
}
if ($operation -notin @('enable', 'disable')) { throw 'Unknown login startup operation.' }
if (-not (Test-Administrator)) { throw 'Reopen LocalPilot as administrator to change login startup.' }
if ($operation -eq 'disable') {
    if ($task) { Disable-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction Stop | Out-Null }
    Remove-VerifiedLegacyShortcut
    @{ enabled = $false } | ConvertTo-Json -Compress
    exit 0
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw 'The selected desktop Python no longer exists.' }
$previousXml = $null
$previousDisabled = $false
if ($task) {
    $previousXml = Export-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction Stop
    $previousDisabled = -not $task.Settings.Enabled
}
$registrationAttempted = $false
try {
    $action = New-ScheduledTaskAction -Execute $executable -Argument $arguments -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $sid
    $principal = New-ScheduledTaskPrincipal -UserId $sid -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $registrationAttempted = $true
    Register-ScheduledTask -TaskName $taskName -TaskPath '\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $description -Force -ErrorAction Stop | Out-Null
    Assert-RegisteredTask (Get-StartupTask)
    Remove-VerifiedLegacyShortcut
} catch {
    if ($registrationAttempted) {
        if ($previousXml) {
            Register-ScheduledTask -TaskName $taskName -TaskPath '\' -Xml $previousXml -Force -ErrorAction Stop | Out-Null
            if ($previousDisabled) { Disable-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction Stop | Out-Null }
        } else {
            $replacement = Get-StartupTask
            Assert-OwnedTask $replacement
            if ($replacement) { Unregister-ScheduledTask -TaskName $taskName -TaskPath '\' -Confirm:$false -ErrorAction Stop }
        }
    }
    throw
}
@{ enabled = $true } | ConvertTo-Json -Compress
"""


def startup_shortcut_path() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft/Windows/Start Menu/Programs/Startup/LocalPilot.lnk"


def _effective_config(root: Path, config_path: str | Path | None) -> Path:
    chosen = Path(config_path or os.environ.get("LOCALPILOT_CONFIG", "localpilot.toml"))
    return (chosen if chosen.is_absolute() else root / chosen).resolve()


def _pythonw(executable: str | Path | None) -> Path:
    selected = Path(executable or sys.executable).resolve()
    sibling = selected.with_name("pythonw.exe")
    return sibling if sibling.is_file() else selected


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def _owned_legacy(link: Any, root: Path, config_path: Path) -> bool:
    if not isinstance(link, dict) or not link.get("root") or not link.get("target"):
        return False
    if not _same_path(link["root"], root) or Path(link["target"]).name.lower() not in {"python.exe", "pythonw.exe"}:
        return False
    options = [[], ["--config", str(config_path)]]
    accepted = {
        subprocess.list2cmdline(["-m", "localpilot.native_avatar_companion", "--root", str(root), *option])
        for option in options
    }
    accepted.add(subprocess.list2cmdline(["-m", "localpilot.cli", "--config", str(config_path), "desktop"]))
    return link.get("arguments") in accepted


def _operation(
    operation: str, root: Path, config_path: str | Path | None, *,
    python_executable: str | Path | None = None, legacy: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = _effective_config(root, config_path)
    environment = os.environ.copy()
    environment.update({
        "LOCALPILOT_STARTUP_OPERATION": operation,
        "LOCALPILOT_STARTUP_ROOT": str(root),
        "LOCALPILOT_STARTUP_ROOT_HASH": hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:16],
        "LOCALPILOT_STARTUP_EXECUTABLE": str(_pythonw(python_executable)),
        "LOCALPILOT_STARTUP_ARGUMENTS": subprocess.list2cmdline(["-m", "localpilot.cli", "--config", str(config), "desktop"]),
        "LOCALPILOT_STARTUP_SHORTCUT": str(startup_shortcut_path()),
        "LOCALPILOT_STARTUP_LEGACY": json.dumps(legacy) if legacy else "",
    })
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _TASK_FUNCTIONS + _TASK_ACTION],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=environment, timeout=30, creationflags=hidden_process_creation_flags(),
    )
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or not isinstance(result.get("enabled"), bool):
        raise ValueError("Windows returned an invalid login startup status.")
    return result


def startup_status(root: Path, config_path: str | Path | None) -> dict[str, Any]:
    if os.name != "nt":
        return {"ok": True, "enabled": False}
    result = _operation("status", root, config_path)
    legacy = _owned_legacy(result.get("legacy"), root.resolve(), _effective_config(root, config_path))
    return {"ok": True, "enabled": result["enabled"] or legacy}


def set_startup(root: Path, config_path: str | Path | None, enabled: bool) -> dict[str, Any]:
    if os.name != "nt":
        return {"ok": False, "reason": "not-windows"}
    if current_process_elevated() is not True:
        return {"ok": False, "reason": "Reopen LocalPilot as administrator to change login startup."}
    status = _operation("status", root, config_path)
    legacy = status.get("legacy")
    if not _owned_legacy(legacy, root.resolve(), _effective_config(root, config_path)):
        legacy = None
    result = _operation("enable" if enabled else "disable", root, config_path, legacy=legacy)
    return {"ok": True, "enabled": result["enabled"]}


def validate_legacy_install(
    root: Path, config_path: str | Path | None, legacy_root: str | Path,
) -> Path:
    """Resolve an explicitly supplied old install sharing the same physical data.

    Require both real config files so a missing alias cannot silently inherit
    defaults and authorize stopping a different installation.
    """
    old_root = Path(legacy_root).resolve()
    old_config = old_root / "localpilot.toml"
    new_config = _effective_config(root, config_path)
    if not old_root.is_dir() or not old_config.is_file() or not new_config.is_file():
        raise ValueError("Both LocalPilot installations must have existing configuration files.")
    old_data = old_root / load_config(old_config).agent.data_dir
    new_data = root.resolve() / load_config(new_config).agent.data_dir
    if not _same_path(old_data, new_data):
        raise ValueError("The legacy installation uses different LocalPilot data; it was preserved.")
    return old_config


def migrate_startup(
    root: Path, config_path: str | Path | None, *, legacy_root: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh an enabled setup; migrate a specified legacy checkout only if it shares data.

    The installer must call this from its existing elevated process. A disabled
    preference is left disabled. No desktop is launched by registration.
    """
    if os.name != "nt":
        return {"ok": False, "reason": "not-windows"}
    status = _operation("status", root, config_path, python_executable=python_executable)
    link = status.get("legacy")
    legacy = None
    if _owned_legacy(link, root.resolve(), _effective_config(root, config_path)):
        legacy = link
    elif legacy_root is not None:
        old_root = Path(legacy_root).resolve()
        old_config = validate_legacy_install(root, config_path, old_root)
        if _owned_legacy(link, old_root, old_config):
            legacy = link
    if not status["enabled"] and legacy is None:
        return {"ok": True, "enabled": False, "migrated": False}
    if current_process_elevated() is not True:
        raise PermissionError("Reopen LocalPilot as administrator to change login startup.")
    result = _operation("enable", root, config_path, python_executable=python_executable, legacy=legacy)
    return {"ok": True, "enabled": result["enabled"], "migrated": legacy is not None}
