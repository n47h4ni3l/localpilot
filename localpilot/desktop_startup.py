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
import re
import stat
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
function Resolve-IdentitySid([string]$identity) {
    if (-not $identity) { return $null }
    try {
        if ($identity -match '^S-1-') {
            return (New-Object Security.Principal.SecurityIdentifier($identity)).Value
        }
        return (New-Object Security.Principal.NTAccount($identity)).Translate(
            [Security.Principal.SecurityIdentifier]).Value
    } catch { return $null }
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
        (Resolve-IdentitySid $task.Principal.UserId) -ine $sid) {
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
        (Resolve-IdentitySid $triggers[0].UserId) -ine $sid) {
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


_BUNDLE_TASK_STATUS = r"""
$spec = $env:LOCALPILOT_BUNDLE_STARTUP_JSON | ConvertFrom-Json
$sid = Get-IdentitySid
if ($sid -cne $spec.owner_sid) { throw 'This bundle belongs to another Windows account.' }
$all = @(Get-ScheduledTask -TaskPath '\' -ErrorAction Stop)
$states = @{}
$enabled = $true
foreach ($role in @('Desktop', 'Broker', 'Worker')) {
    $expected = $spec.tasks.$role
    $matches = @($all | Where-Object { $_.TaskName -ceq $expected.name })
    if ($matches.Count -eq 0) {
        $states[$role] = @{name=$expected.name; present=$false; enabled=$false}
        $enabled = $false
        continue
    }
    if ($matches.Count -ne 1) { throw 'The bundle startup task names are ambiguous.' }
    $task = $matches[0]
    $actions = @($task.Actions)
    $triggers = @($task.Triggers)
    if ($task.Description -cne $expected.description -or $actions.Count -ne 1 -or
        $actions[0].Execute -ine $spec.powershell -or $actions[0].Arguments -cne $expected.arguments -or
        $actions[0].WorkingDirectory -ine $spec.app_root -or
        (Resolve-IdentitySid $task.Principal.UserId) -cne $sid -or
        $task.Principal.RunLevel -cne $expected.run_level -or $task.Principal.LogonType -cne 'Interactive' -or
        $task.Settings.MultipleInstances -cne 'IgnoreNew' -or $triggers.Count -ne 1 -or
        $triggers[0].CimClass.CimClassName -cne 'MSFT_TaskLogonTrigger' -or
        (Resolve-IdentitySid $triggers[0].UserId) -cne $sid) {
        throw ('The ' + $role + ' startup task differs from this bundle owner, launcher, privilege or logon trigger; it was preserved.')
    }
    $active = [bool]($task.Settings.Enabled -and $triggers[0].Enabled)
    $states[$role] = @{name=$expected.name; present=$true; enabled=$active; state=[string]$task.State}
    if (-not $active) { $enabled = $false }
}
@{enabled=$enabled; tasks=$states} | ConvertTo-Json -Compress -Depth 4
"""


def _bundle_startup_spec(root: Path, config_path: str | Path | None) -> dict[str, Any] | None:
    """Recognize bundle ownership without ever falling through on bad metadata."""
    root = root.absolute()
    marker, tasks_path = root / ".nestra-runtime.json", root.parent / "tasks.json"
    if not marker.exists() and not marker.is_symlink() and not (root.name == "app" and tasks_path.exists()):
        return None
    for path in (marker, tasks_path, root / "scripts/launch-nestra.ps1"):
        for candidate in (path, *path.parents):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Linked bundle startup paths are unsupported.")
    def read(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Bundle startup metadata is missing or oversized.")
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Bundle startup metadata must be an object.")
        return value
    runtime, registered = read(marker), read(tasks_path)
    install = root.parent.resolve()
    config = _effective_config(root, config_path)
    if (runtime.get("schema_version") != 1 or registered.get("format_version") != 1
            or root.name != "app" or not _same_path(runtime.get("install_directory", ""), install)
            or not _same_path(runtime.get("app_root", ""), root)
            or not _same_path(runtime.get("config_path", ""), config)
            or not _same_path(registered.get("install_directory", ""), install)):
        raise ValueError("Bundle startup metadata belongs to another root or configuration.")
    # Registration keeps the original spelling in its description/arguments.
    # Resolved path equality above validates identity without changing it.
    install, root = Path(registered["install_directory"]), Path(runtime["app_root"])
    if not install.is_absolute() or not root.is_absolute():
        raise ValueError("Bundle startup paths must be absolute.")
    sid = registered.get("owner_sid", "")
    if not isinstance(sid, str) or not re.fullmatch(r"S-1-5-[0-9-]+", sid):
        raise ValueError("Bundle startup owner SID is invalid.")
    digest = hashlib.sha256((str(install).lower() + "|" + sid).encode()).hexdigest()[:12]
    tasks = {}
    for role in ("Desktop", "Broker", "Worker"):
        name = f"Nestra ALPHA2 {digest} {role}"
        if registered.get("tasks", {}).get(role) != name:
            raise ValueError("Bundle startup task name differs from its installation identity.")
        tasks[role] = {
            "name": name, "run_level": "Limited" if role == "Desktop" else "Highest",
            "description": f"Nestra ALPHA2; root={install}; owner={sid}; role={role}",
            "arguments": subprocess.list2cmdline([
                "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
                "-File", str(root / "scripts/launch-nestra.ps1"), "-Role", role, "-InstallDirectory", str(install),
            ]),
        }
    return {"owner_sid": sid, "app_root": str(root), "tasks": tasks,
            "powershell": str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe")}


def _bundle_startup_status(root: Path, config_path: str | Path | None) -> dict[str, Any] | None:
    try:
        spec = _bundle_startup_spec(root, config_path)
        if spec is None:
            return None
        environment = os.environ.copy()
        environment["LOCALPILOT_BUNDLE_STARTUP_JSON"] = json.dumps(spec)
        result = subprocess.run(
            [spec["powershell"], "-NoProfile", "-NonInteractive", "-Command", _TASK_FUNCTIONS + _BUNDLE_TASK_STATUS],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment,
            timeout=30, creationflags=hidden_process_creation_flags(),
        )
        status = json.loads(result.stdout)
        if not isinstance(status, dict) or not isinstance(status.get("enabled"), bool) or not isinstance(status.get("tasks"), dict):
            raise ValueError("Windows returned invalid bundle startup status.")
        names = ", ".join(item["name"] for item in spec["tasks"].values())
        return {"ok": True, "managed": True, "can_change": False, **status,
                "reason": "Bundle login startup is managed by Windows Task Scheduler. "
                          "To change it, open Task Scheduler as administrator under this account and inspect all three tasks: " + names}
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
        detail = "Windows could not verify the registered task group." if isinstance(exc, subprocess.SubprocessError) else str(exc)[:240]
        return {"ok": False, "enabled": False, "managed": True, "can_change": False,
                "reason": "Bundle login startup could not be verified; its tasks were preserved. "
                          "Open Windows Task Scheduler as administrator under this account to inspect this installation's Nestra ALPHA2 tasks. " + detail}


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
    bundle = _bundle_startup_status(root, config_path)
    if bundle is not None:
        return bundle
    result = _operation("status", root, config_path)
    legacy = _owned_legacy(result.get("legacy"), root.resolve(), _effective_config(root, config_path))
    return {"ok": True, "enabled": result["enabled"] or legacy}


def set_startup(root: Path, config_path: str | Path | None, enabled: bool) -> dict[str, Any]:
    if os.name != "nt":
        return {"ok": False, "reason": "not-windows"}
    bundle = _bundle_startup_status(root, config_path)
    if bundle is not None:
        return {**bundle, "ok": False}
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
