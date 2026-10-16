"""Read-only bundle preflight and isolated PowerShell function tests.

These tests never run an installer payload, create a task, start a runtime,
modify Windows settings, or perform model inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-nestra-bundle.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
WINDOWS_PS = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows PowerShell preflight")
REQUIRED = (
    "source.bundle",
    "payload/dependencies/runtime-deps/python/python/python.exe",
    "payload/dependencies/runtime-deps/python/python/pythonw.exe",
    "payload/dependencies/runtime-deps/python/python/Lib/tkinter/__init__.py",
    "payload/dependencies/runtime-deps/ollama/ollama.exe",
    "payload/dependencies/runtime-deps/gh/bin/gh.exe",
    "payload/dependencies/installers/webview2/MicrosoftEdgeWebView2RuntimeInstallerX64.exe",
    "payload/dependencies/installers/pawnio/PawnIO_setup.exe",
    "payload/dependencies/installers/portable-git/PortableGit-2.49.0-64-bit.7z.exe",
    "payload/dependencies/requirements-offline.txt",
    "payload/dependencies/wheelhouse/example-1.0-py3-none-any.whl",
    "payload/model/bundle-model.json",
    "payload/model/manifest.json",
    "payload/hardware/win-x64/LocalPilot.SystemSense.HardwareProvider.exe",
    "payload/hardware/win-x64/THIRD_PARTY_NOTICES.md",
)
GOOD_HOST = {
    "Architecture": "X64", "Is64BitProcess": True,
    "WindowsMajor": 10, "WindowsBuild": 22631, "Avx2": True,
    "RamBytes": 32 * 1024**3, "FreeBytes": 100 * 1024**3, "FixedDrive": True,
}


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    files = []
    for relative in REQUIRED:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ("not executable: " + relative).encode()
        path.write_bytes(content)
        files.append({"path": relative, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    _write_manifest(bundle, {"format_version": 1, "source_sha": "a" * 40, "version": "0.3.0-alpha.2", "files": files})
    return bundle


def _write_manifest(bundle: Path, manifest: dict) -> None:
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _function_wrapper(tmp_path: Path) -> Path:
    wrapper = tmp_path / "inspect-functions.ps1"
    wrapper.write_text(
        "param($Installer, $Bundle, $Destination, $Operation, $StateJson)\n"
        "$ErrorActionPreference='Stop'\nSet-StrictMode -Version Latest\n"
        "$errorsFound=$null; $tokensFound=$null\n"
        "$ast=[Management.Automation.Language.Parser]::ParseFile($Installer,[ref]$tokensFound,[ref]$errorsFound)\n"
        "if ($errorsFound.Count) { throw ($errorsFound | Out-String) }\n"
        "$functions=$ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]},$true)\n"
        "foreach($function in $functions) { Invoke-Expression $function.Extent.Text }\n"
        "$script:hostState=$StateJson | ConvertFrom-Json\n"
        "function Get-NestraInstallHostState { param($Destination) return $script:hostState }\n"
        "function Invoke-NestraSetupProcess { throw 'A payload must never execute in these tests.' }\n"
        "if ($Operation -eq 'names') { Get-NestraBundleTaskNames -Root $Destination -Sid $Bundle | ConvertTo-Json -Compress; exit 0 }\n"
        "if ($Operation -eq 'quote') { ConvertTo-NestraArgument -Value $Bundle; exit 0 }\n"
        "if ($Operation -eq 'launchers') {\n"
        "  $script:names=Get-NestraBundleTaskNames -Root $Destination -Sid $Bundle\n"
        "  function Get-ScheduledTask { param($ErrorAction) if ($script:hostState.TaskExists) { [pscustomobject]@{TaskName=$script:names.Broker;TaskPath='\\'} } }\n"
        "  Assert-NestraLauncherAvailability -Root $Destination -Sid $Bundle -DesktopDirectory (Join-Path (Split-Path $Destination -Parent) 'desktop')\n"
        "  exit 0\n}\n"
        "if ($Operation -eq 'partial-task-failure') {\n"
        "  function Get-ScheduledTask { param($ErrorAction) }\n"
        "  function New-ScheduledTaskSettingsSet { return @{} }\n"
        "  function New-ScheduledTaskAction { return @{} }\n"
        "  function New-ScheduledTaskTrigger { return @{} }\n"
        "  function New-ScheduledTaskPrincipal { return @{} }\n"
        "  $script:registrationCount=0\n"
        "  function Register-ScheduledTask { $script:registrationCount++; if ($script:registrationCount -eq 2) { throw 'simulated registration failure' } }\n"
        "  function Assert-NestraLauncherAvailability { }\n"
        "  $script:receipt=[ordered]@{stage='register-launchers';status='running';updated_at=''}\n"
        "  $script:receiptPath=Join-Path (Split-Path $Destination -Parent) 'receipt.json'\n"
        "  try { Register-NestraBundleTasks -Root $Destination -Sid $Bundle } catch { Write-Output $_.Exception.Message }\n"
        "  exit 0\n}\n"
        "$inventory=Get-NestraBundleInventory -Root $Bundle\n"
        "if ($Operation -eq 'inventory') { @{total=$inventory.TotalBytes; sha=$inventory.Manifest.source_sha} | ConvertTo-Json -Compress; exit 0 }\n"
        "Assert-NestraInstallPreflight -Inventory $inventory -Destination $Destination\n",
        encoding="utf-8",
    )
    return wrapper


def _run_functions(tmp_path: Path, bundle: str | Path, destination: Path, operation="preflight", host=None):
    return subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(_function_wrapper(tmp_path)),
         str(INSTALLER), str(bundle), str(destination), operation, json.dumps(host or GOOD_HOST)],
        capture_output=True, text=True, timeout=30,
    )


@WINDOWS_PS
def test_native_payload_without_claude_passes_read_only_preflight(tmp_path):
    bundle = _bundle(tmp_path)
    before = _snapshot(bundle)
    assert not any("claude" in relative.lower() for relative in before)
    destination = tmp_path / "new installation"
    result = _run_functions(tmp_path, bundle, destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(destination) in result.stdout
    assert not destination.exists()
    assert _snapshot(bundle) == before


@WINDOWS_PS
@pytest.mark.parametrize("failure", ["hash", "size", "missing", "unlisted", "duplicate", "escape", "ads", "case_duplicate", "missing_ollama"])
def test_inventory_rejects_corruption_and_unsafe_paths_before_any_execution(tmp_path, failure):
    bundle = _bundle(tmp_path)
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    first = manifest["files"][0]
    if failure == "hash":
        first["sha256"] = "0" * 64
    elif failure == "size":
        first["size"] += 1
    elif failure == "missing":
        (bundle / first["path"]).unlink()
    elif failure == "unlisted":
        (bundle / "unlisted.exe").write_bytes(b"do not run")
    elif failure == "duplicate":
        manifest["files"].append(first.copy())
    elif failure == "case_duplicate":
        manifest["files"].append({**first, "path": first["path"].upper()})
    elif failure == "missing_ollama":
        selected = next(item for item in manifest["files"] if item["path"].endswith("ollama/ollama.exe"))
        manifest["files"].remove(selected)
        (bundle / selected["path"]).unlink()
    else:
        first["path"] = "../outside.exe" if failure == "escape" else "source.bundle:stream"
    _write_manifest(bundle, manifest)
    before = _snapshot(bundle)
    destination = tmp_path / "new"
    result = _run_functions(tmp_path, bundle, destination)
    assert result.returncode != 0
    assert "A payload must never execute" not in result.stderr
    assert not destination.exists()
    assert _snapshot(bundle) == before


@WINDOWS_PS
@pytest.mark.parametrize("override, expected", [
    ({"RamBytes": 23 * 1024**3}, "24 GiB"),
    ({"FreeBytes": 1024}, "5 GiB"),
    ({"Avx2": False}, "AVX2"),
    ({"Architecture": "Arm64"}, "x64 processor"),
    ({"WindowsBuild": 17763}, "19041"),
    ({"Is64BitProcess": False}, "64-bit Windows"),
    ({"FixedDrive": False}, "local fixed drive"),
])
def test_host_capacity_preflight_fails_closed_without_changes(tmp_path, override, expected):
    bundle = _bundle(tmp_path)
    destination = tmp_path / "new"
    result = _run_functions(tmp_path, bundle, destination, host={**GOOD_HOST, **override})
    assert result.returncode != 0
    assert expected in result.stderr
    assert not destination.exists()


@WINDOWS_PS
def test_existing_installation_and_draft_are_preserved(tmp_path):
    bundle = _bundle(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    draft = destination / "draft.txt"
    draft.write_text("owner's unsent draft", encoding="utf-8")
    result = _run_functions(tmp_path, bundle, destination)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert draft.read_text(encoding="utf-8") == "owner's unsent draft"
    assert list(destination.iterdir()) == [draft]


@WINDOWS_PS
def test_task_names_are_deterministic_and_separate_installations_and_owners(tmp_path):
    owner = "S-1-5-21-123-456-789-1001"
    other = "S-1-5-21-123-456-789-1002"
    destination = tmp_path / "new"
    def names(sid, target):
        result = _run_functions(tmp_path, sid, target, "names")
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    first = names(owner, destination)
    assert first == names(owner, destination)
    assert set(first.values()).isdisjoint(names(other, destination).values())
    assert set(first.values()).isdisjoint(names(owner, tmp_path / "second").values())
    assert set(first) == {"Broker", "Worker", "Desktop"}


@WINDOWS_PS
@pytest.mark.parametrize("collision", ["task", "shortcut", "none"])
def test_launcher_collision_checks_are_read_only_and_precede_installation(tmp_path, collision):
    desktop = tmp_path / "desktop"
    desktop.mkdir()
    if collision == "shortcut":
        (desktop / "Nestra ALPHA2.lnk").write_bytes(b"existing owner shortcut")
    destination = tmp_path / "new-installation"
    before = _snapshot(tmp_path)
    result = _run_functions(tmp_path, "S-1-5-21-123-456-789-1001", destination, "launchers", {"TaskExists": collision == "task"})
    assert (result.returncode == 0) is (collision == "none"), result.stdout + result.stderr
    if collision != "none":
        assert "preserved" in result.stderr
    assert not destination.exists()
    after = _snapshot(tmp_path)
    after.pop("inspect-functions.ps1", None)
    assert after == before


@WINDOWS_PS
def test_partial_task_registration_receipt_names_each_available_task(tmp_path):
    destination = tmp_path / "new-installation"
    launcher = destination / "app/scripts/launch-nestra.ps1"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# test fixture only", encoding="utf-8")
    result = _run_functions(tmp_path, "S-1-5-21-123-456-789-1001", destination, "partial-task-failure")
    assert result.returncode == 0, result.stderr
    assert "simulated registration failure" in result.stdout
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8-sig"))
    assert receipt["stage"] == "register-launchers"
    assert len(receipt["registered_tasks"]) == 1
    assert receipt["registered_tasks"] == [receipt["planned_tasks"]["Broker"]]
    assert not (destination / "tasks.json").exists()


@WINDOWS_PS
def test_verify_only_failure_does_not_create_a_receipt_or_execute_payloads(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "source.bundle").write_bytes(b"corrupt")
    destination = tmp_path / "installation"
    local_data = tmp_path / "user-local-data"
    local_data.mkdir()
    before = _snapshot(tmp_path)
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER),
         "-BundleRoot", str(bundle), "-InstallDirectory", str(destination), "-VerifyOnly"],
        capture_output=True, text=True, timeout=30, env={**os.environ, "LOCALAPPDATA": str(local_data)},
    )
    assert result.returncode != 0
    assert "Bundle file size differs" in result.stderr
    assert _snapshot(tmp_path) == before
    assert not destination.exists()
    assert not list(local_data.iterdir())


def test_offline_install_contract_uses_pinned_local_dependencies_and_separate_privileges():
    source = INSTALLER.read_text(encoding="utf-8")
    assert "'--require-hashes'" in source
    assert "'--no-index'" in source
    assert "'--no-build-isolation'" in source
    assert "'--no-hardlinks'" in source
    assert "'fetch'" not in source and "Get-Command winget" not in source
    assert "'--port', '11435'" in source
    assert "'Limited' } else { 'Highest'" in source
    assert "'tasks.json'" in source
    assert "'Desktop')" in source
    assert "-TaskName $launchers.Names.Broker" in source
    assert "-TaskName $launchers.Names.Worker" in source
    assert "'-install', '-silent'" in source
    assert "SetEnvironmentVariable" not in source
