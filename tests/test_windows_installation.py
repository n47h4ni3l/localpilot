from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows installation integration")


def _powershell(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args],
        capture_output=True, text=True, timeout=60,
    )


def _copy_script(root: Path, name: str) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    destination = scripts / name
    shutil.copy2(ROOT / "scripts" / name, destination)
    return destination


def test_shortcut_reinstall_preserves_elevation_until_explicitly_disabled(tmp_path: Path):
    root = tmp_path / "repo"
    installer = _copy_script(root, "install-desktop-shortcut.ps1")
    avatar = root / "localpilot" / "webview" / "avatar"
    avatar.mkdir(parents=True)
    shutil.copy2(ROOT / "localpilot" / "webview" / "avatar" / "state-0.png", avatar)
    shortcut = tmp_path / "LocalPilot.lnk"
    wrapper = tmp_path / "install-shortcut.ps1"
    wrapper.write_text(
        "param($Installer, $Python, $Shortcut, $Elevation)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$options = @{ PythonPath=$Python; ShortcutPath=$Shortcut }\n"
        "if ($Elevation -eq 'yes') { $options.RunAsAdministrator = $true }\n"
        "if ($Elevation -eq 'no') { $options.RunAsAdministrator = $false }\n"
        "& $Installer @options\n",
        encoding="utf-8",
    )
    for elevation, expected in (("yes", True), ("preserve", True), ("no", False), ("preserve", False)):
        result = _powershell(wrapper, str(installer), sys.executable, str(shortcut), elevation)
        assert result.returncode == 0, result.stdout + result.stderr
        flags = struct.unpack_from("<I", shortcut.read_bytes(), 0x14)[0]
        assert bool(flags & 0x2000) is expected


@pytest.mark.parametrize("publish_mode", ["fail", "missing", "success", "separate"])
def test_hardware_publish_preserves_installed_provider_until_replacement_is_complete(tmp_path: Path, publish_mode: str):
    root = tmp_path / "repo"
    builder = _copy_script(root, "build-systemsense-hardware.ps1")
    project = root / "tools" / "SystemSense.HardwareProvider"
    project.mkdir(parents=True)
    (project / "THIRD_PARTY_NOTICES.md").write_text("Required notice", encoding="utf-8")
    installed = root / "localpilot" / "_hardware" / "win-x64"
    installed.mkdir(parents=True)
    provider = installed / "LocalPilot.SystemSense.HardwareProvider.exe"
    provider.write_bytes(b"previous working provider")
    (installed / "obsolete.dll").write_bytes(b"old")
    dotnet = tmp_path / "dotnet.ps1"
    dotnet.write_text(
        "$Arguments = $args\n"
        "if ($Arguments[0] -eq '--list-sdks') { '8.0.100 [test]'; $global:LASTEXITCODE=0; return }\n"
        + ("$global:LASTEXITCODE=1; return\n" if publish_mode == "fail" else "")
        + ("$global:LASTEXITCODE=0; return\n" if publish_mode == "missing" else "")
        + "$out = $Arguments[[Array]::IndexOf($Arguments, '--output') + 1]\n"
        "[IO.File]::WriteAllText((Join-Path $out 'LocalPilot.SystemSense.HardwareProvider.exe'), 'replacement provider')\n"
        "$global:LASTEXITCODE=0\n",
        encoding="utf-8",
    )
    extra_args = ()
    if publish_mode == "separate":
        extra_args = ("-OutputDirectory", str(installed.parent / "installer-build" / "test" / "win-x64"))
    result = _powershell(builder, "-DotNetPath", str(dotnet), *extra_args)
    if publish_mode == "success":
        assert result.returncode == 0, result.stdout + result.stderr
        assert provider.read_bytes() == b"replacement provider"
        assert (installed / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == "Required notice"
        assert not (installed / "obsolete.dll").exists()
    elif publish_mode == "separate":
        assert result.returncode == 0, result.stdout + result.stderr
        assert provider.read_bytes() == b"previous working provider"
        assert (installed.parent / "installer-build" / "test" / "win-x64" / provider.name).read_bytes() == b"replacement provider"
    else:
        assert result.returncode != 0
        assert provider.read_bytes() == b"previous working provider"
        assert (installed / "obsolete.dll").read_bytes() == b"old"
    if publish_mode != "separate":
        assert sorted(path.name for path in installed.parent.iterdir()) == ["win-x64"]


def _installation_functions(tmp_path: Path) -> Path:
    wrapper = tmp_path / "installation-functions.ps1"
    wrapper.write_text(
        "param($Installer, $Operation, $Source, $Destination, $Remote)\n"
        "$ErrorActionPreference='Stop'\n"
        "$ast = [Management.Automation.Language.Parser]::ParseFile($Installer, [ref]$null, [ref]$null)\n"
        "$functions = $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)\n"
        "foreach ($function in $functions) { Invoke-Expression $function.Extent.Text }\n"
        "if ($Operation -eq 'copy') { Copy-ReleaseSources -Source $Source -Destination $Destination }\n"
        "if ($Operation -eq 'init') { Initialize-ReleaseCheckout -Root $Destination -RepositoryUrl $Remote }\n",
        # Only function bodies execute; these tests never install prerequisites
        # or create the owner's desktop shortcut.
        encoding="utf-8",
    )
    return wrapper


@pytest.mark.parametrize("condition", ["valid", "corrupt", "wrong_arch"])
def test_release_payload_requires_matching_architecture_and_provider_hash(tmp_path: Path, condition: str):
    root = tmp_path / "bundle"
    provider = root / "localpilot" / "_hardware" / "win-x64" / "LocalPilot.SystemSense.HardwareProvider.exe"
    provider.parent.mkdir(parents=True)
    provider.write_bytes(b"packaged provider")
    manifest = {
        "source_sha": "a" * 40, "version": "0.2.1", "runtime_id": "win-x64",
        "hardware_provider_sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
    }
    (root / ".localpilot-release.json").write_text(json.dumps(manifest), encoding="utf-8")
    if condition == "corrupt":
        provider.write_bytes(b"corrupt provider")
    wrapper = _installation_functions(tmp_path)
    with wrapper.open("a", encoding="utf-8") as file:
        file.write("if ($Operation -eq 'validate') { Assert-ReleasePayload -Root $Destination -ExpectedRuntime $Remote | Out-Null }\n")
    expected_runtime = "win-arm64" if condition == "wrong_arch" else "win-x64"
    result = _powershell(wrapper, str(ROOT / "scripts" / "install-localpilot.ps1"), "validate", "unused", str(root), expected_runtime)
    assert (result.returncode == 0) is (condition == "valid"), result.stdout + result.stderr


def test_release_copy_preserves_existing_configuration_data_and_environment(tmp_path: Path):
    source = tmp_path / "bundle"
    destination = tmp_path / "installed"
    for root, text in ((source, "bundle"), (destination, "owner")):
        root.mkdir()
        (root / "localpilot.toml").write_text(text, encoding="utf-8")
        for directory in ("localpilot-data", ".venv", ".git"):
            folder = root / directory
            folder.mkdir()
            (folder / "preserved").write_text(text, encoding="utf-8")
    (source / "application.txt").write_text("release application", encoding="utf-8")
    result = _powershell(
        _installation_functions(tmp_path), str(ROOT / "scripts" / "install-localpilot.ps1"),
        "copy", str(source), str(destination), "unused",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "application.txt").read_text() == "release application"
    assert (destination / "localpilot.toml").read_text() == "owner"
    for directory in ("localpilot-data", ".venv", ".git"):
        assert (destination / directory / "preserved").read_text() == "owner"


def test_release_git_initialization_can_resume_after_failed_fetch_without_replacing_files(tmp_path: Path):
    def git(*args: str, check: bool = True):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=check, timeout=30)

    source = tmp_path / "source"
    source.mkdir()
    git("-C", str(source), "init", "-b", "main")
    (source / "application.txt").write_text("release application", encoding="utf-8")
    git("-C", str(source), "add", "application.txt")
    git("-C", str(source), "-c", "user.name=Installation Test", "-c", "user.email=test@example.invalid", "commit", "-m", "release")
    sha = git("-C", str(source), "rev-parse", "HEAD").stdout.strip()
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "application.txt").write_text("release application", encoding="utf-8")
    (destination / "localpilot.toml").write_text("owner configuration", encoding="utf-8")
    (destination / ".localpilot-release.json").write_text(json.dumps({"source_sha": sha}), encoding="utf-8")
    remote = tmp_path / "remote.git"
    wrapper = _installation_functions(tmp_path)
    args = (str(ROOT / "scripts" / "install-localpilot.ps1"), "init", "unused", str(destination), str(remote))
    failed = _powershell(wrapper, *args)
    assert failed.returncode != 0
    assert (destination / ".git" / "localpilot-install-pending").exists()
    git("init", "--bare", str(remote))
    git("-C", str(source), "push", str(remote), "main")
    succeeded = _powershell(wrapper, *args)
    assert succeeded.returncode == 0, succeeded.stdout + succeeded.stderr
    assert git("-C", str(destination), "rev-parse", "HEAD").stdout.strip() == sha
    assert git("-C", str(destination), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").stdout.strip() == "origin/main"
    assert (destination / "localpilot.toml").read_text() == "owner configuration"
    assert (destination / "application.txt").read_text() == "release application"
    assert not (destination / ".git" / "localpilot-install-pending").exists()
