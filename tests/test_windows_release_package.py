from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _extraction_wrapper() -> str:
    source = (ROOT / "scripts" / "build-windows-installer.ps1").read_text(encoding="utf-8")
    match = next(
        (
            candidate
            for candidate in re.finditer(
                r"(?ms)@'\r?\n(?P<body>.*?)\r?\n'@\s*\|\s*Set-Content(?P<destination>[^\r\n]*)",
                source,
            )
            if "'Extract-Setup.ps1'" in candidate["destination"]
        ),
        None,
    )
    assert match is not None, "The shipped extraction wrapper must remain inspectable"
    return match["body"]


@pytest.mark.parametrize("installer_exit", [0, 19])
def test_generated_extraction_wrapper_waits_for_installer_and_propagates_failure(tmp_path, installer_exit):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if os.name != "nt" or not powershell:
        pytest.skip("Windows extraction wrapper integration")
    wrapper = tmp_path / "Extract-Setup.ps1"
    wrapper.write_text(_extraction_wrapper(), encoding="utf-8")
    marker = tmp_path / "installer-observation.json"
    with zipfile.ZipFile(tmp_path / "payload.zip", "w") as archive:
        archive.writestr(
            "scripts/install-localpilot.ps1",
            "@{ sourceExists = (Test-Path -LiteralPath $PSScriptRoot); source = $PSScriptRoot } "
            "| ConvertTo-Json | Set-Content -LiteralPath $env:LOCALPILOT_WRAPPER_TEST_MARKER\n"
            "exit ([int]$env:LOCALPILOT_WRAPPER_TEST_EXIT)\n",
        )
    extraction = tmp_path / "temporary extraction"
    extraction.mkdir()
    environment = os.environ.copy()
    environment["TEMP"] = str(extraction)
    environment["LOCALPILOT_WRAPPER_TEST_MARKER"] = str(marker)
    environment["LOCALPILOT_WRAPPER_TEST_EXIT"] = str(installer_exit)
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        env=environment,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    if installer_exit == 0:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    observation = json.loads(marker.read_text(encoding="utf-8-sig"))
    assert observation["sourceExists"] is True
    assert Path(observation["source"]).is_relative_to(extraction)
    if installer_exit == 0:
        assert not list(extraction.glob("LocalPilotSetup-*")), "Successful setup must release its temporary source copy"
    else:
        assert Path(observation["source"]).is_dir(), "Failed setup must retain the reported source evidence"


def test_built_release_archive_matches_committed_identity_and_bundled_provider():
    archive_path = os.environ.get("LOCALPILOT_RELEASE_ARCHIVE")
    if not archive_path:
        pytest.skip("Set LOCALPILOT_RELEASE_ARCHIVE after building the release ZIP")
    path = Path(archive_path)
    assert path.is_file(), "The configured release ZIP is missing"
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        assert len(names) == len(set(names)), "Release ZIP contains duplicate paths"
        for name in names:
            parts = PurePosixPath(name).parts
            assert not name.startswith(("/", "\\")) and ".." not in parts and ":" not in name
            assert parts[0] not in {".git", ".venv", "localpilot-data", "localpilot.toml"}
        manifest = json.loads(archive.read(".localpilot-release.json").decode("utf-8-sig"))
        sha = manifest["source_sha"]
        assert re.fullmatch(r"[0-9a-f]{40}", sha)
        expected_sha = os.environ.get("LOCALPILOT_RELEASE_EXPECTED_SHA")
        if expected_sha:
            assert sha == expected_sha
        project = tomllib.loads(archive.read("pyproject.toml").decode("utf-8-sig"))
        assert manifest["version"] == project["project"]["version"]
        runtime = manifest["runtime_id"]
        assert runtime in {"win-x64", "win-arm64"}
        helper = f"localpilot/_hardware/{runtime}/LocalPilot.SystemSense.HardwareProvider.exe"
        notice = f"localpilot/_hardware/{runtime}/THIRD_PARTY_NOTICES.md"
        with archive.open(helper) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["hardware_provider_sha256"]
        with archive.open(helper) as stream:
            header = stream.read(64)
            assert header[:2] == b"MZ"
            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            stream.seek(pe_offset)
            pe = stream.read(6)
            assert pe[:4] == b"PE\0\0"
            assert struct.unpack_from("<H", pe, 4)[0] == {"win-x64": 0x8664, "win-arm64": 0xAA64}[runtime]
        assert archive.read(notice).strip()
        required = {
            "Install LocalPilot.cmd",
            "scripts/install-localpilot.ps1",
            "scripts/bootstrap.ps1",
            "scripts/install-desktop-shortcut.ps1",
            "config.example.toml",
            "localpilot/cli.py",
            "localpilot/webview/avatar/state-0.png",
        }
        assert required.issubset(names)
        source_files = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", sha],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        assert set(names) == set(source_files) | {helper, notice, ".localpilot-release.json"}
