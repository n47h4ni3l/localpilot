from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from localpilot import desktop_instance


@pytest.mark.parametrize("config", [None, "owner.toml"])
def test_config_identity_resolves_from_selected_root_not_callers_directory(tmp_path, monkeypatch, config):
    root = tmp_path / "selected"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("LOCALPILOT_CONFIG", raising=False)
    owner = desktop_instance.DesktopInstance(root, config)
    assert owner.config_path == (root / (config or "localpilot.toml")).resolve()


def test_real_process_owner_lock_coalesces_relaunch_and_releases_normally(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    script = (
        "import sys; from localpilot.desktop_instance import DesktopInstance; "
        "owner=DesktopInstance(sys.argv[1]); assert owner.acquire(); "
        "print('ready', flush=True); sys.stdin.readline(); "
        "print(owner.activation_requested(), flush=True); owner.close()"
    )
    # This child only holds a file lock. It never starts a broker or GUI.
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        repeated = desktop_instance.DesktopInstance(root)
        assert repeated.acquire() is False
        assert repeated.request_activation(timeout=0.2) is True
        stdout, stderr = process.communicate("release\n", timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "True"
        assert repeated.acquire() is True
        repeated.close()
        assert not (root / "localpilot-data" / "desktop-instance" / "owner.json").exists()
    finally:
        if process.poll() is None:
            process.communicate("release\n", timeout=10)


def test_os_releases_owner_lock_when_process_exits_without_cleanup(tmp_path):
    script = (
        "import sys; from localpilot.desktop_instance import DesktopInstance; "
        "owner=DesktopInstance(sys.argv[1]); assert owner.acquire()"
    )
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "localpilot-data" / "desktop-instance" / "owner.json").exists()
    next_owner = desktop_instance.DesktopInstance(tmp_path)
    assert next_owner.acquire() is True
    next_owner.close()


def test_configuration_mismatch_is_refused_without_activation(tmp_path):
    owner = desktop_instance.DesktopInstance(tmp_path, tmp_path / "first.toml")
    assert owner.acquire()
    try:
        repeated = desktop_instance.DesktopInstance(tmp_path, tmp_path / "second.toml")
        with pytest.raises(RuntimeError, match="different configuration"):
            repeated.request_activation(timeout=0)
        assert owner.activation_requested() is False
    finally:
        owner.close()


def test_elevated_launcher_cannot_reuse_unelevated_or_unknown_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_instance, "current_process_elevated", lambda: False)
    owner = desktop_instance.DesktopInstance(tmp_path)
    assert owner.acquire()
    try:
        monkeypatch.setattr(desktop_instance, "current_process_elevated", lambda: True)
        repeated = desktop_instance.DesktopInstance(tmp_path)
        with pytest.raises(RuntimeError, match="administrator rights"):
            repeated.request_activation(timeout=0)
        assert owner.activation_requested() is False
    finally:
        owner.close()


def test_new_source_requests_cooperative_replacement_instead_of_activation(tmp_path):
    owner = desktop_instance.DesktopInstance(tmp_path)
    owner.source_sha = "a" * 40
    assert owner.acquire()
    try:
        repeated = desktop_instance.DesktopInstance(tmp_path)
        repeated.source_sha = "b" * 40
        assert repeated.request_activation(timeout=0) is False
        assert owner.replacement_requested() is True
        assert owner.activation_requested() is False
    finally:
        owner.close()


def test_reused_pid_metadata_cannot_activate_a_different_process(tmp_path):
    owner = desktop_instance.DesktopInstance(tmp_path)
    assert owner.acquire()
    try:
        metadata = json.loads(owner.metadata.read_text(encoding="utf-8"))
        metadata["created_at"] -= 100
        owner.metadata.write_text(json.dumps(metadata), encoding="utf-8")
        repeated = desktop_instance.DesktopInstance(tmp_path)
        assert repeated.request_activation(timeout=0) is False
        assert owner.activation_requested() is False
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("module", "arguments", "cwd", "blocked"),
    [
        ("localpilot.cli", ["desktop"], "same", True),
        ("localpilot.cli", ["doctor"], "same", False),
        ("localpilot.cli", ["--config", "desktop", "doctor"], "same", False),
        ("localpilot.cli", ["--config=desktop", "doctor"], "same", False),
        ("localpilot.cli", ["--config", "chosen.toml", "desktop"], "same", True),
        ("localpilot.cli", ["--config=chosen.toml", "desktop"], "same", True),
        ("localpilot.webview_app", ["--root", "same"], "other", True),
        ("localpilot.webview_app", ["--root=same"], "other", True),
        ("localpilot.webview_app", ["--root=other"], "same", False),
        ("localpilot.native_avatar_companion", ["--root", "other"], "same", False),
        ("unrelated.module", ["--root", "same"], "same", False),
    ],
)
def test_legacy_detection_uses_exact_module_command_and_root(tmp_path, monkeypatch, module, arguments, cwd, blocked):
    root, other = tmp_path / "same", tmp_path / "other"
    def argument(value):
        if value in {"same", "other"}:
            return str(root if value == "same" else other)
        if value in {"--root=same", "--root=other"}:
            return "--root=" + str(root if value == "--root=same" else other)
        return value
    argv = ["pythonw.exe", "-m", module, *(argument(value) for value in arguments)]
    process = SimpleNamespace(pid=os.getpid() + 10000, cmdline=lambda: argv, cwd=lambda: str(root if cwd == "same" else other))
    monkeypatch.setattr(desktop_instance.psutil, "process_iter", lambda: [process])
    if blocked:
        with pytest.raises(RuntimeError, match="older LocalPilot desktop"):
            desktop_instance.assert_no_legacy_desktop(root)
    else:
        desktop_instance.assert_no_legacy_desktop(root)


@pytest.mark.parametrize("creation_time", [10.0, 11.0])
def test_replacement_retires_only_the_old_owner_identity_not_a_reused_pid(tmp_path, monkeypatch, creation_time):
    pid = os.getpid() + 10000
    process = SimpleNamespace(
        pid=pid, create_time=lambda: creation_time,
        cmdline=lambda: ["pythonw.exe", "-m", "localpilot.native_avatar_companion", "--root", str(tmp_path)],
        cwd=lambda: str(tmp_path),
    )
    monkeypatch.setattr(desktop_instance.psutil, "process_iter", lambda: [process])
    if creation_time == 10.0:
        desktop_instance.assert_no_legacy_desktop(tmp_path, retired_owners={pid: 10.0})
    else:
        with pytest.raises(RuntimeError, match="older LocalPilot desktop"):
            desktop_instance.assert_no_legacy_desktop(tmp_path, retired_owners={pid: 10.0})
