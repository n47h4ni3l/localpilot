"""Owner-run, guarded refresh of the canonical desktop/broker after this release.

Does not elevate itself, alter Windows tasks/settings or force-stop any app.
The normal-user UI is opened through the existing Explorer shell.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import psutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localpilot.config import load_config
from localpilot.desktop import BrokerClient
from localpilot.desktop_updater import _stop_localpilot_processes
from localpilot.process import hidden_process_creation_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--desktop-shortcut", type=Path)
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT.parents[1] / "diagnostics" / ("chat-refresh-" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"created_at": stamp, "dry_run": args.dry_run, "expected_sha": args.expected_sha}
    exit_code = 0
    try:
        flags = hidden_process_creation_flags()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, creationflags=flags).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, creationflags=flags).strip()
        if head != args.expected_sha or dirty:
            raise RuntimeError("The checkout changed or has unsaved source edits. Refresh was not started.")
        config_path = ROOT / "localpilot.toml"
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != args.config_sha256:
            raise RuntimeError("The production configuration changed. Refresh was not started.")
        config = load_config(config_path)
        data_dir = (ROOT / config.agent.data_dir).resolve()
        client = BrokerClient(ROOT, config)
        status = client.request("GET", "/v1/broker/status", timeout=3)
        receipt["original_broker"] = status
        if status.get("pending_requests") != 0:
            raise RuntimeError("Nestra is responding. Let the reply finish, then run this helper again.")
        if not args.dry_run and (os.name != "nt" or not ctypes.windll.shell32.IsUserAnAdmin()):
            raise RuntimeError("Right-click the saved desktop helper and choose Run as administrator.")
        if args.dry_run:
            receipt["status"] = "preflight_passed_no_changes"
            print("Preflight passed. No app or Windows settings were changed.")
            return
        database = data_dir / config.desktop.chat_database
        if database.exists():
            with sqlite3.connect(database) as source, sqlite3.connect(folder / "chat-before-refresh.sqlite3") as backup:
                source.backup(backup, pages=128)
        print("Chat saved. Asking the existing app to close safely...")
        launch_after = time.time()
        _stop_localpilot_processes(ROOT, config_path=config_path, include_background_worker=True, timeout=30)
        executable = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        broker = subprocess.Popen([str(executable), "-m", "localpilot.broker", "--root", str(ROOT),
                                   "--config", str(config_path)], cwd=ROOT, creationflags=flags)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                new_status = client.request("GET", "/v1/broker/status", timeout=2)
                features = client.request("GET", "/v1/chat/features", timeout=2)
                if new_status.get("elevated") and new_status.get("pid") != status.get("pid") and features.get("version") == 1:
                    receipt["new_broker"] = new_status
                    receipt["features"] = features
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("The new broker did not confirm readiness. See this receipt before retrying.")
        if args.restore_worker:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                            "Start-ScheduledTask -TaskName 'LocalPilot Background Worker' -ErrorAction Stop"],
                           check=True, creationflags=flags)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    owner = json.loads((data_dir / "background-worker.pid").read_text(encoding="utf-8"))
                    process = psutil.Process(int(owner["pid"]))
                    command = process.cmdline()
                    if (process.create_time() >= launch_after - 1 and "localpilot.background_worker" in command
                            and Path(owner["root"]).resolve() == ROOT.resolve()):
                        receipt["worker_restoration"] = {"pid": process.pid, "created_at": process.create_time(),
                                                         "source_revision": head, "verified": True}
                        break
                except (OSError, ValueError, KeyError, psutil.Error):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("The background worker did not confirm a fresh process. Check the saved receipt.")
        if args.desktop_shortcut and args.desktop_shortcut.is_file():
            subprocess.Popen(["explorer.exe", str(args.desktop_shortcut)], creationflags=flags)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    owner = json.loads((data_dir / "desktop-instance" / "owner.json").read_text(encoding="utf-8"))
                    process = psutil.Process(int(owner["pid"]))
                    if (owner.get("source_sha") == head and owner.get("elevated") is False
                            and process.create_time() >= launch_after - 1
                            and abs(process.create_time() - float(owner["created_at"])) < 0.01
                            and "localpilot.cli" in process.cmdline()):
                        receipt["normal_desktop"] = {key: owner.get(key) for key in ("pid", "created_at", "source_sha", "elevated")}
                        break
                except (OSError, ValueError, KeyError, psutil.Error):
                    pass
                time.sleep(0.5)
        receipt["status"] = "refreshed"
        print("The new chat backend is ready. Model and configuration are unchanged.")
        if not receipt.get("normal_desktop"):
            print("Open Nestra UI Check from your desktop to show the chat.")
    except Exception as exc:
        receipt.update(status="stopped_with_error", error_type=type(exc).__name__, error=str(exc))
        print("Refresh stopped:", exc)
        exit_code = 1
    finally:
        path = folder / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print("Saved receipt:", path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
