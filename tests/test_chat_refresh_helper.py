import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def helper(tmp_path, monkeypatch, *, busy=False, dirty=False):
    path = Path(__file__).parents[1] / "scripts" / "reload-chat-features.py"
    spec = importlib.util.spec_from_file_location("chat_refresh_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "src" / "localpilot"
    root.mkdir(parents=True)
    config = root / "localpilot.toml"
    config.write_text("test config")
    module.ROOT = root
    module.load_config = lambda _: SimpleNamespace(agent=SimpleNamespace(data_dir="localpilot-data"))
    module.BrokerClient = lambda *_: SimpleNamespace(request=lambda *a, **k: {"pid": 1, "pending_requests": int(busy)})
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kwargs:
                        "dirty" if "status" in args and dirty else "" if "status" in args else "a" * 40)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not stop/launch any process")
    monkeypatch.setattr(module, "_stop_localpilot_processes", forbidden)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    monkeypatch.setattr(sys, "argv", [str(path), "--expected-sha", "a" * 40, "--config-sha256",
                                     hashlib.sha256(config.read_bytes()).hexdigest(), "--dry-run"])
    return module


def test_refresh_preflight_saves_receipt_without_changing_the_running_app(tmp_path, monkeypatch):
    module = helper(tmp_path, monkeypatch)
    assert module.main() is None
    receipt = next((tmp_path / "diagnostics").glob("chat-refresh-*/receipt.json"))
    assert json.loads(receipt.read_text())["status"] == "preflight_passed_no_changes"


def test_refresh_refuses_active_replies(tmp_path, monkeypatch):
    module = helper(tmp_path, monkeypatch, busy=True)
    assert module.main() == 1
    receipt = next((tmp_path / "diagnostics").glob("chat-refresh-*/receipt.json"))
    assert "responding" in json.loads(receipt.read_text())["error"]


def test_refresh_refuses_changed_source(tmp_path, monkeypatch):
    module = helper(tmp_path, monkeypatch, dirty=True)
    assert module.main() == 1
