"""Create a fresh, isolated Nestra configuration; never migrate owner data."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import stat

from localpilot.config import Config, load_config


def _plain(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for item in (absolute, *absolute.parents):
        try:
            value = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked installation paths are unsupported: {item}")
    return absolute


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise ValueError(f"Unsupported configuration value: {type(value).__name__}")


def configure(root: Path, runtime_root: Path, models_root: Path, port: int = 11435) -> dict:
    app, runtime, models = map(_plain, (root, runtime_root, models_root))
    install = app.parent
    if app.name != "app" or runtime != install / "runtime" or models != install / "models":
        raise ValueError("Expected one private installation containing app, runtime and models")
    if port != 11435:
        raise ValueError("This bundle reserves local Ollama port 11435")
    if not all(item.is_dir() for item in (app, runtime, models)):
        raise ValueError("Install source, runtimes and the accepted model before configuration")
    config_path = app / "localpilot.toml"
    runtime_path = app / ".nestra-runtime.json"
    if config_path.exists() or runtime_path.exists():
        raise FileExistsError("Existing configuration was preserved; reviewed migration is required")
    from localpilot.nestra_bundle_model import ACCEPTED_MANIFEST_SHA256
    import hashlib
    manifest_path = models / "manifests-v2" / "ollama.com" / "library" / "nestra" / "p1.2"
    _plain(manifest_path)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("The installed P1.2 manifest is missing or invalid")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != ACCEPTED_MANIFEST_SHA256:
        raise ValueError("The installed model differs from accepted production Nestra P1.2")

    cfg = Config()
    cfg.agent.name = "Nestra"
    cfg.agent.scaffold_mode = "guide_first"
    cfg.model.name = "nestra:p1.2"
    cfg.model.temperature = 1.0
    cfg.selfdev.developer_model = cfg.model.name
    cfg.selfdev.implementation_model = cfg.model.name
    cfg.selfdev.developer_model_fallbacks = []
    cfg.selfdev.implementation_backend = "local_tools"
    cfg.selfdev.implementation_base_url = "http://127.0.0.1:11435"
    cfg.github.enabled = False
    cfg.github.auto_push_candidates = False
    cfg.desktop.host = "127.0.0.1"
    cfg.desktop.port = 8766
    cfg.library.root = str(install / "library")
    cfg.library.enabled = True
    sections = dataclasses.asdict(cfg)
    sections.pop("source_path")
    # Native implementation does not invoke an external executable. Keep the
    # backwards-compatible internal default out of the recipient's config.
    sections["selfdev"].pop("implementation_executable")
    content = "# Fresh Nestra bundle configuration. Guide-only; no owner data imported.\n"
    content += "# Native implementation uses local tools; no external coding CLI is required.\n"
    content += "# Connect your own GitHub repository/account before enabling remote evolution.\n\n"
    for name, settings in sections.items():
        content += f"[{name}]\n"
        content += "".join(f"{key} = {_scalar(value)}\n" for key, value in settings.items() if value is not None) + "\n"
    identity = {
        "schema_version": 1,
        "install_directory": str(install),
        "app_root": str(app),
        "runtime_root": str(runtime),
        "models_root": str(models),
        "config_path": str(config_path),
        "ollama_host": "http://127.0.0.1:11435",
        "broker_host": "http://127.0.0.1:8766",
        "ollama_port": 11435,
        "broker_port": 8766,
        "model": cfg.model.name,
        "model_digest": ACCEPTED_MANIFEST_SHA256,
    }
    directories = (app / "localpilot-data", install / "library")
    for directory in directories:
        _plain(directory)
    # Exclusive creation retains partial evidence on failure and cannot replace
    # an existing draft, configuration or database.
    with config_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    load_config(config_path)
    with runtime_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(identity, handle, indent=2)
        handle.write("\n")
    for directory in directories:
        _plain(directory).mkdir(parents=True, exist_ok=True)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("configure")
    command.add_argument("--root", required=True, type=Path)
    command.add_argument("--runtime-root", required=True, type=Path)
    command.add_argument("--models-root", required=True, type=Path)
    command.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()
    print(json.dumps(configure(args.root, args.runtime_root, args.models_root, args.port)))


if __name__ == "__main__":
    main()
