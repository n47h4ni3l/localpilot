from __future__ import annotations

from pathlib import Path


def test_bootstrap_installs_and_verifies_declared_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")

    assert 'pip install -e ".[dev]"' in script
    assert "pip check" in script
    assert "from PIL import Image, ImageTk" in script
    assert "Pillow could not be imported after installation" in script
    assert "dependency installation failed" in script
    assert "rerun .\\scripts\\bootstrap.ps1" in script


def test_pillow_remains_a_required_runtime_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"Pillow>=10.4,<12"' in pyproject
