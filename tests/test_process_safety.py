from pathlib import Path


def test_runtime_never_enables_shell_true():
    root = Path(__file__).resolve().parents[1] / "localpilot"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "shell=True" not in source

