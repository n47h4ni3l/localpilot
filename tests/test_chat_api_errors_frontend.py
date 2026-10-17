from pathlib import Path
import shutil
import subprocess

import pytest


def test_api_surfaces_bounded_validation_errors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for frontend interaction")
    result = subprocess.run([node, str(Path(__file__).with_name("webview_api_errors.cjs"))],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
