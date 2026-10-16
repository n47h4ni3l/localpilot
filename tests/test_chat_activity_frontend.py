from pathlib import Path
import shutil
import subprocess

import pytest


def test_durable_activity_and_live_errors_in_production_frontend():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for frontend interaction")
    fixture = Path(__file__).with_name("webview_activity.cjs")
    result = subprocess.run([node, str(fixture)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "late polls passed" in result.stdout


def test_api_validation_errors_are_visible_and_bounded():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for frontend interaction")
    fixture = Path(__file__).with_name("webview_api_errors.cjs")
    result = subprocess.run([node, str(fixture)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cancellation, and success passed" in result.stdout
