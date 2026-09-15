from pathlib import Path
import shutil
import subprocess

import pytest


def test_production_history_management_behaviors():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the frontend handlers")
    subprocess.run(
        [node, str(Path(__file__).with_name("webview_history.cjs"))],
        check=True, capture_output=True, text=True,
    )
