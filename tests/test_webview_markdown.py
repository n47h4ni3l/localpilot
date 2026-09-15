"""Exercise the production text-node renderer without a broker or desktop."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_markdown_preserves_fences_whitespace_and_rejects_html_execution():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the frontend renderer")
    script = Path(__file__).with_name("webview_markdown.cjs")
    subprocess.run([node, str(script)], check=True, capture_output=True, text=True)
