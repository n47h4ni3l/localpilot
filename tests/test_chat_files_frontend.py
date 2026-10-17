from pathlib import Path
import shutil
import subprocess

import pytest


def test_chat_file_composer_and_artifact_interactions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for frontend interaction")
    result = subprocess.run([node, str(Path(__file__).with_name("webview_chat_files.cjs"))],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
