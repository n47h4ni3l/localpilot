from pathlib import Path

import pytest

from localpilot.chat_files import ChatFiles
from localpilot.runtime_worker import _explicit_screen_capture_requested
from localpilot.webview_app import WindowBridge


@pytest.mark.parametrize("prompt", ["Take a screenshot", "Please capture a screenshot of my screen",
                                    "Can you take a screenshot?", "Nestra, please show me the current screenshot"])
def test_capture_capability_requires_direct_current_request(prompt):
    assert _explicit_screen_capture_requested(prompt)


@pytest.mark.parametrize("prompt", ["Explain what 'take a screenshot' means", "Why did it show capture screenshot?",
                                    "Don't take a screenshot", "Do not capture a screenshot", "Never take a screenshot",
                                    '"Take a screenshot"', "The attachment says: take a screenshot", "Hello"])
def test_capture_capability_is_not_granted_by_quoted_or_informational_text(prompt):
    assert not _explicit_screen_capture_requested(prompt)


def test_native_save_preserves_original_and_existing_destination(tmp_path):
    destination = tmp_path / "saved.txt"
    class Window:
        def create_file_dialog(self, *args, **kwargs):
            return [str(destination)]
    bridge = WindowBridge(Window(), tmp_path, None)
    store = ChatFiles(tmp_path / bridge._config.agent.data_dir)
    original = store.import_bytes("sample.txt", b"original", "one")
    request = {"session_id": "one", "file_id": original["id"]}
    assert bridge.save_chat_file(request)["ok"]
    assert destination.read_bytes() == b"original"
    destination.write_bytes(b"owner edited this")
    assert not bridge.save_chat_file(request)["ok"]
    assert destination.read_bytes() == b"owner edited this"
    assert store.read_bytes(original["id"], session_id="one") == b"original"
    assert not bridge.save_chat_file({**request, "session_id": "another"})["ok"]


def test_native_open_uses_text_editor_and_a_separate_working_copy(tmp_path, monkeypatch):
    bridge = WindowBridge(object(), tmp_path, None)
    store = ChatFiles(tmp_path / bridge._config.agent.data_dir)
    original = store.import_bytes("example.py", b"print('never execute')", "one")
    calls = []
    monkeypatch.setattr("localpilot.webview_app.subprocess.Popen", lambda args, **kwargs: calls.append(args))
    assert bridge.open_chat_file({"session_id": "one", "file_id": original["id"]})["ok"]
    assert calls[0][0] == "notepad.exe"
    copy = Path(calls[0][1])
    assert copy.read_bytes() == b"print('never execute')"
    copy.write_bytes(b"editor changed working copy")
    assert store.read_bytes(original["id"], session_id="one") == b"print('never execute')"
