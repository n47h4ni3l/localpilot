import base64
import http.client
import json
import threading

import pytest

from localpilot.broker import BrokerApp, BrokerHTTPServer
from localpilot.chat_store import ChatStore
from localpilot.config import Config
from localpilot.runtime_worker import RuntimeWorker
from localpilot.tools.chat_files import create_chat_file


class Runtime:
    running = True
    def __init__(self):
        self.sent = []
    def send(self, command):
        self.sent.append(command)


def app_at(tmp_path):
    app = BrokerApp(tmp_path, Config())
    app.runtime = Runtime()
    return app


def test_authenticated_upload_read_and_cross_session_isolation(tmp_path):
    app = app_at(tmp_path)
    session = app.store.create_session()["id"]
    other = app.store.create_session()["id"]
    server = BrokerHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    def request(method, path, body=None, auth=True):
        headers = {"Authorization": "Bearer " + app.token} if auth else {}
        client.request(method, path, json.dumps(body) if body else None, headers)
        response = client.getresponse()
        return response.status, json.loads(response.read())
    try:
        path = f"/v1/sessions/{session}/files"
        body = {"name": "draft.txt", "data_base64": base64.b64encode(b"original text").decode()}
        assert request("POST", path, body, auth=False)[0] == 401
        status, payload = request("POST", path, body)
        assert status == 201
        file = payload["file"]
        assert "text" not in file["extraction"]
        status, content = request("GET", path + f"/{file['id']}/content")
        assert status == 200 and base64.b64decode(content["data_base64"]) == b"original text"
        assert request("GET", f"/v1/sessions/{other}/files/{file['id']}/content")[0] in {400, 404}
        assert request("POST", path, {"name": "bad.txt", "data_base64": "!bad!"})[0] == 400
        assert request("POST", path, {"name": "../bad.txt", "data_base64": "YQ=="})[0] == 400
        for name in ("invalid.png", "invalid.docx"):
            status, payload = request("POST", path, {"name": name, "data_base64": "bm90IHZhbGlk"})
            assert status == 400 and payload["error"]
        assert request("GET", path + "/../content")[0] == 400
        status, result = request("POST", f"/v1/sessions/{session}/messages",
                                 {"content": "", "file_ids": [file["id"]]})
        assert status == 202 and result["user"]["files"][0]["id"] == file["id"]
        assert app.runtime.sent[-1]["file_ids"] == [file["id"]]
        assert app.runtime.sent[-1]["accessible_file_ids"] == [file["id"]]
        app._on_runtime_message({"kind": "result", "request_id": result["request_id"],
                                 "session_id": session, "answer": "Read"})
        assert ChatStore(app.store.path).messages(session)[0]["files"][0]["sha256"] == file["sha256"]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tool_steps_and_outputs_survive_failure_and_history_reload(tmp_path):
    app = app_at(tmp_path)
    sid = app.store.create_session()["id"]
    request = app.submit(sid, "Create a file")
    request_id = request["request_id"]
    mid = request["assistant"]["id"]
    def event(type, **payload):
        app._on_runtime_message({"kind": "event", "session_id": sid,
                                 "request_id": request_id, "type": type, "payload": payload})
    event("tool.started", tool="create_chat_file", arguments_preview={"name": "note.txt", "api_key": "private"})
    event("tool.completed", tool="create_chat_file", ok=True, result_preview="A real file was created")
    event("tool.started", tool="create_chat_file", arguments_preview={"name": "second.txt"})
    artifact = app.files.import_bytes("note.txt", b"result", sid, "artifact")
    app._on_runtime_message({"kind": "error", "session_id": sid, "request_id": request_id,
                             "error_type": "ConnectError", "artifact_ids": [artifact["id"]]})
    saved = ChatStore(app.store.path).message(mid)
    assert saved["status"] == "error" and "Ollama" in saved["content"]
    assert saved["files"][0]["id"] == artifact["id"]
    assert [step["status"] for step in saved["steps"]] == ["succeeded", "interrupted"]
    assert saved["steps"][0]["id"] != saved["steps"][1]["id"]
    assert "private" not in saved["steps"][0]["arguments_preview"]
    event("tool.completed", tool="create_chat_file", ok=True)
    assert ChatStore(app.store.path).message(mid)["steps"] == saved["steps"]


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_uses_attachment_content_and_returns_real_created_files(tmp_path, fail):
    worker = RuntimeWorker(tmp_path)
    from localpilot.chat_files import ChatFiles
    store = ChatFiles(tmp_path / worker.config.agent.data_dir)
    file = store.import_bytes("input.txt", b"original evidence", "session")
    prompts, output = [], []
    class Agent:
        tools = {}
        def ask(self, prompt, **kwargs):
            assert "create_chat_file" in self.tools
            prompts.append(prompt)
            create_chat_file("output.txt", "edited evidence")
            if fail:
                raise ConnectionError("test error after a file was created")
            return "Here is the file."
    worker._agent = lambda *_: Agent()
    worker._write = output.append
    worker.handle({"kind": "ask", "request_id": "r", "session_id": "session",
                   "prompt": "Edit this file", "file_ids": [file["id"]], "history": []})
    assert "original evidence" in prompts[0] and "UNTRUSTED DATA" in prompts[0]
    result = output[-1]
    assert result["kind"] == ("error" if fail else "result")
    assert len(result["artifact_ids"]) == 1
    assert store.read_bytes(result["artifact_ids"][0], session_id="session") == b"edited evidence"
    assert store.read_bytes(file["id"], session_id="session") == b"original evidence"


def test_file_tool_schemas_work_with_the_installed_ollama_client():
    from ollama._utils import convert_function_to_tool
    from localpilot.tools.chat_files import CHAT_FILE_TOOL_SPECS
    for spec in CHAT_FILE_TOOL_SPECS:
        tool = convert_function_to_tool(spec.fn)
        assert tool.function.name == spec.name


def test_activity_previews_bound_and_redact_structured_and_repr_secrets():
    from localpilot.chat_activity import activity_preview
    for value in ({"api_key": "private"}, "{'api_key': 'private', 'count': 2}", "Bearer private",
                  "{'secret': 'private phrase', 'count': 2}", '{"password": "private phrase',
                  "authorization='private phrase'"):
        assert "private" not in activity_preview(value)
        assert "phrase" not in activity_preview(value)
    assert len(activity_preview({"text": "a" * 100_000})) < 1700
