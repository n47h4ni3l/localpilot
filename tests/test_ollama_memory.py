from __future__ import annotations

from types import SimpleNamespace

import pytest

from localpilot import ollama_memory as memory


MODEL = "nestra:p1.2"
PATH = r"E:\models\blobs\sha256-abc"


class Process:
    def __init__(self, pid, name, children=(), path=PATH):
        self.pid, self._name, self._children, self.path = pid, name, children, path
        self.created = 12345.0

    def name(self):
        return self._name

    def children(self, recursive):
        assert recursive
        return self._children

    def cmdline(self):
        return [self._name, "--model", self.path]

    def create_time(self):
        return self.created

    def is_running(self):
        return True


@pytest.fixture
def scenario(monkeypatch):
    runner = Process(607, "llama-server.exe")
    owner = Process(101, "ollama.exe", [runner])
    row = {"name": MODEL, "model": MODEL, "size": 1000, "size_vram": 1000,
           "context_length": 32768, "digest": "fixture-digest"}
    state = SimpleNamespace(runner=runner, owner=owner, row=row, calls=[], trims=[], models=None)
    monkeypatch.setattr(memory, "_windows", lambda: True)
    monkeypatch.setattr(memory.psutil, "net_connections", lambda **kwargs: [
        SimpleNamespace(status=memory.psutil.CONN_LISTEN, laddr=("127.0.0.1", 11434), pid=owner.pid)])
    monkeypatch.setattr(memory.psutil, "Process", lambda pid: owner)

    def metadata(endpoint, path, payload=None):
        state.calls.append((endpoint, path, payload))
        assert path in {"/api/ps", "/api/show"}
        return ({"models": [dict(row)] if state.models is None else state.models}
                if path == "/api/ps" else {"modelfile": "FROM " + PATH})

    monkeypatch.setattr(memory, "_metadata", metadata)
    monkeypatch.setattr(memory, "_trim_runner", lambda *args: state.trims.append(args))
    return state


def test_reclaims_exact_existing_runner_using_metadata_only(scenario):
    result = memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)
    assert result == {"attempted": True, "success": True,
                      "reason": "working_set_reclaim_requested", "pid": 607}
    assert scenario.trims == [(607, 12345.0)]
    assert scenario.calls == [("http://127.0.0.1:11434", "/api/ps", None),
                              ("http://127.0.0.1:11434", "/api/show", {"model": MODEL}),
                              ("http://127.0.0.1:11434", "/api/ps", None)]


@pytest.mark.parametrize("url", ["https://localhost:11434", "http://example.com:11434",
                                "http://user:pass@localhost:11434", "http://localhost:11434/v1",
                                "http://localhost:11434?x=1", "http://localhost:11434/#fragment",
                                "http://localhost:99999", "http://localhost:0"])
def test_nonlocal_or_ambiguous_endpoint_never_probes_or_trims(scenario, url):
    result = memory.reclaim_loaded_gpu_model_host_memory(url, MODEL)
    assert not result["success"] and not result["attempted"]
    assert not scenario.calls and not scenario.trims


@pytest.mark.parametrize("update", [{"size_vram": 940}, {"size_vram": None},
                                   {"size": 0}, {"size": "1000"}, {"size_vram": True},
                                   {"name": "another-model"}])
def test_missing_gpu_cpu_offload_or_wrong_model_is_not_touched(scenario, update):
    scenario.row.update(update)
    result = memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)
    assert not result["attempted"] and not scenario.trims
    assert len(scenario.calls) == 1


@pytest.mark.parametrize("models", [[], [{"name": MODEL}, {"name": "other"}], None])
def test_ambiguous_loaded_model_fails_closed(scenario, models):
    scenario.models = models if models is not None else ["invalid"]
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["success"]
    assert not scenario.trims


@pytest.mark.parametrize("change", ["other_owner", "two_runners", "wrong_model_path"])
def test_endpoint_to_exact_model_runner_identity_is_required(scenario, change):
    if change == "other_owner":
        scenario.owner._name = "unrelated.exe"
    elif change == "two_runners":
        scenario.owner._children.append(Process(608, "llama-server.exe"))
    else:
        scenario.runner.path = r"E:\models\other.gguf"
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_ambiguous_listener_owners_fail_closed(scenario, monkeypatch):
    monkeypatch.setattr(memory.psutil, "net_connections", lambda **kwargs: [
        SimpleNamespace(status=memory.psutil.CONN_LISTEN, laddr=("127.0.0.1", 11434), pid=pid)
        for pid in (101, 102)])
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_access_denied_does_not_elevate_or_try_alternate_runner(scenario, monkeypatch):
    def denied(*args):
        scenario.trims.append(args)
        raise PermissionError("fixture")
    monkeypatch.setattr(memory, "_trim_runner", denied)
    result = memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)
    assert result["attempted"] and not result["success"]
    assert scenario.trims == [(607, 12345.0)]


def test_changed_process_creation_time_prevents_trim(scenario, monkeypatch):
    original = memory._runner
    calls = []
    def changed(*args):
        calls.append(True)
        if len(calls) == 2:
            scenario.runner.created += 1
        return original(*args)
    monkeypatch.setattr(memory, "_runner", changed)
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_changed_loaded_model_prevents_trim(scenario, monkeypatch):
    original = memory._metadata
    def changed(endpoint, path, payload=None):
        if len(scenario.calls) == 2:
            scenario.row["digest"] = "replacement"
        return original(endpoint, path, payload)
    monkeypatch.setattr(memory, "_metadata", changed)
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_missing_absolute_model_path_prevents_trim(scenario, monkeypatch):
    original = memory._metadata
    monkeypatch.setattr(memory, "_metadata", lambda endpoint, path, payload=None:
                        {"modelfile": "FROM nestra:p1.2"} if path == "/api/show"
                        else original(endpoint, path, payload))
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_non_windows_never_queries_metadata(scenario, monkeypatch):
    monkeypatch.setattr(memory, "_windows", lambda: False)
    assert memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["reason"] == "unsupported_platform"
    assert not scenario.calls and not scenario.trims


def test_network_failure_is_fail_closed(scenario, monkeypatch):
    monkeypatch.setattr(memory, "_metadata", lambda *args: (_ for _ in ()).throw(OSError("offline")))
    assert not memory.reclaim_loaded_gpu_model_host_memory("http://localhost:11434", MODEL)["attempted"]
    assert not scenario.trims


def test_metadata_api_rejects_inference_or_unload_paths():
    for path, payload in [("/api/generate", {"model": MODEL}), ("/api/chat", {}),
                          ("/api/ps", {"keep_alive": 0}), ("/api/show", None)]:
        with pytest.raises(ValueError, match="metadata endpoints"):
            memory._metadata("http://127.0.0.1:11434", path, payload)


def test_metadata_response_is_bounded_and_redirects_are_disabled(monkeypatch):
    observed = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            observed.append(size)
            return b" " * size
    class Opener:
        def open(self, req, timeout):
            assert req.full_url == "http://127.0.0.1:11434/api/ps" and timeout == 2.0
            return Response()
    monkeypatch.setattr(memory.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(ValueError, match="bounded response"):
        memory._metadata("http://127.0.0.1:11434", "/api/ps")
    assert observed == [131073]
    with pytest.raises(ValueError, match="redirects"):
        memory._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")


@pytest.mark.parametrize("handle_created,should_trim", [(12345.0, True), (12346.0, False)])
def test_windows_handle_is_typed_and_creation_time_is_verified(monkeypatch, handle_created, should_trim):
    import ctypes
    handle = 0x123456781234
    actions = []
    class Function:
        def __init__(self, implementation):
            self.implementation = implementation
        def __call__(self, *args):
            return self.implementation(*args)
    def times(actual_handle, creation, *_args):
        assert actual_handle == handle
        ticks = int((handle_created + 11644473600) * 10000000)
        creation._obj.dwLowDateTime = ticks & 0xFFFFFFFF
        creation._obj.dwHighDateTime = ticks >> 32
        return True
    kernel = SimpleNamespace(
        OpenProcess=Function(lambda *_: handle),
        GetProcessTimes=Function(times),
        CloseHandle=Function(lambda actual: actions.append(("close", actual)) or True))
    psapi = SimpleNamespace(EmptyWorkingSet=Function(lambda actual: actions.append(("trim", actual)) or True))
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: kernel if name == "kernel32" else psapi,
                        raising=False)
    if should_trim:
        memory._trim_runner(607, 12345.0)
    else:
        with pytest.raises(ValueError, match="creation time changed"):
            memory._trim_runner(607, 12345.0)
    assert kernel.OpenProcess.restype is memory.wintypes.HANDLE
    assert psapi.EmptyWorkingSet.argtypes == [memory.wintypes.HANDLE]
    assert psapi.EmptyWorkingSet.restype is memory.wintypes.BOOL
    assert actions == ([("trim", handle)] if should_trim else []) + [("close", handle)]
