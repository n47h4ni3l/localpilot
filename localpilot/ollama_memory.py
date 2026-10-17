"""Reclaim one verified, already GPU-resident Ollama runner's host working set.

This is a Windows handoff preparation, not permission to exceed a resource guard.
Callers must establish a safe tool boundary and resample their unchanged limits
afterward. Metadata does not establish runner idleness. No inference, model load,
unload, process termination, elevation or persistent monitoring occurs here.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import ipaddress
import json
import ntpath
import os
from urllib import request
from urllib.parse import urlsplit

import psutil


def _windows() -> bool:
    return os.name == "nt"


def _endpoint(base_url: str) -> tuple[str, str, int]:
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if (parsed.scheme != "http" or host not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be a plain loopback HTTP Ollama URL")
    port = 80 if parsed.port is None else parsed.port
    if not 1 <= port <= 65535:
        raise ValueError("invalid endpoint port")
    # Avoid proxy environment variables and DNS/name-resolution substitutions.
    host = "127.0.0.1" if host == "localhost" else host
    authority = f"[{host}]" if host == "::1" else host
    return f"http://{authority}:{port}", host, port


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("metadata redirects are not allowed")


def _metadata(endpoint: str, path: str, payload: dict | None = None) -> dict:
    if (path, payload is None) not in {("/api/ps", True), ("/api/show", False)}:
        raise ValueError("only model metadata endpoints are allowed")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint + path, data=body, headers={"Content-Type": "application/json"})
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=2.0) as response:
        raw = response.read(131073)
    if len(raw) > 131072:
        raise ValueError("model metadata exceeds the bounded response limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid model metadata")
    return value


def _loaded_gpu_model(metadata: dict, model: str) -> dict:
    models = metadata.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("loaded model identity is absent or ambiguous")
    row = models[0]
    names = [row[key] for key in ("name", "model") if key in row]
    if not names or any(name != model for name in names):
        raise ValueError("loaded model does not match the accepted model")
    size, vram = row.get("size"), row.get("size_vram")
    if (type(size) is not int or type(vram) is not int or size <= 0
            or vram < size * 0.95):
        raise ValueError("full GPU residency is not established")
    return row


def _model_path(metadata: dict) -> str:
    modelfile = metadata.get("modelfile")
    if not isinstance(modelfile, str):
        raise ValueError("model file identity is unavailable")
    paths = [line.split(None, 1)[1].strip() for line in modelfile.splitlines()
             if line.strip().upper().startswith("FROM ")]
    if len(paths) != 1:
        raise ValueError("model file identity is ambiguous")
    path = paths[0]
    if len(path) > 4096 or "\0" in path:
        raise ValueError("invalid model file identity")
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"\"", "'"}:
        path = path[1:-1]
    if not ntpath.isabs(path) or not ntpath.splitdrive(path)[0]:
        raise ValueError("model metadata did not identify an absolute Windows model file")
    return ntpath.normcase(ntpath.normpath(path))


def _command_model(process) -> str:
    args = process.cmdline()
    paths = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == "--model"]
    paths += [arg[len("--model="):] for arg in args if arg.startswith("--model=")]
    if len(paths) != 1:
        raise ValueError("runner model argument is absent or ambiguous")
    return ntpath.normcase(ntpath.normpath(paths[0]))


def _runner(host: str, port: int, model_path: str):
    owners: set[int] = set()
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status != psutil.CONN_LISTEN or not connection.laddr:
            continue
        address, bound_port = connection.laddr[:2]
        if bound_port != port:
            continue
        # IPv4 and IPv6 listeners are deliberately kept distinct.
        allowed = {host, "::" if ipaddress.ip_address(host).version == 6 else "0.0.0.0"}
        if address in allowed:
            if connection.pid is None:
                raise ValueError("endpoint listener owner is unavailable")
            owners.add(connection.pid)
    if len(owners) != 1:
        raise ValueError("endpoint listener owner is absent or ambiguous")
    owner = psutil.Process(next(iter(owners)))
    if owner.name().casefold() != "ollama.exe":
        raise ValueError("endpoint listener is not an Ollama server")
    runners = [child for child in owner.children(recursive=True)
               if child.name().casefold() == "llama-server.exe"]
    if len(runners) != 1 or _command_model(runners[0]) != model_path:
        raise ValueError("model runner identity is absent or ambiguous")
    return owner, runners[0]


def _identity(process) -> tuple[int, float]:
    if not process.is_running():
        raise ValueError("process exited during identity verification")
    return process.pid, process.create_time()


def _trim_runner(pid: int, created: float) -> None:
    """Use the same queried handle for identity verification and the trim."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    psapi.EmptyWorkingSet.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0100 | 0x1000, False, pid)  # SET_QUOTA | QUERY_LIMITED_INFORMATION
    if not handle:
        raise PermissionError("runner handle access denied")
    try:
        creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in
                                              (creation, exit_time, kernel_time, user_time))):
            raise PermissionError("runner handle identity unavailable")
        handle_created = ((creation.dwHighDateTime << 32) | creation.dwLowDateTime) / 10000000 - 11644473600
        if abs(handle_created - created) > 0.001:
            raise ValueError("runner handle creation time changed")
        if not psapi.EmptyWorkingSet(handle):
            raise PermissionError("runner working set reclaim denied")
    finally:
        kernel.CloseHandle(handle)


def reclaim_loaded_gpu_model_host_memory(base_url: str, model: str) -> dict:
    """Best-effort metadata-only preparation; callers must resample capacity.

    A successful trim is an OS request, not proof of available memory or lower
    commit usage. Other model/server layouts fail closed and remain guarded.
    """
    result = {"attempted": False, "success": False, "reason": "unsupported_platform", "pid": None}
    if not _windows():
        return result
    try:
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise ValueError("accepted model identity is invalid")
        endpoint, host, port = _endpoint(base_url)
        original = _loaded_gpu_model(_metadata(endpoint, "/api/ps"), model)
        model_path = _model_path(_metadata(endpoint, "/api/show", {"model": model}))
        owner, runner = _runner(host, port, model_path)
        owner_identity, runner_identity = _identity(owner), _identity(runner)
        current = _loaded_gpu_model(_metadata(endpoint, "/api/ps"), model)
        if any(current.get(key) != original.get(key) for key in
               ("name", "model", "digest", "size", "size_vram", "context_length")):
            raise ValueError("loaded model identity changed")
        # Re-read listener ownership, process ancestry, command and creation
        # times immediately before opening the single target process handle.
        current_owner, current_runner = _runner(host, port, model_path)
        if _identity(current_owner) != owner_identity or _identity(current_runner) != runner_identity:
            raise ValueError("endpoint or runner process identity changed")
        result["pid"] = runner_identity[0]
        result["attempted"] = True
        _trim_runner(*runner_identity)
        result.update(success=True, reason="working_set_reclaim_requested")
    except (OSError, ValueError, TypeError, AttributeError, psutil.Error):
        # Keep path/command/provider details out of tool/audit summaries. No
        # elevation or alternate target is attempted after any failure.
        result["reason"] = "identity_or_permission_check_failed"
    return result
