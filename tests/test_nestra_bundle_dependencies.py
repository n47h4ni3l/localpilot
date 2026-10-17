from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage-nestra-dependencies.py"
spec = importlib.util.spec_from_file_location("nestra_dependency_stager", SCRIPT)
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)


def package(data=b"data", *, name="python312", kind="zip", filename="runtime.zip"):
    return {"name": name, "kind": kind, "url": "https://www.python.org/" + filename,
            "sha256": hashlib.sha256(data).hexdigest(), "max_bytes": max(len(data), 1024),
            "max_unpacked_bytes": 1024 * 1024}


def lock(entry=None):
    return {"schema_version": 1, "packages": [entry or package()],
            "wheels": {"requirements": ["example==1.2.3"], "max_total_bytes": 1024 * 1024}}


@pytest.mark.parametrize("url", ["http://python.org/x.zip", "https://user:secret@python.org/x.zip", "https://localhost/x.zip", "https://127.0.0.1/x.zip", "https://10.0.0.1/x.zip", "https://python.org:8443/x.zip", "https://python.org/x.zip#fragment"])
def test_only_public_credential_free_https_urls(url):
    with pytest.raises(ValueError):
        stager.validate_url(url)


@pytest.mark.parametrize("change", [{"sha256": "x" * 64}, {"max_bytes": True}, {"name": "../escape"}, {"filename": "CON.zip"}, {"kind": "7z"}])
def test_lock_validates_hash_limits_names_and_kind(change):
    with pytest.raises(ValueError):
        stager.validate_lock(lock(package() | change))


def test_lock_rejects_duplicate_packages_unpinned_wheels_and_overallocation():
    raw = lock()
    raw["packages"].append(package())
    with pytest.raises(ValueError, match="unique"):
        stager.validate_lock(raw)
    for requirement in ("example>=1", "example==1.*", "https://example.com/demo.whl", "example==1; python_version>'3'", "-r local.txt"):
        raw = lock()
        raw["wheels"]["requirements"] = [requirement]
        with pytest.raises(ValueError, match="exact"):
            stager.validate_lock(raw)
    raw = lock()
    raw["policy"] = {"max_total_download_bytes": 10}
    with pytest.raises(ValueError, match="total policy"):
        stager.validate_lock(raw)
    with pytest.raises(ValueError, match="Model weight"):
        stager.validate_lock(lock(package(kind="file", filename="candidate.gguf")))


def test_cached_download_streams_checks_hash_reuses_exact_cache_and_preserves_corruption(tmp_path, monkeypatch):
    data = b"downloaded-data" * 300
    item = stager.validate_lock(lock(package(data)))["packages"][0]
    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(data))}
        def geturl(self):
            return item["url"]
    class Opener:
        def open(self, request, timeout):
            assert timeout == 60
            assert request.full_url.startswith("https://")
            return Response(data)
    monkeypatch.setattr(stager, "build_opener", lambda *_: Opener())
    destination = stager.cached_download(item, tmp_path)
    assert destination.read_bytes() == data
    monkeypatch.setattr(stager, "build_opener", lambda *_: pytest.fail("A valid cache must not use network"))
    assert stager.cached_download(item, tmp_path) == destination
    destination.write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="preserved"):
        stager.cached_download(item, tmp_path)
    assert destination.read_bytes() == b"bad cache"


def test_failed_hash_or_oversized_download_never_commits_cache(tmp_path, monkeypatch):
    data = b"actual"
    item = stager.validate_lock(lock(package(data)))["packages"][0]
    class Response(io.BytesIO):
        headers = {}
        def geturl(self):
            return item["url"]
    class Opener:
        def open(self, *_args, **_kwargs):
            return Response(b"incorrect")
    monkeypatch.setattr(stager, "build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="SHA256"):
        stager.cached_download(item, tmp_path)
    assert not list(tmp_path.rglob("*.zip"))
    item["max_bytes"] = 2
    with pytest.raises(ValueError, match="byte limit"):
        stager.cached_download(item, tmp_path)
    assert not list(tmp_path.rglob("*.part-*"))


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "a\\escape.txt", "a/../../escape.txt", "name.txt:stream", "folder./file.txt"])
def test_zip_traversal_and_windows_aliases_are_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    make_zip(archive, [(name, b"bad")])
    if "\\" in name:
        # On Windows ZipInfo normalizes backslashes while writing; construct
        # the malformed incoming archive bytes explicitly for this case.
        archive.write_bytes(archive.read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
    with pytest.raises(ValueError):
        stager.extract_zip(archive, tmp_path / "out", max_bytes=1024, max_entries=10)
    assert not (tmp_path / "escape.txt").exists()


def test_zip_symlink_duplicate_and_expansion_bomb_are_rejected(tmp_path):
    archive = tmp_path / "links.zip"
    with zipfile.ZipFile(archive, "w") as output:
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, "../target")
    with pytest.raises(ValueError, match="links"):
        stager.extract_zip(archive, tmp_path / "link-out", max_bytes=1024, max_entries=10)
    archive = tmp_path / "duplicates.zip"
    make_zip(archive, [("A.txt", b"one"), ("a.txt", b"two")])
    with pytest.raises(ValueError, match="duplicate"):
        stager.extract_zip(archive, tmp_path / "duplicate-out", max_bytes=1024, max_entries=10)
    archive = tmp_path / "bomb.zip"
    make_zip(archive, [("large.txt", b"0" * 10000)])
    with pytest.raises(ValueError, match="expansion"):
        stager.extract_zip(archive, tmp_path / "bomb-out", max_bytes=10, max_entries=10)
    assert not (tmp_path / "bomb-out").exists()


def make_tar(path, *, name="python/python.exe", data=b"runtime", kind=tarfile.REGTYPE):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            member.linkname = "../outside"
        member.size = len(data) if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(data) if member.size else None)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_tar_links_and_special_files_are_rejected(tmp_path, kind):
    archive = tmp_path / "unsafe.tar.gz"
    make_tar(archive, kind=kind)
    with pytest.raises(ValueError, match="not permitted"):
        stager.extract_tar(archive, tmp_path / "out", max_bytes=1024, max_entries=10)


def test_tar_traversal_expansion_and_valid_file_hashes(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    make_tar(archive, name="../escape")
    with pytest.raises(ValueError):
        stager.extract_tar(archive, tmp_path / "unsafe-out", max_bytes=1024, max_entries=10)
    make_tar(archive, data=b"x" * 100)
    with pytest.raises(ValueError, match="byte limit"):
        stager.extract_tar(archive, tmp_path / "large-out", max_bytes=10, max_entries=10)
    receipt = stager.extract_tar(archive, tmp_path / "good-out", max_bytes=1024, max_entries=10)
    assert receipt == [{"path": "python/python.exe", "size": 100, "sha256": hashlib.sha256(b"x" * 100).hexdigest()}]
    assert (tmp_path / "good-out" / "python" / "python.exe").read_bytes() == b"x" * 100


def make_wheel(directory, name="demo", version="1.2.3"):
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n")
    return path


def test_wheel_download_is_binary_targeted_and_output_is_hash_pinned(tmp_path, monkeypatch):
    calls = []
    class FakeProcess:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            destination = Path(command[command.index("--dest") + 1])
            make_wheel(destination)
        def poll(self):
            return 0
    monkeypatch.setattr(stager.subprocess, "Popen", FakeProcess)
    result = stager.download_wheels(stager.validate_lock(lock())["wheels"], tmp_path)
    command, kwargs = calls[0]
    for flag, value in (("--platform", "win_amd64"), ("--python-version", "312"), ("--implementation", "cp"), ("--abi", "cp312")):
        assert command[command.index(flag) + 1] == value
    assert "--only-binary=:all:" in command
    assert "--isolated" in command
    assert command[command.index("-m") + 1] == "pip"
    assert "install" not in command
    assert kwargs["env"]["TEMP"] == str(tmp_path / "pip-work")
    digest = result["files"][0]["sha256"]
    assert (tmp_path / "requirements-offline.txt").read_text() == f"demo==1.2.3 --hash=sha256:{digest}\n"
    assert "--no-index" in result["offline_install_arguments"]
    assert "--require-hashes" in result["offline_install_arguments"]


def _prebuilt_lock(tmp_path):
    packaging = tmp_path / "packaging"
    wheels = packaging / "wheels"
    wheels.mkdir(parents=True)
    wheel = make_wheel(wheels, name="proxy_tools", version="0.1.0")
    raw = lock()
    raw["wheels"]["requirements"] = ["proxy-tools==0.1.0"]
    raw["wheels"]["prebuilt"] = [{"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "max_bytes": 4096}]
    path = packaging / "lock.json"
    path.write_text(json.dumps(raw))
    return path, wheel, raw


def test_reviewed_prebuilt_wheel_is_verified_and_offered_without_source_build(tmp_path, monkeypatch):
    lock_path, original, raw = _prebuilt_lock(tmp_path)
    normalized, _ = stager.load_lock(lock_path)
    assert normalized["wheels"]["prebuilt_root"] == str(lock_path.parent / "wheels")
    output = tmp_path / "output"
    output.mkdir()
    calls = []
    class FakeProcess:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append(command)
            destination = Path(command[command.index("--dest") + 1])
            assert command[command.index("--find-links") + 1] == str(destination)
            assert (destination / original.name).read_bytes() == original.read_bytes()
            assert "--only-binary=:all:" in command
            assert "download" in command and "wheel" not in command and "install" not in command
        def poll(self):
            return 0
    monkeypatch.setattr(stager.subprocess, "Popen", FakeProcess)
    result = stager.download_wheels(normalized["wheels"], output)
    assert len(calls) == 1
    assert result["files"][0]["name"] == "proxy_tools"
    assert result["files"][0]["version"] == "0.1.0"
    assert result["files"][0]["sha256"] == raw["wheels"]["prebuilt"][0]["sha256"]
    assert (output / "requirements-offline.txt").read_text().startswith("proxy_tools==0.1.0 --hash=sha256:")


def test_corrupt_prebuilt_is_rejected_before_pip_is_started(tmp_path, monkeypatch):
    lock_path, wheel, _ = _prebuilt_lock(tmp_path)
    normalized, _ = stager.load_lock(lock_path)
    wheel.write_bytes(b"corrupted reviewed wheel")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(stager.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Corrupt prebuilt bytes must stop before pip"))
    with pytest.raises(ValueError, match="hash differs"):
        stager.download_wheels(normalized["wheels"], output)


@pytest.mark.parametrize("filename", ["../outside.whl", "C:/outside.whl", "folder\\outside.whl"])
def test_prebuilt_outside_filename_is_rejected_before_pip(tmp_path, monkeypatch, filename):
    lock_path, wheel, raw = _prebuilt_lock(tmp_path)
    raw["wheels"]["prebuilt"][0]["filename"] = filename
    lock_path.write_text(json.dumps(raw))
    monkeypatch.setattr(stager.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Outside prebuilt paths must stop before pip"))
    with pytest.raises(ValueError, match="plain Windows"):
        stager.load_lock(lock_path)


def test_linked_prebuilt_parent_is_rejected_before_pip(tmp_path, monkeypatch):
    lock_path, wheel, _ = _prebuilt_lock(tmp_path)
    normalized, _ = stager.load_lock(lock_path)
    linked_root = wheel.parent
    original_lstat = Path.lstat
    def flagged_lstat(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        if path == linked_root:
            # Exercise Windows reparse metadata without requiring permission
            # to create a real symlink or changing the user's filesystem.
            from types import SimpleNamespace
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result
    monkeypatch.setattr(Path, "lstat", flagged_lstat)
    monkeypatch.setattr(stager.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Linked prebuilt parent must stop before pip"))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="junctions"):
        stager.download_wheels(normalized["wheels"], output)


def test_stage_uses_verified_cache_and_records_each_extracted_file(tmp_path, monkeypatch):
    archive = tmp_path / "runtime.zip"
    make_zip(archive, [("python.exe", b"runtime"), ("Lib/site.py", b"library")])
    data = archive.read_bytes()
    item = package(data)
    raw = lock(item)
    lock_path = tmp_path / "dependencies.json"
    lock_path.write_text(json.dumps(raw))
    cache = tmp_path / "cache"
    cached = cache / item["sha256"] / "runtime.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(data)
    monkeypatch.setattr(stager, "build_opener", lambda *_: pytest.fail("Verified cached archives need no network"))
    output = tmp_path / "out"
    receipt = stager.stage(lock_path, cache, output, skip_wheels=True)
    assert receipt["status"] == "staged"
    assert receipt["lock_sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert receipt["packages"][0]["sha256"] == item["sha256"]
    assert len(receipt["packages"][0]["files"]) == 2
    assert (output / "runtime-deps" / "python312" / "python.exe").read_bytes() == b"runtime"
    assert (output / "archives" / "python312" / "runtime.zip").read_bytes() == data
    with pytest.raises(ValueError, match="existing build evidence"):
        stager.stage(lock_path, cache, output, skip_wheels=True)


def test_failed_stage_retains_receipt_and_partial_evidence(tmp_path, monkeypatch):
    archive = tmp_path / "bad.zip"
    make_zip(archive, [("../escape", b"bad")])
    data = archive.read_bytes()
    item = package(data)
    raw = lock(item)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(raw))
    cache = tmp_path / "cache"
    cached = cache / item["sha256"] / "runtime.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(data)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        stager.stage(lock_path, cache, output, skip_wheels=True)
    receipt = json.loads((output / "dependency-stage-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"]
    assert (output / "archives" / "python312" / "runtime.zip").exists()
