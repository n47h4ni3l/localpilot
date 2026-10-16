from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-nestra-bundle.py"
spec = importlib.util.spec_from_file_location("nestra_bundle_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def identity(path):
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "Bundle Test")
    git(source, "config", "user.email", "bundle@example.invalid")
    git(source, "remote", "add", "origin", builder.ORIGIN)
    write(source, ".gitignore", b"localpilot-data/\n*.log\n__pycache__/\n")
    for _, relative in builder._ROOT_FILES.items():
        write(source, relative, ("Release file: " + relative + "\n").encode())
    write(source, "packaging/licenses/EXAMPLE.txt", b"Test third party license\n")
    write(source, "packaging/licenses/provenance.json", b'{"source":"test fixture"}\n')
    write(source, "localpilot/__init__.py", b"# test source\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "Release fixture")
    sha = git(source, "rev-parse", "HEAD")
    git(source, "update-ref", "refs/remotes/origin/main", sha)
    write(source, "localpilot-data/private.sqlite", b"owner material must never be included")
    dependencies = tmp_path / "dependencies"
    runtime = write(dependencies, "runtime-deps/python/python/python.exe", b"runtime fixture")
    archive = write(dependencies, "archives/python/python.tar.gz", b"archive fixture")
    wheel = write(dependencies, "wheelhouse/example-1-py3-none-any.whl", b"wheel fixture")
    write(dependencies, "requirements-download.txt", b"example==1\n")
    write(dependencies, "requirements-offline.txt", ("example==1 --hash=sha256:" + identity(wheel)["sha256"] + "\n").encode())
    write(dependencies, "pip-work/ignored-temp.bin", b"temporary dependency work must stay outside payload")
    write(dependencies, "pip-download.log", b"build-time log")
    receipt = {"schema_version": 1, "status": "staged", "packages": [
        {"name": "python", "kind": "tar.gz", "path": "archives/python/python.tar.gz", **identity(archive),
         "extracted_root": "runtime-deps/python", "files": [{"path": "python/python.exe", **identity(runtime)}]}],
        "wheels": {"status": "downloaded", "files": [{"file": wheel.name, **identity(wheel)}]}}
    write(dependencies, "dependency-stage-receipt.json", json.dumps(receipt).encode())
    model = tmp_path / "model"
    model_bytes = b"GGUF miniature builder fixture; never model-loaded"
    digest = hashlib.sha256(model_bytes).hexdigest()
    manifest = write(model, "manifest.json", b'{"schemaVersion":2,"fixture":true}\n')
    model_spec = {"schema_version": 1, "model": "test-only", "manifest_sha256": identity(manifest)["sha256"],
                  "blobs": [{"sha256": digest, "size": len(model_bytes), "media_type": "application/vnd.ollama.image.model"}]}
    write(model, "bundle-model.json", json.dumps(model_spec).encode())
    write(model, "blobs/sha256-" + digest, model_bytes)
    monkeypatch.setattr(builder, "model_spec", lambda *_: model_spec)
    hardware = tmp_path / "hardware"
    write(hardware, "win-x64/LocalPilot.SystemSense.HardwareProvider.exe", b"hardware fixture")
    write(hardware, "win-x64/THIRD_PARTY_NOTICES.md", b"Test notices")
    return source, dependencies, model, hardware, sha, model_spec


def build(inputs, tmp_path, *, output=None):
    source, dependencies, model, hardware, _, _ = inputs
    return builder.build_bundle(source, dependencies, model, hardware, output or tmp_path / "output", "0.2.1-alpha.2")


def test_streamed_zip_has_exact_inventory_identity_source_history_and_model_storage(inputs, tmp_path):
    result = build(inputs, tmp_path)
    source, dependencies, model, hardware, sha, model_spec = inputs
    output = tmp_path / "output"
    archive_path = output / result["zip"]
    assert result["status"] == "built"
    assert result["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert result["size"] == archive_path.stat().st_size
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle-manifest.json"))
        assert manifest["source_sha"] == sha
        assert manifest["source_ref"] == "refs/heads/main"
        assert manifest["schema_version"] == manifest["format_version"] == 1
        assert {item["path"] for item in manifest["files"]} == names - {"bundle-manifest.json"}
        assert "README.md" in names and "THIRD_PARTY_NOTICES.md" in names
        assert "licenses/EXAMPLE.txt" in names and "licenses/provenance.json" in names
        assert not any("localpilot-data" in name or "pip-work" in name or "pip-download.log" in name for name in names)
        for item in manifest["files"]:
            data = archive.read(item["path"])
            assert item["size"] == len(data)
            assert item["sha256"] == hashlib.sha256(data).hexdigest()
            if item["path"].startswith("payload/model/blobs/"):
                assert archive.getinfo(item["path"]).compress_type == zipfile.ZIP_STORED
            else:
                assert archive.getinfo(item["path"]).compress_type == zipfile.ZIP_DEFLATED
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--no-checkout", "--branch", "main", str(output / "source.bundle"), str(clone)], capture_output=True, check=True)
    assert git(clone, "rev-parse", "HEAD") == sha
    assert "localpilot-data" not in git(clone, "ls-tree", "-r", "--name-only", "HEAD")
    assert git(source, "status", "--porcelain") == ""


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "C:/drive", "a\\b", "a//b", "NUL.txt", "a./x", "name:stream", "a/.. /b"])
def test_relative_paths_reject_windows_escape_and_aliases(bad):
    with pytest.raises(ValueError):
        builder.relative_path(bad)


def test_committed_release_wheels_cannot_drift_on_rebuild(inputs, tmp_path):
    source, dependencies, *_ = inputs
    staged = json.loads((dependencies / "dependency-stage-receipt.json").read_text())
    locked = {"schema_version": 1, "target": staged.get("target"), "files": staged["wheels"]["files"]}
    locked["files"][0]["sha256"] = "0" * 64
    write(source, "packaging/nestra-alpha2-wheels.json", json.dumps(locked).encode())
    git(source, "add", "packaging/nestra-alpha2-wheels.json")
    git(source, "commit", "-m", "Seal reviewed release wheel identity")
    git(source, "update-ref", "refs/remotes/origin/main", git(source, "rev-parse", "HEAD"))
    with pytest.raises(ValueError, match="reviewed release wheel"):
        build(inputs, tmp_path)
    assert not (tmp_path / "output").exists()


def test_source_must_be_clean_main_at_canonical_origin_and_remote_revision(inputs, tmp_path):
    source = inputs[0]
    write(source, "untracked.txt", b"must reject")
    with pytest.raises(ValueError, match="clean"):
        build(inputs, tmp_path)
    (source / "untracked.txt").unlink()
    git(source, "checkout", "-b", "feature")
    with pytest.raises(ValueError, match="main branch"):
        build(inputs, tmp_path)
    git(source, "checkout", "main")
    git(source, "remote", "set-url", "origin", "https://example.com/wrong.git")
    with pytest.raises(ValueError, match="canonical LocalPilot"):
        build(inputs, tmp_path)
    git(source, "remote", "set-url", "origin", builder.ORIGIN)
    write(source, "extra.txt", b"ahead of origin")
    git(source, "add", "extra.txt")
    git(source, "commit", "-m", "Unpublished fixture")
    with pytest.raises(ValueError, match="origin/main"):
        build(inputs, tmp_path)
    assert not (tmp_path / "output").exists()


def test_existing_output_and_unlisted_owner_files_are_preserved(inputs, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = write(output, "keep.txt", b"preserve")
    with pytest.raises(ValueError, match="preserved"):
        build(inputs, tmp_path)
    assert marker.read_bytes() == b"preserve"
    dependencies = inputs[1]
    write(dependencies, "private.sqlite", b"not an allowed dependency artifact")
    with pytest.raises(ValueError, match="Unlisted file"):
        build(inputs, tmp_path, output=tmp_path / "other-output")


def test_unlisted_python_bytecode_caches_are_omitted_and_local_files_preserved(inputs):
    dependencies = inputs[1]
    receipt_path = dependencies / "dependency-stage-receipt.json"
    raw_receipt = receipt_path.read_bytes()
    caches = [
        write(dependencies, "runtime-deps/python/python/Lib/__pycache__/argparse.cpython-312.pyc", b"generated cache"),
        write(dependencies, "runtime-deps/python/python/Lib/tkinter/__pycache__/__init__.cpython-312.pyc", b"generated tkinter cache"),
    ]
    original_bytes = {path: path.read_bytes() for path in caches}
    files = builder.dependency_files(dependencies)
    names = {name for name, _, _ in files}
    assert not any("__pycache__" in name for name in names)
    assert "payload/dependencies/runtime-deps/python/python/python.exe" in names
    assert receipt_path.read_bytes() == raw_receipt
    assert {path: path.read_bytes() for path in caches} == original_bytes


def test_unlisted_python_source_executables_and_other_bytecode_still_rejected(inputs):
    dependencies = inputs[1]
    for relative in (
        "runtime-deps/python/python/Lib/unlisted.py",
        "runtime-deps/python/python/Lib/__pycache__/payload.exe",
        "runtime-deps/python/python/Scripts/__pycache__/other.pyc",
        "runtime-deps/gh/__pycache__/other.pyc",
    ):
        path = write(dependencies, relative, b"unlisted material")
        with pytest.raises(ValueError, match="Unlisted file"):
            builder.dependency_files(dependencies)
        assert path.read_bytes() == b"unlisted material"
        path.unlink()  # Only this test's own temporary fixture file.


def test_archive_listed_python_bytecode_keeps_exact_receipt_hash(inputs, tmp_path):
    dependencies = inputs[1]
    relative = "runtime-deps/python/python/Lib/demo/__pycache__/original.cpython-312.pyc"
    path = write(dependencies, relative, b"archive bytecode")
    receipt_path = dependencies / "dependency-stage-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    expected = identity(path)
    receipt["packages"][0]["files"].append({"path": "python/Lib/demo/__pycache__/original.cpython-312.pyc", **expected})
    receipt_path.write_text(json.dumps(receipt))
    rows = builder.dependency_files(dependencies)
    name, source, pinned = next(row for row in rows if row[0] == "payload/dependencies/" + relative)
    assert pinned == expected
    source.write_bytes(b"altered bytecode")
    with zipfile.ZipFile(tmp_path / "pinned-bytecode.zip", "w") as archive, source.open("rb") as stream:
        with pytest.raises(ValueError, match="pinned identity|expected size"):
            builder.write_member(archive, name, stream, expected=pinned)


def test_corrupted_payload_hash_leaves_failed_zip_and_receipt(inputs, tmp_path):
    runtime = inputs[1] / "runtime-deps/python/python/python.exe"
    runtime.write_bytes(b"changed contents")
    with pytest.raises(ValueError, match="expected size|pinned identity"):
        build(inputs, tmp_path)
    output = tmp_path / "output"
    receipt = json.loads((output / "bundle-build-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert (output / receipt["zip"]).exists()
    assert (output / "source.bundle").exists()
    assert runtime.read_bytes() == b"changed contents"


def test_model_corruption_is_detected_during_single_zip_write_pass(inputs, tmp_path, monkeypatch):
    model = inputs[2]
    blob = next((model / "blobs").iterdir())
    data = blob.read_bytes()
    blob.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="pinned identity"):
        build(inputs, tmp_path)
    receipt = json.loads((tmp_path / "output/bundle-build-receipt.json").read_text())
    assert receipt["status"] == "failed"


def test_public_dependency_receipt_omits_owner_paths_without_mutating_local_evidence(inputs, tmp_path):
    receipt_path = inputs[1] / "dependency-stage-receipt.json"
    original = json.loads(receipt_path.read_text())
    token = "owner-tokenlike-value-5471"
    owner = "C:/Users/natha/" + token
    original["local_work_directory"] = owner
    original["packages"][0]["local_cache_path"] = owner + "/cache/runtime.tar.gz"
    original["packages"][0]["url"] = "https://www.python.org/runtime.tar.gz"
    original["wheels"]["cache_directory"] = owner + "/pip-cache"
    original["wheels"]["requested_requirements"] = ["example==1"]
    original["wheels"]["files"][0].update(name="example", version="1")
    original["wheels"]["download_command"] = [owner + "/python.exe", "-m", "pip", "download", "--index-url", "https://pypi.org/simple", "--dest", owner + "/wheelhouse", "--requirement", owner + "/requirements.txt", "--find-links", owner + "/reviewed-prebuilt"]
    raw = (json.dumps(original, indent=2) + "\n").encode()
    receipt_path.write_bytes(raw)
    result = build(inputs, tmp_path)
    assert receipt_path.read_bytes() == raw
    with zipfile.ZipFile(tmp_path / "output" / result["zip"]) as archive:
        public_bytes = archive.read("payload/dependencies/dependency-stage-receipt.json")
        public = json.loads(public_bytes)
        for name in archive.namelist():
            assert token.encode() not in archive.read(name)
            assert b"C:/Users/natha" not in archive.read(name)
        assert public["packages"][0]["sha256"] == original["packages"][0]["sha256"]
        assert public["packages"][0]["url"] == original["packages"][0]["url"]
        assert public["wheels"]["files"][0]["version"] == "1"
        assert public["wheels"]["download_command"][0] == "python"
        assert public["wheels"]["download_command"][-6:] == ["--dest", "wheelhouse", "--requirement", "requirements-download.txt", "--find-links", "wheelhouse"]
        assert "local_work_directory" not in public
        assert "cache_directory" not in public["wheels"]


def test_committed_wrapper_size_checked_before_reading_object(inputs, tmp_path):
    source = inputs[0]
    write(source, "Install Nestra.cmd", b"x" * (builder.CHUNK + 1))
    git(source, "add", "Install Nestra.cmd")
    git(source, "commit", "-m", "Oversized fixture")
    git(source, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        build(inputs, tmp_path)
    assert not (tmp_path / "output").exists()


def test_reparse_or_symlink_input_is_rejected_without_following(inputs, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked = inputs[3] / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows test user cannot create symbolic links")
    with pytest.raises(ValueError, match="reparse"):
        build(inputs, tmp_path)
