from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from localpilot import nestra_bundle_model as bundle


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def model_fixture(tmp_path, monkeypatch):
    # A small synthetic manifest exercises every preserved Ollama layer. Tests
    # replace the pinned identities locally; release CLI offers no override.
    source = tmp_path / "production"
    source.mkdir()
    bodies = {
        "application/vnd.docker.container.image.v1+json": b'{"model_family":"gpt-oss"}',
        "application/vnd.ollama.image.model": b"GGUF-test-weights" * 17,
        "application/vnd.ollama.image.template": b"{{ .System }} custom template {{ .Prompt }}\r\n",
        "application/vnd.ollama.image.params": b'{"temperature":1}',
        "application/vnd.ollama.image.system": b"preserved system",
        "application/vnd.ollama.image.license": b"preserved license",
    }
    descriptors = []
    for media_type, body in bodies.items():
        digest = _hash(body)
        _write(source / "blobs" / f"sha256-{digest}", body)
        descriptors.append({"mediaType": media_type, "digest": f"sha256:{digest}", "size": len(body)})
    raw = json.dumps({"schemaVersion": 2, "config": descriptors[0], "layers": descriptors[1:]}, separators=(",", ":")).encode()
    monkeypatch.setattr(bundle, "ACCEPTED_MANIFEST_SHA256", _hash(raw))
    monkeypatch.setattr(bundle, "ACCEPTED_GGUF_SHA256", _hash(bodies["application/vnd.ollama.image.model"]))
    monkeypatch.setattr(bundle, "COPY_CHUNK_BYTES", 7)
    manifest_path = source / bundle._MANIFEST_PATHS[0]
    _write(manifest_path, raw)
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({
        "package": 3, "accepted_as_lineage_head": True,
        "deployment": {"model": "nestra:p1.2", "digest": _hash(raw),
                       "gguf_sha256": bundle.ACCEPTED_GGUF_SHA256,
                       "guide_only": True, "temperature": 1.0},
    }))
    return source, lineage, raw, bodies


def test_roundtrip_preserves_exact_identity_and_all_layers_without_touching_source(tmp_path, model_fixture):
    source, lineage, raw, bodies = model_fixture
    original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    target = tmp_path / "pack"
    expected = bundle.inspect_source(source, lineage)
    assert bundle.export_model(source, target, lineage) == expected
    assert bundle.verify_bundle(target) == expected
    private = tmp_path / "private-models"
    assert bundle.install_model(target, private) == expected
    for name in bundle._MANIFEST_PATHS[:2]:
        assert (private / name).read_bytes() == raw
        assert _hash((private / name).read_bytes()) == bundle.ACCEPTED_MANIFEST_SHA256
    assert len(list((private / "blobs").iterdir())) == len(bodies)
    for body in bodies.values():
        assert (private / "blobs" / f"sha256-{_hash(body)}").read_bytes() == body
    assert original == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_inspect_only_checks_metadata_and_sizes(monkeypatch, model_fixture):
    source, lineage, _, _ = model_fixture
    def forbidden(*_args, **_kwargs):
        pytest.fail("inspect must not stream/hash model blobs")
    monkeypatch.setattr(bundle, "_stream_blob", forbidden)
    assert bundle.inspect_source(source, lineage)["model"] == "nestra:p1.2"


def test_model_blob_reads_are_bounded(tmp_path, monkeypatch, model_fixture):
    source, lineage, _, _ = model_fixture
    real_open = Path.open
    sizes = []

    class BoundedInput:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size=-1):
            assert size == bundle.COPY_CHUNK_BYTES
            sizes.append(size)
            return self.stream.read(size)

    def checked_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return BoundedInput(stream) if mode == "rb" and path.parent.name == "blobs" else stream

    monkeypatch.setattr(Path, "open", checked_open)
    pack = tmp_path / "pack"
    bundle.export_model(source, pack, lineage)
    bundle.verify_bundle(pack)
    assert len(sizes) > 2


@pytest.mark.parametrize("layout", bundle._MANIFEST_PATHS)
def test_source_accepts_both_ollama_manifest_layouts(layout, model_fixture):
    source, lineage, raw, _ = model_fixture
    (source / bundle._MANIFEST_PATHS[0]).unlink()
    _write(source / layout, raw)
    assert bundle.inspect_source(source, lineage)["manifest_sha256"] == _hash(raw)


def test_conflicting_source_alias_refuses_export(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    _write(source / bundle._MANIFEST_PATHS[1], b'{"unexpected":"model"}')
    with pytest.raises(ValueError, match="digest"):
        bundle.export_model(source, tmp_path / "pack", lineage)
    assert not (tmp_path / "pack").exists()


@pytest.mark.parametrize("operation", ["export", "install"])
def test_existing_destination_is_never_overwritten(tmp_path, operation, model_fixture):
    source, lineage, _, _ = model_fixture
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    if operation == "export":
        call = lambda: bundle.export_model(source, target, lineage)
    else:
        pack = tmp_path / "pack"
        bundle.export_model(source, pack, lineage)
        call = lambda: bundle.install_model(pack, target)
    with pytest.raises(FileExistsError):
        call()
    assert marker.read_text() == "keep"
    assert list(target.iterdir()) == [marker]


def test_destinations_inside_source_are_rejected_without_creation(model_fixture):
    source, lineage, _, _ = model_fixture
    with pytest.raises(ValueError, match="separate"):
        bundle.export_model(source, source / "pack", lineage)
    assert not (source / "pack").exists()


def test_corrupt_blob_preserves_failure_but_never_writes_completion_spec(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    blob = source / "blobs" / f"sha256-{bundle.ACCEPTED_GGUF_SHA256}"
    blob.write_bytes(b"x" * blob.stat().st_size)
    pack = tmp_path / "pack"
    with pytest.raises(ValueError, match="SHA-256"):
        bundle.export_model(source, pack, lineage)
    assert pack.exists()
    assert not (pack / "bundle-model.json").exists()
    assert blob.read_bytes() == b"x" * blob.stat().st_size


def test_install_verifies_blob_before_exposing_any_alias(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    pack = tmp_path / "pack"
    bundle.export_model(source, pack, lineage)
    blob = pack / "blobs" / f"sha256-{bundle.ACCEPTED_GGUF_SHA256}"
    blob.write_bytes(b"x" * blob.stat().st_size)
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="SHA-256"):
        bundle.install_model(pack, private)
    assert not (private / "manifests").exists()
    assert not (private / "manifests-v2").exists()
    assert not (private / "nestra-bundle-installed.json").exists()


def test_bundle_spec_cannot_redirect_blob_paths(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    pack = tmp_path / "pack"
    bundle.export_model(source, pack, lineage)
    metadata = json.loads((pack / "bundle-model.json").read_bytes())
    metadata["blobs"][0]["sha256"] = "../outside"
    (pack / "bundle-model.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="specification"):
        bundle.install_model(pack, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_wrong_lineage_is_rejected_before_destination_creation(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    value = json.loads(lineage.read_text())
    value["deployment"]["model"] = "nestra:p1.2-alt"
    lineage.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact accepted"):
        bundle.export_model(source, tmp_path / "pack", lineage)
    assert not (tmp_path / "pack").exists()


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        bundle._json(b'{"model":"nestra:p1.2","model":"other"}')


def test_linked_blob_is_rejected(tmp_path, model_fixture):
    source, lineage, _, _ = model_fixture
    blob = source / "blobs" / f"sha256-{bundle.ACCEPTED_GGUF_SHA256}"
    outside = tmp_path / "external-blob"
    blob.rename(outside)
    try:
        blob.symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links")
    with pytest.raises(ValueError, match="Linked or reparse"):
        bundle.inspect_source(source, lineage)


def test_windows_reparse_points_are_rejected_without_following(tmp_path, monkeypatch):
    target = tmp_path / "junction"
    real_lstat = Path.lstat

    class ReparseStat:
        st_mode = 0
        st_file_attributes = 0x400

    def fake_lstat(path):
        return ReparseStat() if path == target else real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="Linked or reparse"):
        bundle._check_no_links(target)


def test_metadata_has_bounded_read(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (bundle._MAX_METADATA_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        bundle._small_read(path)
