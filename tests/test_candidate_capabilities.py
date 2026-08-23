import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest

from localpilot.candidate_resources import CandidateResourceStore
from localpilot.selfdev import CandidateTools


PUBLIC_RESOLUTION = [(None, None, None, None, ("93.184.216.34", 443))]


class FakeResponse:
    def __init__(self, payload: bytes, *, url: str = "https://example.com/data.json", mime: str = "application/json"):
        self.payload = io.BytesIO(payload)
        self.url = url
        self.headers = {"Content-Type": mime, "Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        return self.payload.read(size)

    def geturl(self) -> str:
        return self.url


def _store(tmp_path: Path, payload: bytes, **kwargs) -> CandidateResourceStore:
    return CandidateResourceStore(
        tmp_path / "resources",
        quota_bytes=kwargs.pop("quota_bytes", 1024 * 1024),
        max_file_bytes=kwargs.pop("max_file_bytes", 1024 * 1024),
        opener=kwargs.pop("opener", lambda *_args, **_kwargs: FakeResponse(payload)),
        resolver=lambda *_args, **_kwargs: PUBLIC_RESOLUTION,
        **kwargs,
    )


def test_directories_are_free_and_file_budget_is_a_real_hard_ceiling(tmp_path: Path):
    tools = CandidateTools(tmp_path, max_files=2, soft_file_budget=1)

    for index in range(50):
        tools.create_project_directory(f"lab/area-{index}/nested")
    tools.write_project_file("lab/one.py", "ONE = 1\n")
    tools.write_project_file("lab/two.py", "TWO = 2\n")

    assert len(tools.directories_created) == 50
    assert "complexity=above_default_budget" in tools.complexity_report()
    with pytest.raises(RuntimeError, match="hard ceiling"):
        tools.write_project_file("lab/three.py", "THREE = 3\n")


def test_directory_traversal_and_symlink_escape_are_rejected(tmp_path: Path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    tools = CandidateTools(workspace)

    with pytest.raises(ValueError, match="traversal"):
        tools.create_project_directory("safe/../also-safe")

    link = workspace / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not available on this Windows account")
    with pytest.raises(ValueError, match="Symlinks"):
        tools.write_project_file("escape/payload.py", "x = 1\n")
    assert not (outside / "payload.py").exists()


def test_zip_creation_uses_only_safe_bounded_members(tmp_path: Path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "rows.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    tools = CandidateTools(workspace, max_zip_members=2, max_zip_bytes=1024)

    result = tools.create_zip("artifacts/dataset.zip", ["dataset"])

    assert "no content was executed" in result
    with zipfile.ZipFile(workspace / "artifacts" / "dataset.zip") as archive:
        assert archive.namelist() == ["dataset/rows.jsonl"]
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in archive.namelist())
    with pytest.raises(ValueError, match="traversal"):
        tools.create_zip("artifacts/bad.zip", ["../outside"])

    (workspace / "dataset" / "more.jsonl").write_text("{}\n", encoding="utf-8")
    limited = CandidateTools(workspace, max_zip_members=1, max_zip_bytes=1024)
    with pytest.raises(RuntimeError, match="member-count"):
        limited.create_zip("artifacts/too-many.zip", ["dataset"])


def test_default_budget_allows_repo_scale_changes_and_reports_complexity(tmp_path: Path):
    tools = CandidateTools(tmp_path)
    for index in range(101):
        tools.write_project_file(f"generated/file-{index}.py", f"VALUE = {index}\n")

    assert len(tools.files_written) == 101
    assert tools.max_files == 500
    assert "soft_budget=100" in tools.complexity_report()
    assert "complexity=above_default_budget" in tools.complexity_report()


def test_resource_download_records_hash_provenance_and_stale_source(tmp_path: Path):
    payloads = [b'{"version": 1}', b'{"version": 2}']

    def opener(*_args, **_kwargs):
        return FakeResponse(payloads.pop(0))

    store = _store(tmp_path, b"", opener=opener)
    first = store.download(
        "https://example.com/data.json", "data.json",
        candidate_branch="localpilot/candidate-model-lab", task_id="model-lab", cycle_id=7,
    )
    second = store.download(
        "https://example.com/data.json", "data.json",
        candidate_branch="localpilot/candidate-model-lab", task_id="model-lab", cycle_id=7,
    )

    records = store.records()
    assert first.sha256 == hashlib.sha256(b'{"version": 1}').hexdigest()
    assert second.sha256 == hashlib.sha256(b'{"version": 2}').hexdigest()
    assert records[0].stale is True
    assert records[1].stale is False
    assert records[1].requested_url == "https://example.com/data.json"
    assert records[1].mime_type == "application/json"
    assert records[1].candidate_branch == "localpilot/candidate-model-lab"
    assert records[1].task_id == "model-lab"
    assert records[1].cycle_id == 7


def test_resource_https_quota_executable_and_interruptibility_guards(tmp_path: Path):
    opened = []
    store = _store(
        tmp_path / "https",
        b"data",
        opener=lambda *_args, **_kwargs: opened.append(True) or FakeResponse(b"data"),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        store.download(
            "http://example.com/data.json", "data.json",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )
    assert opened == []

    redirected = _store(
        tmp_path / "redirect", b"data",
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"data", url="http://example.com/downgraded.json"
        ),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        redirected.download(
            "https://example.com/data.json", "data.json",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )
    executable_redirect = _store(
        tmp_path / "exe-redirect", b"renamed",
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"renamed", url="https://example.com/installer.exe"
        ),
    )
    with pytest.raises(PermissionError, match="installer"):
        executable_redirect.download(
            "https://example.com/data", "data.dat",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )

    quota_payloads = [b"four", b"five"]
    quota = _store(
        tmp_path / "quota", b"", quota_bytes=6, max_file_bytes=4,
        opener=lambda *_args, **_kwargs: FakeResponse(quota_payloads.pop(0)),
    )
    quota.download(
        "https://example.com/one.json", "one.json",
        candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
    )
    with pytest.raises(RuntimeError, match="quota"):
        quota.download(
            "https://example.com/two.json", "two.json",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )
    assert quota.usage() == (4, 1)

    executable = _store(tmp_path / "exe", b"MZpayload")
    with pytest.raises(PermissionError, match="Executable"):
        executable.download(
            "https://example.com/data.bin", "data.dat",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )
    assert executable.usage() == (0, 0)

    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("resource governor interrupted download")

    interrupted = _store(tmp_path / "interrupt", b"data", governor_check=interrupt)
    with pytest.raises(RuntimeError, match="resource governor"):
        interrupted.download(
            "https://example.com/data.json", "data.json",
            candidate_branch="localpilot/candidate-x", task_id="x", cycle_id=1,
        )
    assert interrupted.usage() == (0, 0)
