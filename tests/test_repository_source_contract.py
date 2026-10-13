from pathlib import Path
import os
import subprocess

import pytest

from localpilot.study import RepositoryGroundingValidator
from localpilot.repository_source import repository_source_fingerprint


def _write(root, relative, content="source\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _git(root, *arguments):
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def test_tracked_source_edit_delete_and_untracked_add_invalidate(tmp_path):
    _git(tmp_path, "init", "--quiet")
    path = _write(tmp_path, "localpilot/component.py", "class Original: pass\n")
    _git(tmp_path, "add", "localpilot/component.py")
    original = repository_source_fingerprint(tmp_path)
    path.write_text("class Revised: pass\n", encoding="utf-8")
    revised = repository_source_fingerprint(tmp_path)
    assert revised != original
    path.unlink()
    deleted = repository_source_fingerprint(tmp_path)
    assert deleted != revised and "localpilot/component.py" not in dict(deleted)
    _write(tmp_path, "localpilot/new_component.py", "class Added: pass\n")
    added = repository_source_fingerprint(tmp_path)
    assert added != deleted and "localpilot/new_component.py" in dict(added)


def test_git_metadata_keeps_tracked_inputs_and_excludes_ignored_untracked_data(tmp_path):
    _git(tmp_path, "init", "--quiet")
    _write(tmp_path, ".gitignore", "private/\n*.custom\n")
    _write(tmp_path, "fixtures/source.custom")
    _git(tmp_path, "add", "-f", "fixtures/source.custom")
    _write(tmp_path, "private/operator.json")
    _write(tmp_path, "untracked.custom")
    _write(tmp_path, "docs/new guide.md")
    _write(tmp_path, "training/outputs/run/config.json")
    _write(tmp_path, "training/reports/scorecard.json")
    _write(tmp_path, "weights.safetensors")
    _git(tmp_path, "add", "training/outputs", "training/reports", "weights.safetensors")
    names = set(dict(repository_source_fingerprint(tmp_path)))
    assert names == {".gitignore", "fixtures/source.custom", "docs/new guide.md"}
    assert names == RepositoryGroundingValidator(root=tmp_path).live_evidence_index()["paths"]


def test_source_distribution_contract_when_git_is_unavailable(tmp_path, monkeypatch):
    import localpilot.repository_source as contract

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("Git is not installed")

    monkeypatch.setattr(contract.subprocess, "run", unavailable)
    _write(tmp_path, "training/manifests/accepted.json")
    _write(tmp_path, "tests/fixtures/expected.txt")
    _write(tmp_path, "training/outputs/run/config.json")
    _write(tmp_path, "unknown.payload")
    names = set(dict(repository_source_fingerprint(tmp_path)))
    assert names == {"training/manifests/accepted.json", "tests/fixtures/expected.txt"}


def test_source_selector_does_not_follow_linked_files_or_directories(tmp_path, monkeypatch):
    source = _write(tmp_path, "source.py")
    linked_file = _write(tmp_path, "linked.py")
    linked_child = _write(tmp_path, "linked_directory/child.py")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink",
        lambda path: path in {linked_file, linked_child.parent} or original(path),
    )
    assert [path for path, _ in repository_source_fingerprint(tmp_path)] == [
        source.relative_to(tmp_path).as_posix(),
    ]


def test_source_selector_prunes_windows_junctions(tmp_path, monkeypatch):
    _write(tmp_path, "source.py")
    junction = _write(tmp_path, "junction/child.py").parent
    original = Path.resolve
    monkeypatch.setattr(
        Path, "resolve",
        lambda path, *args, **kwargs: tmp_path.parent / "external" if path == junction
        else original(path, *args, **kwargs),
    )
    assert set(dict(repository_source_fingerprint(tmp_path))) == {"source.py"}


@pytest.mark.parametrize("relative", [
    "training/outputs/run/checkpoint-123/adapter_model.safetensors",
    "training/outputs/run/config.json", "training/reports/scorecard.json",
    "training/reports/trace.log", "output/transcript.json", "logs/operator.jsonl",
    "localpilot-data/learning.json", "unsloth_compiled_cache/generated.py",
    ".cache/study/result.json", "tools/WindowsInstaller/bin/Release/generated.py",
    "tools/SystemSense.HardwareProvider/obj/build.json", "dist/package.json",
    "build/generated.py", "models/p1/pytorch_model.bin", "weights.safetensors",
    "adapter.gguf", "localpilot/_hardware/win-x64/provider.json", "localpilot.toml",
])
def test_generated_state_additions_and_changes_do_not_invalidate(tmp_path, relative):
    _write(tmp_path, "localpilot/component.py", "class Source: pass\n")
    baseline = repository_source_fingerprint(tmp_path)
    artifact = _write(tmp_path, relative, "first generated value\n")
    assert repository_source_fingerprint(tmp_path) == baseline
    artifact.write_text("second generated value\n", encoding="utf-8")
    assert repository_source_fingerprint(tmp_path) == baseline
    index = RepositoryGroundingValidator(root=tmp_path).live_evidence_index()
    assert relative not in index["paths"]


@pytest.mark.parametrize("relative", [
    "pyproject.toml", "localpilot.example.toml", ".github/workflows/tests.yml",
    "training/configs/qlora.yaml", "training/schema/example.schema.json",
    "training/manifests/eval_manifest.json", "training/lineage/package-1.json",
    "training/baselines/accepted.json", "training/datasets/seed.jsonl",
    "training/evals/cases.jsonl", "training/sources/lock.jsonl",
    "tests/fixtures/expected.json", "training/tests/test_contract.py",
    "docs/reports/architecture.md", "tools/WindowsInstaller/app.manifest",
    "tools/WindowsInstaller/Program.cs", "tools/WindowsInstaller/project.csproj",
    "scripts/launch.sh", "localpilot/webview/index.html", "localpilot/webview/app.js",
])
def test_source_configuration_docs_schema_fixtures_and_manifests_are_truth(tmp_path, relative):
    path = _write(tmp_path, relative, "# source\n")
    baseline = repository_source_fingerprint(tmp_path)
    assert relative in dict(baseline)
    assert relative in RepositoryGroundingValidator(root=tmp_path).live_evidence_index()["paths"]
    path.write_text("# changed source\n", encoding="utf-8")
    assert repository_source_fingerprint(tmp_path) != baseline


def test_generated_payloads_are_not_opened_even_with_multigigabyte_metadata(tmp_path, monkeypatch):
    source = _write(tmp_path, "source.py", "class Source: pass\n")
    artifact = _write(tmp_path, "training/outputs/checkpoint-123/adapter_model.safetensors")
    original_open, original_stat = Path.open, Path.stat
    opened = []

    def metadata(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path in {source, artifact}:
            values = list(result)
            values[6] = 3 * 1024 ** 3
            return os.stat_result(values)
        return result

    def read(path, *args, **kwargs):
        assert path != artifact, "Generated model payload must never be opened."
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", metadata)
    monkeypatch.setattr(Path, "open", read)
    fingerprint = repository_source_fingerprint(tmp_path)
    assert "source.py" in dict(fingerprint)  # No file-size limit for legitimate source.
    assert "training/outputs/checkpoint-123/adapter_model.safetensors" not in dict(fingerprint)
    assert source in opened


def test_fingerprint_order_and_paths_are_deterministic(tmp_path, monkeypatch):
    for relative in ("zeta/module.py", "alpha/schema.json", "README.md"):
        _write(tmp_path, relative)
    expected = repository_source_fingerprint(tmp_path)
    original_walk = os.walk

    def reverse_walk(*args, **kwargs):
        for directory, children, names in original_walk(*args, **kwargs):
            yield directory, children, list(reversed(names))
            children.reverse()

    monkeypatch.setattr(os, "walk", reverse_walk)
    assert repository_source_fingerprint(tmp_path) == expected
    assert [name for name, _ in expected] == sorted(name for name, _ in expected)
    assert all("\\" not in name and not name.startswith("/") for name, _ in expected)




def test_nested_standalone_source_does_not_inherit_parent_git_ignores(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init", "--quiet")
    _write(parent, ".gitignore", "nested/\n")
    nested = parent / "nested"
    _write(nested, "source.py", "class Nested: pass\n")
    assert set(dict(repository_source_fingerprint(nested))) == {"source.py"}
    assert RepositoryGroundingValidator(root=nested).live_evidence_index()["paths"] == {"source.py"}
