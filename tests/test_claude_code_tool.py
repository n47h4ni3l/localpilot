from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localpilot.config import Config
from localpilot.implementation_backend import ImplementationResult, ImplementationStatus
from localpilot.resource import ResourceState
from localpilot.safety import RiskLevel, SafetyPolicy
from localpilot.tools import registry
from localpilot.tools.claude_code import ClaudeCodeTool, _PrivateClaudeCodeBackend, claude_code_run_lock


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return completed.stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "owner"
    root.mkdir()
    files = {
        "module.py": "VALUE = 1\n",
        ".gitignore": "localpilot-data/\n",
        ".env": "TOKEN=private\n",
        "localpilot.toml": 'name = "owner-private"\n',
        ".claude/settings.json": '{"hooks": "private"}\n',
        "training/evals/private.json": '{"private":true}\n',
        "training/outputs/private.txt": "weights metadata\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "init", "--quiet", "--template=")
    _git(root, "config", "user.name", "Tool test")
    _git(root, "config", "user.email", "tool-test@invalid.local")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "credential.helper", "!echo NEVER_COPY_ME")
    _git(root, "remote", "add", "origin", "https://private.invalid/owner/repo")
    _git(root, "add", "--force", "--", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root


@pytest.fixture
def tool(repo, monkeypatch):
    cfg = Config()
    cfg.model.name = cfg.selfdev.developer_model = cfg.selfdev.implementation_model = "nestra:p1.2"
    cfg.systemsense.enabled = False
    result = ClaudeCodeTool(repo, cfg)
    monkeypatch.setattr(result.governor, "sample", lambda **kwargs: ResourceState(0, 5, 30, False, "active", False, True, "active", ""))
    monkeypatch.setattr(result.governor, "prepare_ollama_handoff", lambda *_args: {"attempted": False, "success": False})
    return result


def _backend(monkeypatch, tool, effect=None, status=ImplementationStatus.COMPLETED):
    calls = []

    class Backend:
        def run(self, request):
            calls.append(request)
            changed = ()
            if effect:
                changed = effect(request)
            return ImplementationResult(status, "claude_code", "nestra:p1.2", "A real bounded answer.", changed_paths=changed,
                                        diff_digest="a" * 64 if changed else "", usage={"tests_not_run": request.read_only})

    monkeypatch.setattr(tool, "_backend", lambda private_home: Backend())
    return calls


def test_readonly_snapshot_omits_owner_data_and_git_identity(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    result = tool.claude_code("Explain module.py")
    request = calls[0]
    candidate = request.workspace
    assert request.read_only and request.allowed_paths == ()
    assert (candidate / "module.py").read_text() == "VALUE = 1\n"
    for name in (".env", "localpilot.toml", ".claude", "training/evals", "training/outputs"):
        assert not (candidate / name).exists()
    assert _git(candidate, "remote", "-v") == ""
    assert "NEVER_COPY_ME" not in (candidate / ".git/config").read_text()
    assert not (candidate / ".git/objects/info/alternates").exists()
    assert result["model"] == "nestra:p1.2"
    assert result["tests_not_run"] is True and result["tests_independently_verified"] is False
    assert not result["pending_review"] and not result["applied_to_production"]
    assert not (tool.root / "localpilot-data/claude-code/active.json").exists()


def test_write_proposal_preserves_production_and_includes_new_file_patch(tool, monkeypatch):
    def edit(request):
        (request.workspace / "module.py").write_text("VALUE = 2\n")
        (request.workspace / "new.py").write_text("NEW = True\n")
        return ("module.py", "new.py")

    calls = _backend(monkeypatch, tool, edit)
    before = _git(tool.root, "rev-parse", "HEAD")
    result = tool.claude_code("Improve the module", ["module.py", "new.py"])
    patch = Path(result["proposal_path"]).read_bytes()
    assert b"VALUE = 2" in patch and b"NEW = True" in patch
    assert result["proposal_sha256"] == hashlib.sha256(patch).hexdigest()
    assert result["pending_review"] and result["review_status"] == "pending_review"
    assert result["tests_independently_verified"] is False
    assert (tool.root / "module.py").read_text() == "VALUE = 1\n"
    assert not (tool.root / "new.py").exists()
    assert _git(tool.root, "rev-parse", "HEAD") == before
    assert not calls[0].read_only
    assert json.loads((Path(result["workspace"]).parent / "receipt.json").read_text())["pending_review"]


@pytest.mark.parametrize("path", ["../other.py", "C:\\owner.py", "/owner.py", "module.py:stream", "a/../../x.py", "*.py", "module.py)", "nul.py", "a/./x.py", "a./x.py"])
def test_invalid_paths_never_start_child(tool, monkeypatch, path):
    calls = _backend(monkeypatch, tool)
    with pytest.raises(ValueError):
        tool.claude_code("Change this", [path])
    assert calls == []


@pytest.mark.parametrize("path", [".git/config", ".github/workflows/ci.yml", ".env", "localpilot.toml", ".claude/settings.json", "training/evals/private.json", "training/reports/run.json", "training/outputs/model.json", "module.exe"])
def test_protected_paths_never_start_child(tool, monkeypatch, path):
    calls = _backend(monkeypatch, tool)
    with pytest.raises(ValueError, match="protected"):
        tool.claude_code("Change this", [path])
    assert calls == []


def test_dirty_source_is_preserved_and_never_mistaken_for_committed_snapshot(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    (tool.root / "module.py").write_text("OWNER_DRAFT = True\n")
    result = tool.claude_code("Inspect the module")
    assert (calls[0].workspace / "module.py").read_text() == "VALUE = 1\n"
    assert result["source_basis"] == "committed_head_snapshot"
    assert result["working_tree_included"] is False
    assert "uncommitted working-tree edits are not included" in calls[0].prompt
    assert (tool.root / "module.py").read_text() == "OWNER_DRAFT = True\n"
    assert not (tool.root / "localpilot-data/claude-code/active.json").exists()


def test_exact_model_selectors_required(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    tool.config.selfdev.implementation_model = "another-model"
    with pytest.raises(ValueError, match="exact accepted local model"):
        tool.claude_code("Inspect the module")
    assert calls == []


def test_reversible_permission_is_enforced_even_when_called_directly(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    tool.config.safety.auto_allow_reversible = False
    with pytest.raises(PermissionError, match="reversible"):
        tool.claude_code("Inspect the module")
    assert calls == []


def test_capacity_checked_before_snapshot_and_during_inference(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    monkeypatch.setattr(tool.governor, "sample", lambda **kwargs: ResourceState(0, 5, 95, False, "pressure", False, False, "active", "memory pressure"))
    with pytest.raises(RuntimeError, match="deferred"):
        tool.claude_code("Inspect the module")
    with pytest.raises(RuntimeError, match="memory boundary"):
        tool._resource_guard(during_inference=True)
    assert calls == []


def test_handoff_reclaims_before_capacity_check_without_loading_or_weakening_guard(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    memory = {"percent": 91}
    observed = []

    def reclaim(base_url, model):
        observed.append((base_url, model))
        # Kernel serialization must already be held during reclamation.
        assert (tool.root / tool.config.agent.data_dir / "claude-code/active.json").exists()
        memory["percent"] = 50
        return {"attempted": True, "success": True, "pid": 17}

    monkeypatch.setattr(tool.governor, "prepare_ollama_handoff", reclaim)
    monkeypatch.setattr(tool.governor, "sample", lambda **kwargs: ResourceState(
        0, 5, memory["percent"], False, "active", False, memory["percent"] <= 82,
        "active", "memory pressure" if memory["percent"] > 82 else ""))
    result = tool.claude_code("Read module.py")
    assert len(calls) == 1 and observed == [(tool.config.selfdev.implementation_base_url, "nestra:p1.2")]
    assert result["memory_handoff"]["success"] is True
    assert tool.config.resource.max_memory_percent_for_background == 82


def test_handoff_still_defers_if_reclaim_does_not_restore_capacity(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    monkeypatch.setattr(tool.governor, "prepare_ollama_handoff", lambda *_args: {
        "attempted": True, "success": True, "pid": 17})
    monkeypatch.setattr(tool.governor, "sample", lambda **kwargs: ResourceState(
        0, 5, 91, False, "pressure", False, False, "active", "memory pressure"))
    with pytest.raises(RuntimeError, match="deferred"):
        tool.claude_code("Read module.py")
    assert calls == []


def test_shared_lock_blocks_nested_invocations_and_preserves_receipt(tool, monkeypatch):
    calls = _backend(monkeypatch, tool)
    data = tool.root / tool.config.agent.data_dir
    with claude_code_run_lock(data):
        with pytest.raises(RuntimeError, match="already running"):
            tool.claude_code("Inspect the module")
        assert (data / "claude-code/active.json").exists()
    assert calls == []
    assert not (data / "claude-code/active.json").exists()


def _directory_link(target: Path, link: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_configured_data_junction_is_supported_without_moving_owner_history(tool, tmp_path, monkeypatch):
    calls = _backend(monkeypatch, tool)
    actual = tmp_path / "historical-data"
    actual.mkdir()
    history = actual / "owner-history.jsonl"
    history.write_bytes(b'owner history remains here\n')
    alias = tool.root / "relocated-data"
    _directory_link(actual, alias)
    tool.config.agent.data_dir = "relocated-data"
    result = tool.claude_code("Read module.py")
    assert Path(result["workspace"]).is_relative_to(actual / "claude-code")
    assert len(calls) == 1 and history.read_bytes() == b'owner history remains here\n'
    assert alias.is_dir()


def test_junction_and_resolved_data_paths_share_one_kernel_lock(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    _directory_link(actual, alias)
    with claude_code_run_lock(alias):
        with pytest.raises(RuntimeError, match="already running"):
            with claude_code_run_lock(actual):
                pytest.fail("both aliases must refer to the same lock")


def test_junction_below_configured_data_anchor_is_still_rejected(tool, tmp_path, monkeypatch):
    calls = _backend(monkeypatch, tool)
    data = tool.root / tool.config.agent.data_dir
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _directory_link(outside, data / "claude-code")
    with pytest.raises(ValueError, match="link or junction"):
        tool.claude_code("Read module.py")
    assert calls == [] and list(outside.iterdir()) == []


def test_private_child_home_and_cloud_credentials(tool, tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "OWNER_SECRET")
    backend = _PrivateClaudeCodeBackend(private_home=tmp_path / "private", model="nestra:p1.2")
    env = backend._environment()
    assert "OWNER_SECRET" not in json.dumps(env)
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ollama"
    assert Path(env["HOME"]) == tmp_path / "private"
    assert Path(env["CLAUDE_CONFIG_DIR"]).parent == tmp_path / "private"
    assert env["HOME"] == env["USERPROFILE"]


def test_kernel_lock_survives_missing_metadata_and_blocks_other_process(tool):
    data = tool.root / tool.config.agent.data_dir
    code = (
        "import sys\nfrom localpilot.tools.claude_code import claude_code_run_lock\n"
        "try:\n with claude_code_run_lock(sys.argv[1]): pass\n"
        "except RuntimeError:\n sys.exit(17)\n"
    )
    with claude_code_run_lock(data):
        (data / "claude-code/active.json").unlink()
        completed = subprocess.run([sys.executable, "-c", code, str(data)], capture_output=True, timeout=20)
        assert completed.returncode == 17, completed.stderr
    with claude_code_run_lock(data):
        assert (data / "claude-code/active.json").exists()


def test_backend_failure_keeps_exact_status_without_claiming_completion(tool, monkeypatch):
    _backend(monkeypatch, tool, status=ImplementationStatus.CLI_ERROR)
    result = tool.claude_code("Inspect the module")
    assert result["status"] == "cli_error" and result["backend"] == "claude_code"
    assert result["applied_to_production"] is False and result["proposal_path"] == ""


def test_partial_proposal_without_test_result_remains_explicitly_unverified(tool, monkeypatch):
    def edit(request):
        (request.workspace / "module.py").write_text("VALUE = 2\n")
        return ("module.py",)
    _backend(monkeypatch, tool, edit, status=ImplementationStatus.CANDIDATE_TEST_FAILURE)
    result = tool.claude_code("Propose a change", ["module.py"])
    assert result["status"] == "candidate_test_failure"
    assert "test verification is incomplete" in result["summary"]
    assert result["tests_status"] == "unverified" and result["pending_review"]
    assert Path(result["proposal_path"]).is_file()


def test_registry_exposes_reversible_first_class_tool(tool):
    tools = registry(tool.root, config=tool.config)
    spec = tools["claude_code"]
    assert spec.risk is RiskLevel.REVERSIBLE
    assert spec.fn.__name__ == "claude_code"
    assert SafetyPolicy().permits_without_confirmation(spec.risk)
    assert not SafetyPolicy(auto_allow_reversible=False).permits_without_confirmation(spec.risk)
