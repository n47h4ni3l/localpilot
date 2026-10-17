"""Read-only launcher validation and isolated functions; never starts a runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch-nestra.ps1"
PS = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(os.name != "nt" or not PS, reason="Windows PowerShell launcher")


def _installation(tmp_path):
    root = tmp_path / "Nestra test installation"
    app, runtime = root / "app", root / "runtime"
    paths = (
        "app/.venv/Scripts/python.exe", "app/.venv/Scripts/pythonw.exe",
        "runtime/python/python/python.exe", "runtime/ollama/ollama.exe",
        "runtime/git/cmd/git.exe", "runtime/git/bin/bash.exe", "runtime/gh/bin/gh.exe",
        "runtime/claude/claude.exe",
        "app/scripts/launch-nestra.ps1",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not executable: synthetic validation fixture")
    (root / "models").mkdir()
    (app / "localpilot-data").mkdir()
    (app / "localpilot-data/broker-token").write_text("A" * 43)
    config = ('[agent]\nscaffold_mode = "guide_first"\ndata_dir = "localpilot-data"\n'
              '[model]\nprovider = "ollama"\nname = "nestra:p1.2"\n'
              '[selfdev]\nimplementation_backend = "local_tools"\ndeveloper_model = "nestra:p1.2"\nimplementation_model = "nestra:p1.2"\n'
              f'implementation_executable = {json.dumps(str(runtime / "claude" / "claude.exe"))}\n'
              '[desktop]\nhost = "127.0.0.1"\nport = 8766\n')
    (app / "localpilot.toml").write_text(config)
    metadata = {
        "schema_version": 1, "install_directory": str(root), "app_root": str(app),
        "runtime_root": str(runtime), "models_root": str(root / "models"),
        "config_path": str(app / "localpilot.toml"),
        "ollama_host": "http://127.0.0.1:11435", "broker_host": "http://127.0.0.1:8766",
        "ollama_port": 11435, "broker_port": 8766,
    }
    (app / ".nestra-runtime.json").write_text(json.dumps(metadata))
    (root / "tasks.json").write_text(json.dumps({
        "format_version": 1, "owner_sid": "S-1-5-21-123-456-789-1001", "install_directory": str(root),
        "tasks": {role: "Nestra test " + role for role in ("Desktop", "Broker", "Worker")},
    }))
    return root


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _verify(root, role="Desktop"):
    return subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER),
                           "-Role", role, "-InstallDirectory", str(root), "-VerifyOnly"],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("role", ["Desktop", "Broker", "Worker"])
@pytest.mark.parametrize("directory", ["ordinary", "folder # with comment marker"])
def test_verify_only_does_not_execute_or_write(tmp_path, role, directory):
    root = _installation(tmp_path / directory)
    before = _snapshot(root)
    result = _verify(root, role)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["status"] == "verified"
    assert value["model"] == "nestra:p1.2"
    assert value["guide_only"] is True
    assert (root / "runtime/claude/claude.exe").is_file()
    assert _snapshot(root) == before
    assert not (root / "state").exists()


@pytest.mark.parametrize("field,value", [("ollama_host", "http://127.0.0.1:11434"), ("broker_port", 8765),
                                         ("app_root", "C:\\another-installation")])
def test_cross_installation_or_production_endpoints_are_rejected(tmp_path, field, value):
    root = _installation(tmp_path)
    path = root / "app/.nestra-runtime.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    before = _snapshot(root)
    result = _verify(root)
    assert result.returncode != 0
    assert _snapshot(root) == before


@pytest.mark.parametrize("original,replacement", [
    ('name = "nestra:p1.2"', 'name = "gpt-oss:20b"'),
    ('scaffold_mode = "guide_first"', 'scaffold_mode = "legacy_strict"'),
    ('implementation_backend = "local_tools"', 'implementation_backend = "claude_code"'),
    ('implementation_model = "nestra:p1.2"', 'implementation_model = "foreign-model"'),
])
def test_wrong_model_guide_or_earlier_claude_backend_is_rejected(tmp_path, original, replacement):
    root = _installation(tmp_path)
    path = root / "app/localpilot.toml"
    path.write_text(path.read_text().replace(original, replacement))
    result = _verify(root)
    assert result.returncode != 0
    assert "accepted bundle identity" in result.stderr


@pytest.mark.parametrize("executable", ["runtime/gh/bin/gh.exe", "runtime/claude/claude.exe"])
def test_missing_packaged_runtime_fails_without_global_fallback(tmp_path, executable):
    root = _installation(tmp_path)
    (root / executable).unlink()
    result = _verify(root)
    assert result.returncode != 0
    assert "runtime file is missing" in result.stderr


def test_private_claude_path_cannot_be_redirected_to_another_program(tmp_path):
    root = _installation(tmp_path)
    config = root / "app/localpilot.toml"
    config.write_text(config.read_text().replace(
        json.dumps(str(root / "runtime/claude/claude.exe")), json.dumps(str(root / "runtime/gh/bin/gh.exe")),
    ))
    result = _verify(root)
    assert result.returncode != 0
    assert "Nestra Claude Code tool must use" in result.stderr


def _functions(tmp_path, root, operation, mutation=""):
    wrapper = tmp_path / "inspect-launcher-functions.ps1"
    wrapper.write_text(
        "param($Launcher,$Root,$Operation,$Mutation)\n"
        "$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest\n"
        "$tokens=$null;$errors=$null\n"
        "$ast=[Management.Automation.Language.Parser]::ParseFile($Launcher,[ref]$tokens,[ref]$errors)\n"
        "if($errors.Count){throw ($errors | Out-String)}\n"
        "$functions=$ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)\n"
        "foreach($function in $functions){Invoke-Expression $function.Extent.Text}\n"
        "function Start-Process {throw 'No process may execute in these tests.'}\n"
        "$c=Get-NestraLaunchConfiguration $Root\n"
        "if($Operation -eq 'environment'){\n"
        "  Set-NestraPrivateEnvironment $c\n"
        "  @{claude_config=$env:CLAUDE_CONFIG_DIR;bash=$env:CLAUDE_CODE_GIT_BASH_PATH;path=$env:PATH} | ConvertTo-Json -Compress;exit 0\n"
        "}\n"
        "if($Operation -eq 'port'){\n"
        "  $script:rows=@([pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=8766;RemoteAddress='0.0.0.0';RemotePort=0;OwningProcess=987;State=$Mutation})\n"
        "  if($Mutation -in @('Established','TimeWait')){$script:rows[0].RemoteAddress='127.0.0.1';$script:rows[0].RemotePort=50001}\n"
        "  if($Mutation -eq 'nonloopback'){$script:rows[0].LocalAddress='0.0.0.0'}\n"
        "  if($Mutation -eq 'twoowners'){$script:rows+=([pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=8766;RemoteAddress='0.0.0.0';RemotePort=0;OwningProcess=988;State='Bound'})}\n"
        "  if($Mutation -eq 'mixed'){$script:rows+=([pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=8766;RemoteAddress='127.0.0.1';RemotePort=50001;OwningProcess=988;State='Established'})}\n"
        "  $script:portMutation=$Mutation\n"
        "  function Get-NetTCPConnection {param($LocalPort,$ErrorAction,$State) if($PSBoundParameters.ContainsKey('State')){throw 'State filtering hides bound server endpoints'};if($LocalPort -ne 8766){throw 'Wrong port'};return $script:rows}\n"
        "  function Get-CimInstance {param($ClassName,$Filter,$ErrorAction)\n"
        "    if($Filter -cne 'ProcessId = 987'){throw 'Wrong endpoint owner lookup'}\n"
        "    $owner=if($script:portMutation -eq 'wrongpid'){988}else{987}\n"
        "    return [pscustomobject]@{ProcessId=$owner;CreationDate=[DateTime]::UtcNow}\n"
        "  }\n"
        "  $found=Get-NestraPortProcess 8766\n"
        "  if($null -eq $found){'absent'}else{$found.ProcessId};exit 0\n"
        "}\n"
        "if($Operation -eq 'tasks'){\n"
        "  $c.Tasks.owner_sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value\n"
        "  $script:taskConfig=$c;$script:taskMutation=$Mutation\n"
        "  function Get-ScheduledTask {param($TaskName,$ErrorAction)\n"
        "    $r=$TaskName.Replace('Nestra test ','');$c=$script:taskConfig\n"
        "    $level=if($r -eq 'Desktop'){'Limited'}else{'Highest'}\n"
        "    $args='-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File '+(ConvertTo-NestraLaunchArgument $c.Files.Launcher)+' -Role '+$r+' -InstallDirectory '+(ConvertTo-NestraLaunchArgument $c.Root)\n"
        "    $t=[pscustomobject]@{Principal=[pscustomobject]@{UserId=$c.Tasks.owner_sid;RunLevel=$level;LogonType='Interactive'};Actions=@([pscustomobject]@{Execute=(Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe');Arguments=$args;WorkingDirectory=$c.App})}\n"
        "    if($script:taskMutation -eq 'level'){$t.Principal.RunLevel='Highest'}\n"
        "    if($script:taskMutation -eq 'owner'){$t.Principal.UserId='S-1-5-21-999-999-999-999'}\n"
        "    if($script:taskMutation -eq 'action'){$t.Actions[0].Arguments+=' foreign'}\n"
        "    return $t\n"
        "  }\n"
        "  Assert-NestraLaunchTasks $c;'verified';exit 0\n"
        "}\n"
        "if($Operation -eq 'exact'){\n"
        "  $a=Get-NestraBrokerArguments $c\n"
        "  $p=[pscustomobject]@{ExecutablePath=$c.Files.Python;CommandLine=((ConvertTo-NestraLaunchArgument $c.Files.Python)+' '+(Join-NestraLaunchArguments $a))}\n"
        "  if($Mutation -eq 'foreign'){$p.CommandLine+=' extra'}\n"
        "  Assert-NestraExactProcess $p $c.Files.Python $a 8766; 'verified'; exit 0\n"
        "}\n"
        "if($Operation -eq 'redirect'){\n"
        "  $script:redirectArgs=Get-NestraBrokerArguments $c\n"
        "  $script:redirectParent=[pscustomobject]@{ProcessId=900;CreationDate=[DateTime]::UtcNow.AddSeconds(-1);ExecutablePath=$c.Files.Python;CommandLine=((ConvertTo-NestraLaunchArgument $c.Files.Python)+' '+(Join-NestraLaunchArguments $script:redirectArgs))}\n"
        "  $p=[pscustomobject]@{ProcessId=901;ParentProcessId=900;CreationDate=[DateTime]::UtcNow;ExecutablePath=$c.Files.BasePython;CommandLine=((ConvertTo-NestraLaunchArgument $c.Files.BasePython)+' '+(Join-NestraLaunchArguments $script:redirectArgs))}\n"
        "  if($Mutation -eq 'parent'){$script:redirectParent.CommandLine+=' foreign'}\n"
        "  if($Mutation -eq 'reuse'){$script:redirectParent.CreationDate=[DateTime]::UtcNow.AddSeconds(1)}\n"
        "  function Get-CimInstance {param($ClassName,$Filter,$ErrorAction) if($Filter -cne 'ProcessId = 900'){throw 'Wrong redirector parent lookup'}; return $script:redirectParent}\n"
        "  Assert-NestraExactProcess $p $c.Files.Python $script:redirectArgs 8766 $c.Files.BasePython; 'verified';exit 0\n"
        "}\n"
        "$script:launchConfig=$c\n"
        "$script:live=[pscustomobject]@{ProcessId=987;CreationDate=[DateTime]::UtcNow;ExecutablePath=$null;CommandLine=$null}\n"
        "function Get-NestraPortProcess {param($Port) return $script:live}\n"
        "function Get-NestraAccountSid {param($Account) if($Account -ne 'fixture-user'){throw 'Wrong process owner'};return $script:launchConfig.Tasks.owner_sid}\n"
        "$created=([DateTimeOffset]$script:live.CreationDate).ToUnixTimeMilliseconds()/1000.0\n"
        "$script:reply=[pscustomobject]@{pid=987;elevated=$true;runtime_identity=[pscustomobject]@{\n"
        "  root=$c.App;config_path=$c.Config;data_dir=$c.Data;executable=$c.Files.Python;host_executable=$c.Files.BasePython;\n"
        "  command_line=@($c.Files.Python)+@(Get-NestraBrokerArguments $c);\n"
        "  created_at_unix=$created;user='fixture-user';desktop_host='127.0.0.1';desktop_port=8766}}\n"
        "function Invoke-RestMethod {param($Uri,$Headers,$TimeoutSec)\n"
        "  if($Headers.Authorization -cne ('Bearer '+('A'*43))){throw 'Wrong token'}\n"
        "  if($Uri -cne 'http://127.0.0.1:8766/v1/broker/status'){throw 'Wrong endpoint'}\n"
        "  return $script:reply\n"
        "}\n"
        "if($Mutation -eq 'root'){$script:reply.runtime_identity.root='C:\\foreign'}\n"
        "if($Mutation -eq 'pid'){$script:reply.pid=988}\n"
        "if($Mutation -eq 'start'){$script:reply.runtime_identity.created_at_unix-=60}\n"
        "if($Mutation -eq 'owner'){$script:reply.runtime_identity.user='foreign-user'}\n"
        "if($Mutation -eq 'command'){$script:reply.runtime_identity.command_line+=@('foreign')}\n"
        "Assert-NestraAuthenticatedBroker $c ($Operation -eq 'full') | Out-Null\n"
        "'verified'\n", encoding="utf-8",
    )
    return subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
                           str(LAUNCHER), str(root), operation, mutation],
                          capture_output=True, text=True, timeout=30)


def test_standard_desktop_verifies_elevated_broker_without_protected_commandline(tmp_path):
    result = _functions(tmp_path, _installation(tmp_path), "auth")
    assert result.returncode == 0, result.stderr
    assert "verified" in result.stdout


def test_claude_tool_environment_uses_private_config_and_git_bash(tmp_path):
    root = _installation(tmp_path)
    result = _functions(tmp_path, root, "environment")
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)
    assert Path(environment["claude_config"]) == root / "state/claude"
    assert Path(environment["bash"]) == root / "runtime/git/bin/bash.exe"
    assert str(root / "runtime/claude") in environment["path"]
    assert not (root / "state").exists()


@pytest.mark.parametrize("state", ["Listen", "Bound", "NONE", "mixed"])
def test_peerless_loopback_server_endpoint_is_discovered_across_states(tmp_path, state):
    result = _functions(tmp_path, _installation(tmp_path), "port", state)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "987"


@pytest.mark.parametrize("state", ["Established", "TimeWait"])
def test_connections_with_remote_peers_are_not_server_endpoints(tmp_path, state):
    result = _functions(tmp_path, _installation(tmp_path), "port", state)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "absent"


@pytest.mark.parametrize("mutation", ["nonloopback", "twoowners", "wrongpid"])
def test_bound_endpoint_cannot_bypass_local_address_or_exact_pid_checks(tmp_path, mutation):
    result = _functions(tmp_path, _installation(tmp_path), "port", mutation)
    assert result.returncode != 0


@pytest.mark.parametrize("mutation,expected", [("", 0), ("level", 1), ("owner", 1), ("action", 1)])
def test_registered_task_owner_action_and_desktop_privilege_are_verified(tmp_path, mutation, expected):
    result = _functions(tmp_path, _installation(tmp_path), "tasks", mutation)
    assert (result.returncode != 0) == bool(expected), result.stderr


@pytest.mark.parametrize("mutation", ["root", "pid", "start", "owner", "command"])
def test_authenticated_foreign_or_reused_broker_is_rejected(tmp_path, mutation):
    result = _functions(tmp_path, _installation(tmp_path), "auth", mutation)
    assert result.returncode != 0


def test_elevated_reuse_requires_full_executable_and_command_identity(tmp_path):
    result = _functions(tmp_path, _installation(tmp_path), "full")
    assert result.returncode != 0
    assert "unverified executable or command" in result.stderr


@pytest.mark.parametrize("mutation,expected", [("", 0), ("parent", 1), ("reuse", 1)])
def test_private_venv_redirector_requires_exact_live_parent(tmp_path, mutation, expected):
    result = _functions(tmp_path, _installation(tmp_path), "redirect", mutation)
    assert (result.returncode != 0) == bool(expected), result.stderr


@pytest.mark.parametrize("mutation,expected", [("", 0), ("foreign", 1)])
def test_exact_process_command_with_spaces_is_verified(tmp_path, mutation, expected):
    result = _functions(tmp_path, _installation(tmp_path), "exact", mutation)
    assert (result.returncode != 0) == bool(expected), result.stderr
