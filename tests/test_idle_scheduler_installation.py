from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


def _quote_ps(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_functions(body: str) -> dict:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if os.name != "nt" or not powershell:
        pytest.skip("Windows PowerShell is needed for scheduler installation checks")
    script = Path(__file__).resolve().parents[1] / "scripts" / "install-idle-evolve-task.ps1"
    command = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({_quote_ps(script)}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ throw ($errors | Out-String) }}
$functions = $ast.FindAll({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }}, $false)
foreach ($function in $functions) {{ . ([scriptblock]::Create($function.Extent.Text)) }}
{body}
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    return json.loads(result.stdout)


def test_worker_installer_selects_existing_runtime_without_creating_venv(tmp_path):
    root = tmp_path / "production checkout"
    root.mkdir()
    installed = tmp_path / "installed Python"
    installed.mkdir()
    python = installed / "python.exe"
    pythonw = installed / "pythonw.exe"
    python.touch()
    pythonw.touch()
    result = _run_functions(f"""
$selected = Resolve-WorkerPython -Executable {_quote_ps(pythonw)} -Root {_quote_ps(root)}
$missingRejected = $false
try {{ Resolve-WorkerPython -Executable '' -Root {_quote_ps(root)} | Out-Null }} catch {{ $missingRejected = $true }}
@{{ console = $selected.Console; gui = $selected.Gui; missingRejected = $missingRejected }} | ConvertTo-Json -Compress
""")
    assert Path(result["console"]) == python.resolve()
    assert Path(result["gui"]) == pythonw.resolve()
    assert result["missingRejected"] is True
    assert not (root / ".venv").exists()


def test_worker_installer_preserves_existing_elevation_unless_explicitly_changed():
    result = _run_functions("""
@{
    preservedHighest = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel Highest
    preservedLimited = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel Limited
    newLimited = Resolve-WorkerRunLevel -Specified $false -Administrator $false -ExistingRunLevel ''
    explicitHighest = Resolve-WorkerRunLevel -Specified $true -Administrator $true -ExistingRunLevel Limited
    explicitLimited = Resolve-WorkerRunLevel -Specified $true -Administrator $false -ExistingRunLevel Highest
} | ConvertTo-Json -Compress
""")
    assert result == {
        "preservedHighest": "Highest",
        "preservedLimited": "Limited",
        "newLimited": "Limited",
        "explicitHighest": "Highest",
        "explicitLimited": "Limited",
    }


@pytest.mark.parametrize("enabled,running", [(True, True), (True, False), (False, False)])
def test_failed_worker_replacement_restores_original_definition_and_state(enabled, running):
    result = _run_functions(f"""
$script:events = [System.Collections.Generic.List[string]]::new()
function Register-ScheduledTask {{ param($TaskName, $Xml, [switch]$Force, $ErrorAction); $script:events.Add("register:$TaskName`:$Xml") }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction); $script:events.Add("disable:$TaskName") }}
function Start-ScheduledTask {{ param($TaskName, $ErrorAction); $script:events.Add("start:$TaskName") }}
function Unregister-ScheduledTask {{ param($TaskName, $Confirm, $ErrorAction); throw 'An existing task must be restored rather than deleted' }}
Restore-PreviousWorkerTask -Name 'LocalPilot Background Worker' -Xml '<Task>old-definition</Task>' -Enabled ${str(enabled).lower()} -Running ${str(running).lower()}
@{{ events = @($script:events) }} | ConvertTo-Json -Compress
""")
    expected = ["register:LocalPilot Background Worker:<Task>old-definition</Task>"]
    if not enabled:
        expected.append("disable:LocalPilot Background Worker")
    elif running:
        expected.append("start:LocalPilot Background Worker")
    assert result["events"] == expected


@pytest.mark.parametrize("image,parent_image,expected", [
    ("selected", "unrelated", True),
    ("host", "selected", True),
    ("host", "unrelated", False),
    ("unrelated", "selected", False),
])
def test_worker_identity_requires_selected_image_or_its_own_redirector(image, parent_image, expected):
    result = _run_functions(f"""
function Get-CimInstance {{ param($ClassName, $Filter, $ErrorAction); return [pscustomobject]@{{ ExecutablePath='{parent_image}' }} }}
$candidate = [pscustomobject]@{{ ExecutablePath='{image}'; ParentProcessId=123 }}
@{{ accepted = Test-WorkerInterpreter -Process $candidate -SelectedGui selected -HostGui host }} | ConvertTo-Json -Compress
""")
    assert result["accepted"] is expected


def test_worker_identity_accepts_actual_windows_venv_redirector(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows Python redirector behavior")
    environment = tmp_path / "selected environment"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True, timeout=30)
    pythonw = environment / "Scripts" / "pythonw.exe"
    pid_file = tmp_path / "probe-pid.json"
    stop_file = tmp_path / "finish-probe"
    code = (
        "import json,os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'host':sys._base_executable})); "
        "stop=Path(sys.argv[2]); deadline=time.monotonic()+60\n"
        "while not stop.exists() and time.monotonic()<deadline: time.sleep(.1)\n"
    )
    probe = subprocess.Popen(
        [str(pythonw), "-c", code, str(pid_file), str(stop_file)],
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.1)
        identity = json.loads(pid_file.read_text())
        result = _run_functions(f"""
$candidate = Get-CimInstance Win32_Process -Filter 'ProcessId = {identity['pid']}'
@{{ accepted = Test-WorkerInterpreter -Process $candidate -SelectedGui {_quote_ps(pythonw.resolve())} -HostGui {_quote_ps(Path(identity['host']).resolve())}; image=$candidate.ExecutablePath }} | ConvertTo-Json -Compress
""")
        assert result["accepted"] is True
        assert Path(result["image"]).resolve() == Path(identity["host"]).resolve()
    finally:
        stop_file.touch()
        probe.wait(timeout=10)


def _graceful_stop_fixture(tmp_path):
    root = tmp_path / "worker checkout"
    root.mkdir()
    launcher = tmp_path / "scheduled environment" / "pythonw.exe"
    launcher.parent.mkdir()
    launcher.touch()
    pid_path = root / "background-worker.pid"
    pid_path.write_text(json.dumps({"pid": 123, "root": str(root)}), encoding="utf-8")
    return root, launcher, pid_path


def test_worker_replacement_waits_for_old_host_and_task_before_returning(tmp_path):
    root, launcher, pid_path = _graceful_stop_fixture(tmp_path)
    result = _run_functions(f"""
$script:polls = 0
$script:disabled = 0
$worker = [pscustomobject]@{{ Name='pythonw.exe'; ExecutablePath={_quote_ps(launcher)}; CreationDate='original'; CommandLine='pythonw.exe -m localpilot.background_worker --root "{root}" --config "{root / 'localpilot.toml'}"' }}
$task = [pscustomobject]@{{ TaskName='LocalPilot Background Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(launcher)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction); $script:disabled++ }}
function Get-ScheduledTask {{
    param($TaskName, $ErrorAction)
    [pscustomobject]@{{ State=$(if ($script:polls -lt 3) {{ 'Running' }} else {{ 'Ready' }}) }}
}}
function Get-CimInstance {{
    param($ClassName, $Filter, $ErrorAction)
    $script:polls++
    if ($script:polls -lt 3) {{ return $worker }}
}}
function Stop-ScheduledTask {{ throw 'A worker must never be force-stopped.' }}
function Start-Sleep {{ param($Milliseconds) }}
Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(root / 'localpilot.toml')} -TimeoutSeconds 5
$request = Get-Content -LiteralPath {_quote_ps(pid_path.with_suffix('.stop'))} -Raw | ConvertFrom-Json
@{{ disabled=$script:disabled; polls=$script:polls; target=$request.target_pid }} | ConvertTo-Json -Compress
""")
    assert result == {"disabled": 1, "polls": 3, "target": 123}


@pytest.mark.parametrize("parent_matches", [True, False])
def test_worker_stop_requires_the_previous_tasks_actual_venv_parent(tmp_path, parent_matches):
    root, launcher, pid_path = _graceful_stop_fixture(tmp_path)
    parent = str(launcher) if parent_matches else str(tmp_path / "unrelated" / "pythonw.exe")
    result = _run_functions(f"""
$script:polls = 0
$script:stopped = $false
$worker = [pscustomobject]@{{ Name='pythonw.exe'; ExecutablePath='C:\\base-python\\pythonw.exe'; ParentProcessId=99; CreationDate='original'; CommandLine='pythonw.exe -m localpilot.background_worker --root "{root}" --config "{root / 'localpilot.toml'}"' }}
$task = [pscustomobject]@{{ TaskName='LocalPilot Background Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(launcher)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction) }}
function Get-ScheduledTask {{ param($TaskName, $ErrorAction); [pscustomobject]@{{ State=$(if ($script:stopped) {{ 'Ready' }} else {{ 'Running' }}) }} }}
function Get-CimInstance {{
    param($ClassName, $Filter, $ErrorAction)
    if ($Filter -eq 'ProcessId = 99') {{ return [pscustomobject]@{{ ExecutablePath={_quote_ps(parent)} }} }}
    $script:polls++
    if ($script:polls -eq 1) {{ return $worker }}
    $script:stopped = $true
}}
function Stop-ScheduledTask {{ throw 'A worker must never be force-stopped.' }}
$errorMessage = ''
try {{ Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(root / 'localpilot.toml')} -TimeoutSeconds 5 }} catch {{ $errorMessage = $_.Exception.Message }}
@{{ error=$errorMessage; request=Test-Path -LiteralPath {_quote_ps(pid_path.with_suffix('.stop'))} }} | ConvertTo-Json -Compress
""")
    if parent_matches:
        assert result == {"error": "", "request": True}
    else:
        assert "scheduled interpreter" in result["error"]
        assert result["request"] is False


def test_worker_stop_timeout_preserves_live_process_without_force_stopping(tmp_path):
    root, launcher, pid_path = _graceful_stop_fixture(tmp_path)
    result = _run_functions(f"""
$script:clock = [DateTime]::UtcNow
$worker = [pscustomobject]@{{ Name='pythonw.exe'; ExecutablePath={_quote_ps(launcher)}; CreationDate='original'; CommandLine='pythonw.exe -m localpilot.background_worker --root "{root}" --config "{root / 'localpilot.toml'}"' }}
$task = [pscustomobject]@{{ TaskName='LocalPilot Background Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(launcher)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction) }}
function Get-ScheduledTask {{ param($TaskName, $ErrorAction); [pscustomobject]@{{ State='Running' }} }}
function Get-CimInstance {{ param($ClassName, $Filter, $ErrorAction); $worker }}
function Get-Date {{ $script:clock = $script:clock.AddSeconds(1); $script:clock }}
function Stop-ScheduledTask {{ throw 'A worker must never be force-stopped.' }}
$errorMessage = ''
try {{ Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(root / 'localpilot.toml')} -TimeoutSeconds 1 }} catch {{ $errorMessage = $_.Exception.Message }}
@{{ error=$errorMessage; request=Test-Path -LiteralPath {_quote_ps(pid_path.with_suffix('.stop'))} }} | ConvertTo-Json -Compress
""")
    assert "not force-terminated" in result["error"]
    assert result["request"] is True


@pytest.mark.parametrize("state", ["Ready", "Disabled", "Running", "Queued", "Unknown"])
def test_missing_worker_pid_requires_an_inactive_task_and_no_matching_process(tmp_path, state):
    root, launcher, pid_path = _graceful_stop_fixture(tmp_path)
    pid_path.unlink()
    result = _run_functions(f"""
$task = [pscustomobject]@{{ TaskName='LocalPilot Background Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(launcher)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction) }}
function Get-ScheduledTask {{ param($TaskName, $ErrorAction); [pscustomobject]@{{ State='{state}' }} }}
function Get-CimInstance {{ param($ClassName, $Filter, $ErrorAction) }}
function Stop-ScheduledTask {{ throw 'An unidentified worker must never be force-stopped.' }}
$errorMessage = ''
try {{ Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(root / 'localpilot.toml')} -TimeoutSeconds 5 }} catch {{ $errorMessage = $_.Exception.Message }}
@{{ error=$errorMessage; request=Test-Path -LiteralPath {_quote_ps(pid_path.with_suffix('.stop'))} }} | ConvertTo-Json -Compress
""")
    assert result["request"] is False
    if state in {"Ready", "Disabled"}:
        assert result["error"] == ""
    else:
        assert "not inactive" in result["error"]


@pytest.mark.parametrize("owner_kind", ["missing", "empty", "stale"])
def test_unowned_orphan_worker_blocks_replacement_even_when_task_is_ready(tmp_path, owner_kind):
    root, launcher, pid_path = _graceful_stop_fixture(tmp_path)
    if owner_kind == "missing":
        pid_path.unlink()
    elif owner_kind == "empty":
        pid_path.write_text("{}", encoding="utf-8")
    result = _run_functions(f"""
$task = [pscustomobject]@{{ TaskName='LocalPilot Background Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(launcher)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction) }}
function Get-ScheduledTask {{ param($TaskName, $ErrorAction); [pscustomobject]@{{ State='Ready' }} }}
function Get-CimInstance {{
    param($ClassName, $Filter, $ErrorAction)
    if ($Filter -eq "Name = 'pythonw.exe'") {{
        [pscustomobject]@{{ ExecutablePath='C:\\base-python\\pythonw.exe'; ParentProcessId=999; CommandLine='pythonw.exe -m localpilot.background_worker --root "{root}" --config "{root / 'localpilot.toml'}"' }}
    }}
}}
function Stop-ScheduledTask {{ throw 'An unidentified orphan must never be force-stopped.' }}
$errorMessage = ''
try {{ Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(root / 'localpilot.toml')} -TimeoutSeconds 5 }} catch {{ $errorMessage = $_.Exception.Message }}
@{{ error=$errorMessage; request=Test-Path -LiteralPath {_quote_ps(pid_path.with_suffix('.stop'))} }} | ConvertTo-Json -Compress
""")
    assert "configured worker process is still alive" in result["error"]
    assert result["request"] is False


def test_worker_stop_waits_for_real_isolated_worker_to_release_its_lock(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows worker replacement")
    root = tmp_path / "isolated worker checkout"
    root.mkdir()
    config = root / "localpilot.toml"
    config.write_text('[selfdev]\nenabled = false\n', encoding="utf-8")
    pythonw = Path(sys.executable).resolve().with_name("pythonw.exe")
    if not pythonw.is_file():
        pytest.skip("A Windows windowless interpreter is required")
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
    probe = subprocess.Popen(
        [str(pythonw), "-m", "localpilot.background_worker", "--root", str(root), "--config", str(config)],
        cwd=repository,
        env=env,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    pid_path = root / "localpilot-data" / "background-worker.pid"
    try:
        deadline = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < deadline:
            assert probe.poll() is None, "The isolated worker exited before recording its identity"
            time.sleep(.1)
        identity = json.loads(pid_path.read_text())
        result = _run_functions(f"""
$task = [pscustomobject]@{{ TaskName='Isolated Test Worker'; Actions=@([pscustomobject]@{{ Execute={_quote_ps(pythonw)} }}) }}
function Disable-ScheduledTask {{ param($TaskName, $ErrorAction) }}
function Get-ScheduledTask {{
    param($TaskName, $ErrorAction)
    $alive = Get-CimInstance Win32_Process -Filter 'ProcessId = {identity['pid']}'
    [pscustomobject]@{{ State=$(if ($alive) {{ 'Running' }} else {{ 'Ready' }}) }}
}}
function Stop-ScheduledTask {{ throw 'The real probe must exit cooperatively.' }}
Stop-ExistingWorkerGracefully -Task $task -PidPath {_quote_ps(pid_path)} -Root {_quote_ps(root)} -ConfigPath {_quote_ps(config)} -TimeoutSeconds 10
@{{ alive=[bool](Get-CimInstance Win32_Process -Filter 'ProcessId = {identity['pid']}'); owner=Get-Content -LiteralPath {_quote_ps(pid_path)} -Raw | ConvertFrom-Json }} | ConvertTo-Json -Compress
""")
        assert result["alive"] is False
        assert result["owner"] == {}
        assert probe.wait(timeout=5) == 0
        from localpilot.background_worker import WorkerLock

        replacement = WorkerLock(pid_path.with_suffix(".lock"))
        assert replacement.acquire(root=root), "The replacement must acquire the old worker's released lock"
        replacement.release()
    finally:
        if probe.poll() is None:
            identity = json.loads(pid_path.read_text())
            pid_path.with_suffix(".stop").write_text(json.dumps({"target_pid": identity["pid"]}), encoding="utf-8")
            probe.wait(timeout=10)
