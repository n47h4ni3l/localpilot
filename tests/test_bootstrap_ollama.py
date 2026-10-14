import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows bootstrap integration")


def run_functions(tmp_path, body):
    wrapper = tmp_path / "functions.ps1"
    wrapper.write_text(
        "param($Bootstrap, $Body)\n$ErrorActionPreference='Stop'\n"
        "$tokens=$null; $errors=$null\n"
        "$ast=[Management.Automation.Language.Parser]::ParseFile($Bootstrap,[ref]$tokens,[ref]$errors)\n"
        "if ($errors.Count) { throw $errors[0] }\n"
        "$functions=$ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]},$true)\n"
        "foreach($fn in $functions) { . ([scriptblock]::Create($fn.Extent.Text)) }\n"
        ". ([scriptblock]::Create([IO.File]::ReadAllText($Body)))\n", encoding="utf-8")
    body_path = tmp_path / "case.ps1"
    body_path.write_text(body, encoding="utf-8")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
         str(ROOT / "scripts/bootstrap.ps1"), str(body_path)], capture_output=True, text=True, timeout=30,
    )


def test_python_launcher_precedes_store_alias(tmp_path):
    launcher = tmp_path / "py.ps1"
    launcher.write_text("$global:LASTEXITCODE=0; '3.12.1'", encoding="utf-8")
    result = run_functions(tmp_path,
        f"$launcher='{launcher}'\n"
        "function Get-Command { param($Name,$ErrorAction) if($Name -eq 'py') { return [pscustomobject]@{Source=$launcher} }; throw 'Store alias was considered before usable py' }\n"
        "Find-CompatiblePython | ConvertTo-Json\n")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["PrefixArgs"] == ["-3.12"]


def test_store_alias_is_not_launched(tmp_path):
    result = run_functions(tmp_path,
        r"$value=Test-Python311 -Executable 'C:\Users\test\AppData\Local\Microsoft\WindowsApps\python.exe'" + "\n"
        "if($null -ne $value) { throw 'Store alias accepted' }; 'unusable'\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unusable"


def test_missing_private_model_is_not_offered_as_public_download(tmp_path):
    result = run_functions(tmp_path,
        "function Read-OllamaModels { [pscustomobject]@{Ready=$true; Names=@()} }\n"
        "function Read-Host { throw 'Unexpected public download offer' }\n"
        "Ensure-OllamaModel -Model 'nestra:p1.2'\n")
    assert result.returncode != 0
    assert "Restore that model" in result.stderr
    assert "Unexpected public" not in result.stderr


@pytest.mark.parametrize("mode", ["ready", "offline", "malformed"])
def test_server_readiness_is_http_based_and_bounded(tmp_path, mode):
    body = {"ready": "[pscustomobject]@{models=@([pscustomobject]@{name='gpt-oss:20b'})}",
            "offline": "throw 'connection refused'", "malformed": "[pscustomobject]@{version='client'}"}[mode]
    result = run_functions(tmp_path,
        "$env:OLLAMA_HOST='127.0.0.1:11434'\n"
        "function Invoke-RestMethod { param($Uri,$TimeoutSec,$ErrorAction) "
        "if($TimeoutSec -ne 5 -or $Uri.AbsolutePath -ne '/api/tags') { throw 'Unbounded/incorrect probe' }; " + body + " }\n"
        "Read-OllamaModels | ConvertTo-Json\n")
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["Ready"] is (mode == "ready")
    assert value["Names"] == (["gpt-oss:20b"] if mode == "ready" else [])


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_unavailable_model_drive_requires_explicit_local_fallback(tmp_path, allow_fallback):
    result = run_functions(tmp_path,
        fr"$env:USERPROFILE='{tmp_path}'; $env:OLLAMA_MODELS='Z:\unavailable\models'" + "\n"
        + r"function Test-Path { param($LiteralPath,$PathType) if($LiteralPath -eq 'Z:\') { return $false }; Microsoft.PowerShell.Management\Test-Path @PSBoundParameters }" + "\n"
        "function Read-Host { param($Prompt) 'y' }\n"
        + ("Assert-OllamaModelDirectory\n" if allow_fallback else "Assert-OllamaModelDirectory -Unattended\n")
        + "'verified-local-path'\n")
    assert (result.returncode == 0) is allow_fallback, result.stdout + result.stderr
    if allow_fallback:
        assert (tmp_path / ".ollama/models").is_dir()
        assert not list((tmp_path / ".ollama/models").glob(".localpilot-write-check-*"))
    else:
        assert "OLLAMA_MODELS" in result.stderr and "unavailable" in result.stderr
        assert not (tmp_path / ".ollama").exists()


@pytest.mark.parametrize("mode, expected_pulls", [("resume", 2), ("false_success", 3), ("decline", 0), ("private_missing", 0)])
def test_model_install_requires_verified_inventory_and_bounded_resume(tmp_path, mode, expected_pulls):
    marker = tmp_path / "pull-count.txt"
    result = run_functions(tmp_path,
        "$script:pulls=0; $ollama='Invoke-TestPull'\n"
        "function Invoke-TestPull { param($Operation,$Model) $script:pulls++; $global:LASTEXITCODE=0 }\n"
        "function Read-OllamaModels { [pscustomobject]@{Ready=$true; Names=@("
        + ("if($script:pulls -ge 2) { 'gpt-oss:20b' }" if mode == "resume" else "")
        + ")} }\nfunction Read-Host { param($Prompt) 'n' }\n"
        + "try { Ensure-OllamaModel -Model '" + ("nestra:p1.2" if mode == "private_missing" else "gpt-oss:20b") + "' "
        + ("" if mode == "decline" else "-Unattended")
        + f"; 'Bootstrap complete.' }} finally {{ $script:pulls | Set-Content -LiteralPath '{marker}' }}\n")
    assert int(marker.read_text(encoding="utf-8-sig").strip()) == expected_pulls
    assert (result.returncode == 0) is (mode == "resume"), result.stdout + result.stderr
    assert ("Bootstrap complete." in result.stdout) is (mode == "resume")
    if mode == "private_missing":
        assert "Restore that model" in result.stderr
