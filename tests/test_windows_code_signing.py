from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "windows-code-signing.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows signature policy")
PUBLISHER = "CN=Approved Publisher, O=Approved Publisher, C=AU"


def _run_policy(tmp_path: Path, body: str, **environment: str) -> subprocess.CompletedProcess[str]:
    harness = tmp_path / "policy-check.ps1"
    harness.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "Set-StrictMode -Version Latest\n"
        ". $env:LOCALPILOT_SIGNING_HELPER\n" + body,
        encoding="utf-8",
    )
    env = os.environ.copy()
    # A pytest process started by PowerShell 7 can inherit its module search path.
    # Let the selected PowerShell executable load its own built-in security module.
    env.pop("PSModulePath", None)
    env.update(
        LOCALPILOT_SIGNING_HELPER=str(HELPER),
        LOCALPILOT_TEST_PUBLISHER=PUBLISHER,
        **environment,
    )
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )


# These certificate objects are policy fixtures, not issued certificates. Nothing
# in this suite creates signing identities, signs a release, or installs trust.
VALID_SIGNATURE = r"""
$script:signature = [pscustomobject]@{
    Status = 'Valid'
    SignatureType = 'Authenticode'
    SignerCertificate = [pscustomobject]@{
        Subject = $env:LOCALPILOT_TEST_PUBLISHER
        Issuer = 'CN=Public Code Signing CA'
        PublicKey = [pscustomobject]@{ Oid = [pscustomobject]@{ Value = '1.2.840.113549.1.1.1' } }
        Extensions = @([pscustomobject]@{
            Oid = [pscustomobject]@{ Value = '2.5.29.37' }
            EnhancedKeyUsages = @([pscustomobject]@{ Value = '1.3.6.1.5.5.7.3.3' })
        })
    }
    TimeStamperCertificate = [pscustomobject]@{ Subject = 'CN=Trusted Timestamp Authority' }
}
function Get-AuthenticodeSignature {
    [CmdletBinding()]
    param([string]$LiteralPath)
    $script:signature
}
function Get-WindowsSignTool {
    param([string]$SignToolPath)
    'Invoke-TestSignTool'
}
function Invoke-TestSignTool {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    ConvertTo-Json -InputObject $Arguments -Compress |
        Add-Content -LiteralPath $env:LOCALPILOT_TEST_TOOL_LOG
    $global:LASTEXITCODE = [int]$env:LOCALPILOT_TEST_TOOL_EXIT
    'verification-diagnostic'
}
"""


@pytest.mark.parametrize(
    ("mutation", "tool_exit", "message", "tool_called"),
    [
        ("$script:signature.Status = 'NotSigned'", 0, "signature is not valid", False),
        ("$script:signature.Status = 'HashMismatch'", 0, "signature is not valid", False),
        ("$script:signature.SignatureType = 'Catalog'", 0, "catalog signature is insufficient", False),
        ("$script:signature.SignerCertificate.Subject = 'CN=Another Publisher'", 0, "approved publisher", False),
        (
            "$script:signature.SignerCertificate.Subject = $env:LOCALPILOT_TEST_PUBLISHER.ToLowerInvariant()",
            0,
            "approved publisher",
            False,
        ),
        (
            "$script:signature.SignerCertificate.Issuer = $script:signature.SignerCertificate.Subject",
            0,
            "self-signed",
            False,
        ),
        ("$script:signature.SignerCertificate.Extensions = @()", 0, "Code Signing", False),
        (
            "$script:signature.SignerCertificate.Extensions[0].EnhancedKeyUsages[0].Value = '1.3.6.1.5.5.7.3.1'",
            0,
            "Code Signing",
            False,
        ),
        ("$script:signature.TimeStamperCertificate = $null", 0, "trusted timestamp", False),
        ("", 1, "exit 1", True),
        ("", 2, "exit 2", True),
    ],
)
def test_signature_gate_rejects_untrusted_policy_boundaries(tmp_path, mutation, tool_exit, message, tool_called):
    artifact = tmp_path / "artifact [literal].exe"
    artifact.write_bytes(b"test policy fixture")
    log = tmp_path / "verification.jsonl"
    result = _run_policy(
        tmp_path,
        VALID_SIGNATURE + mutation + "\n"
        "Assert-WindowsCodeSignature -Path $env:LOCALPILOT_TEST_ARTIFACT "
        "-ExpectedPublisher $env:LOCALPILOT_TEST_PUBLISHER\n",
        LOCALPILOT_TEST_ARTIFACT=str(artifact),
        LOCALPILOT_TEST_TOOL_LOG=str(log),
        LOCALPILOT_TEST_TOOL_EXIT=str(tool_exit),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert message.lower() in (result.stdout + result.stderr).lower()
    assert log.exists() is tool_called


@pytest.mark.parametrize("key_oid", ["1.2.840.113549.1.1.1", "1.2.840.10045.2.1"])
def test_success_verifies_every_embedded_signature_and_requires_timestamp_without_pipeline_output(tmp_path, key_oid):
    artifact = tmp_path / "artifact [literal].exe"
    artifact.write_bytes(b"test policy fixture")
    log = tmp_path / "verification.jsonl"
    result = _run_policy(
        tmp_path,
        VALID_SIGNATURE + f"$script:signature.SignerCertificate.PublicKey.Oid.Value = '{key_oid}'\n" + r"""
$objects = @(Assert-WindowsCodeSignature -Path $env:LOCALPILOT_TEST_ARTIFACT `
    -ExpectedPublisher $env:LOCALPILOT_TEST_PUBLISHER)
if ($objects.Count -ne 0) { throw 'Signature assertion leaked pipeline output.' }
""",
        LOCALPILOT_TEST_ARTIFACT=str(artifact),
        LOCALPILOT_TEST_TOOL_LOG=str(log),
        LOCALPILOT_TEST_TOOL_EXIT="0",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(log.read_text(encoding="utf-8-sig")) == ["verify", "/pa", "/all", "/tw", "/v", str(artifact)]
    assert "verification-diagnostic" not in result.stdout


@pytest.mark.parametrize(("callback_exit", "tool_exit"), [(0, 0), (19, 0), (0, 2)])
def test_signer_callback_is_synchronous_then_verifies_each_exact_artifact(tmp_path, callback_exit, tool_exit):
    artifacts = [tmp_path / "first [literal].exe", tmp_path / "second.exe"]
    for artifact in artifacts:
        artifact.write_bytes(b"test policy fixture")
    callback = tmp_path / "authorized-provider.ps1"
    callback.write_text(
        "param([string[]]$Paths)\n"
        "ConvertTo-Json -InputObject $Paths -Compress | Set-Content -LiteralPath $env:LOCALPILOT_TEST_CALLBACK_LOG\n"
        "'provider-output-must-not-leak'\n"
        "exit ([int]$env:LOCALPILOT_TEST_CALLBACK_EXIT)\n",
        encoding="utf-8",
    )
    callback_log = tmp_path / "signed-paths.json"
    tool_log = tmp_path / "verified-paths.jsonl"
    result = _run_policy(
        tmp_path,
        VALID_SIGNATURE + r"""
$objects = @(Invoke-WindowsArtifactSigning `
    -Paths @($env:LOCALPILOT_TEST_ARTIFACT, $env:LOCALPILOT_TEST_SECOND_ARTIFACT, $env:LOCALPILOT_TEST_ARTIFACT) `
    -SigningScriptPath $env:LOCALPILOT_TEST_CALLBACK `
    -ExpectedPublisher $env:LOCALPILOT_TEST_PUBLISHER)
if ($objects.Count -ne 0) { throw 'Signing operation leaked pipeline output.' }
""",
        LOCALPILOT_TEST_ARTIFACT=str(artifacts[0]),
        LOCALPILOT_TEST_SECOND_ARTIFACT=str(artifacts[1]),
        LOCALPILOT_TEST_CALLBACK=str(callback),
        LOCALPILOT_TEST_CALLBACK_LOG=str(callback_log),
        LOCALPILOT_TEST_CALLBACK_EXIT=str(callback_exit),
        LOCALPILOT_TEST_TOOL_LOG=str(tool_log),
        LOCALPILOT_TEST_TOOL_EXIT=str(tool_exit),
    )
    assert json.loads(callback_log.read_text(encoding="utf-8-sig")) == [str(path) for path in artifacts]
    assert "provider-output-must-not-leak" not in result.stdout
    if callback_exit:
        assert result.returncode != 0
        assert "provider failed" in (result.stdout + result.stderr)
        assert not tool_log.exists(), "A failed provider must never enter artifact verification"
    elif tool_exit:
        assert result.returncode != 0
        assert "exit 2" in (result.stdout + result.stderr)
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        calls = [json.loads(line) for line in tool_log.read_text(encoding="utf-8-sig").splitlines()]
        assert [call[-1] for call in calls] == [str(path) for path in artifacts]


def test_invalid_configuration_never_invokes_signing_provider(tmp_path):
    callback = tmp_path / "authorized-provider.ps1"
    callback.write_text("throw 'Provider was invoked before configuration validation.'\n", encoding="utf-8")
    result = _run_policy(
        tmp_path,
        VALID_SIGNATURE + r"""
Invoke-WindowsArtifactSigning -Paths @($env:LOCALPILOT_TEST_ARTIFACT) `
    -SigningScriptPath $env:LOCALPILOT_TEST_CALLBACK -ExpectedPublisher $env:LOCALPILOT_TEST_PUBLISHER
""",
        LOCALPILOT_TEST_ARTIFACT=str(tmp_path / "missing.exe"),
        LOCALPILOT_TEST_CALLBACK=str(callback),
    )
    assert result.returncode != 0
    assert "Provider was invoked" not in (result.stdout + result.stderr)


def test_signtool_resolution_uses_literal_explicit_path_and_latest_sdk(tmp_path):
    explicit = tmp_path / "approved [sdk]" / "custom-signtool.exe"
    explicit.parent.mkdir()
    explicit.write_bytes(b"MZ")
    sdk_bin = tmp_path / "sdk" / "Windows Kits" / "10" / "bin"
    old = sdk_bin / "10.0.9.0" / "x64" / "signtool.exe"
    latest = sdk_bin / "10.0.10.0" / "x64" / "signtool.exe"
    for tool in [old, latest]:
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"MZ")
    result = _run_policy(
        tmp_path,
        r"""
$explicit = Get-WindowsSignTool -SignToolPath $env:LOCALPILOT_TEST_EXPLICIT_TOOL
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
${env:ProgramFiles(x86)} = $env:LOCALPILOT_TEST_SDK
$env:ProgramFiles = $env:LOCALPILOT_TEST_SDK
$automatic = Get-WindowsSignTool
ConvertTo-Json -InputObject @($explicit, $automatic) -Compress
""",
        LOCALPILOT_TEST_EXPLICIT_TOOL=str(explicit),
        LOCALPILOT_TEST_SDK=str(sdk_bin.parents[2]),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == [str(explicit), str(latest)]


def test_signtool_explicit_missing_path_fails_without_falling_back(tmp_path):
    result = _run_policy(
        tmp_path,
        "Get-WindowsSignTool -SignToolPath $env:LOCALPILOT_TEST_EXPLICIT_TOOL\n",
        LOCALPILOT_TEST_EXPLICIT_TOOL=str(tmp_path / "missing-signtool.exe"),
    )
    assert result.returncode != 0
    assert "missing-signtool.exe" in result.stderr


def test_real_unsigned_pe_is_rejected_before_signtool(tmp_path):
    launcher = Path(sysconfig.get_path("purelib")) / "pip" / "_vendor" / "distlib" / "t64.exe"
    if not launcher.is_file():
        pytest.skip("Installed pip does not include the unsigned Windows launcher fixture")
    artifact = tmp_path / "unsigned.exe"
    shutil.copyfile(launcher, artifact)
    assert artifact.read_bytes()[:2] == b"MZ"
    result = _run_policy(
        tmp_path,
        r"""
function Get-WindowsSignTool { throw 'SignTool must not run for an unsigned artifact.' }
$signature = Get-AuthenticodeSignature -LiteralPath $env:LOCALPILOT_TEST_ARTIFACT
if ([string]$signature.Status -ne 'NotSigned') { throw 'The PE fixture was unexpectedly signed.' }
Assert-WindowsCodeSignature -Path $env:LOCALPILOT_TEST_ARTIFACT `
    -ExpectedPublisher $env:LOCALPILOT_TEST_PUBLISHER
""",
        LOCALPILOT_TEST_ARTIFACT=str(artifact),
    )
    assert result.returncode != 0
    assert "signature is not valid" in (result.stdout + result.stderr)
    assert "NotSigned" in (result.stdout + result.stderr)
    assert "SignTool must not run" not in (result.stdout + result.stderr)
