from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from localpilot.process import hidden_process_creation_flags


_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|pwd|token|secret|api[-_]?key|authorization|"
    r"cookie|session|credential|client[-_]?secret|access[-_]?key)",
    re.IGNORECASE,
)
_URL_CREDENTIAL = re.compile(
    r"(?P<prefix>https?://[^/\s:@]+:)(?P<secret>[^@\s/]+)@",
    re.IGNORECASE,
)


def sanitize_command_line(parts: Iterable[object]) -> str:
    """Return a bounded command line with likely credential values redacted."""
    output: list[str] = []
    redact_next = False
    for raw in parts:
        arg = str(raw or "")
        if redact_next:
            output.append("<redacted>")
            redact_next = False
            continue

        if "=" in arg:
            key, _value = arg.split("=", 1)
            if _SENSITIVE_KEY.search(key):
                output.append(f"{key}=<redacted>")
                continue

        normalized_key = arg.lstrip("-/").rstrip(":")
        if _SENSITIVE_KEY.fullmatch(normalized_key):
            output.append(arg)
            redact_next = True
            continue

        arg = _URL_CREDENTIAL.sub(
            lambda match: match.group("prefix") + "<redacted>@",
            arg,
        )
        output.append(arg)

    return " ".join(output)[:4000]


def _powershell_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def inspect_executable_artifact(path: str) -> dict[str, Any]:
    """Capture immutable-ish executable provenance at observation time.

    Hash + file stat are always attempted locally. On Windows, file-version and
    Authenticode metadata are added. This function never executes the target.
    """
    executable = str(path or "").strip()
    if not executable:
        return {"available": False, "reason": "missing_executable_path"}

    target = Path(executable)
    try:
        stat = target.stat()
    except OSError as exc:
        return {
            "available": False,
            "reason": type(exc).__name__,
            "path": executable,
        }

    sha256 = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(chunk)
    except OSError as exc:
        return {
            "available": False,
            "reason": type(exc).__name__,
            "path": str(target),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    result: dict[str, Any] = {
        "available": True,
        "path": str(target.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "sha256": sha256.hexdigest().upper(),
        "company_name": None,
        "product_name": None,
        "file_description": None,
        "file_version": None,
        "product_version": None,
        "original_filename": None,
        "signature_status": None,
        "signer_subject": None,
        "signer_issuer": None,
    }

    if os.name != "nt":
        return result

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        result["metadata_error"] = "powershell_unavailable"
        return result

    quoted = _powershell_literal(str(target))
    script = f"""
$ErrorActionPreference = 'Stop'
$path = {quoted}
$version = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)
$signature = Get-AuthenticodeSignature -LiteralPath $path -ErrorAction SilentlyContinue
[ordered]@{{
    company_name = $version.CompanyName
    product_name = $version.ProductName
    file_description = $version.FileDescription
    file_version = $version.FileVersion
    product_version = $version.ProductVersion
    original_filename = $version.OriginalFilename
    signature_status = if ($signature) {{ [string]$signature.Status }} else {{ $null }}
    signer_subject = if ($signature -and $signature.SignerCertificate) {{
        [string]$signature.SignerCertificate.Subject
    }} else {{ $null }}
    signer_issuer = if ($signature -and $signature.SignerCertificate) {{
        [string]$signature.SignerCertificate.Issuer
    }} else {{ $null }}
}} | ConvertTo-Json -Compress
"""
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=hidden_process_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["metadata_error"] = type(exc).__name__
        return result
    if completed.returncode != 0:
        result["metadata_error"] = "powershell_failed"
        return result
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        result["metadata_error"] = "invalid_metadata_json"
        return result
    if isinstance(payload, dict):
        for key in (
            "company_name",
            "product_name",
            "file_description",
            "file_version",
            "product_version",
            "original_filename",
            "signature_status",
            "signer_subject",
            "signer_issuer",
        ):
            result[key] = payload.get(key)
    return result
