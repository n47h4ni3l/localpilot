from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import os
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from localpilot.audit import AuditLog


_BLOCKED_SUFFIXES = {
    ".app", ".bat", ".cmd", ".com", ".cpl", ".dll", ".dmg",
    ".exe", ".hta", ".iso", ".jar", ".js", ".jse", ".lnk", ".msi",
    ".msp", ".msu", ".ps1", ".reg", ".scr", ".sh", ".sys", ".vbe",
    ".vbs", ".wsf",
}
_BLOCKED_MIME_PARTS = (
    "executable", "x-msdownload", "x-msdos-program", "x-sh", "x-shellscript",
)
_CHUNK_SIZE = 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class CandidateResource:
    id: int
    path: Path
    requested_url: str
    final_url: str
    fetched_at: str
    sha256: str
    mime_type: str
    extension: str
    size_bytes: int
    candidate_branch: str
    task_id: str
    cycle_id: int
    stale: bool


class CandidateResourceStore:
    """Bounded, provenance-preserving storage for inert candidate resources."""

    def __init__(
        self,
        root: str | Path,
        *,
        quota_bytes: int,
        max_file_bytes: int,
        governor_check: Callable[[], None] | None = None,
        audit: AuditLog | None = None,
        opener: Callable[..., object] = urllib.request.urlopen,
        resolver: Callable[..., list] = socket.getaddrinfo,
    ) -> None:
        self.root = Path(root).resolve()
        self.files_root = self.root / "files"
        self.quota_bytes = max(1, int(quota_bytes))
        self.max_file_bytes = max(1, min(int(max_file_bytes), self.quota_bytes))
        self.governor_check = governor_check or (lambda: None)
        self.audit = audit
        self._opener = opener
        self._resolver = resolver
        self.files_root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "resources.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS candidate_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stored_path TEXT NOT NULL,
                    requested_url TEXT NOT NULL,
                    final_url TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    candidate_branch TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    cycle_id INTEGER NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS candidate_resources_source_idx
                    ON candidate_resources(requested_url, task_id, candidate_branch, id DESC);
                CREATE INDEX IF NOT EXISTS candidate_resources_hash_idx
                    ON candidate_resources(sha256);
                """
            )

    @staticmethod
    def _validate_https_url(url: str, resolver: Callable[..., list]) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(str(url))
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Candidate resources require a credential-free HTTPS URL.")
        host = parsed.hostname.rstrip(".").lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            raise ValueError("Local/private hosts are not valid candidate resource sources.")
        try:
            literal = ipaddress.ip_address(host)
            addresses = [literal]
        except ValueError:
            try:
                addresses = {
                    ipaddress.ip_address(item[4][0])
                    for item in resolver(host, parsed.port or 443, type=socket.SOCK_STREAM)
                }
            except OSError as exc:
                raise ValueError(f"Could not resolve candidate resource host: {host}") from exc
        if not addresses or any(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
            for address in addresses
        ):
            raise ValueError("Local/private hosts are not valid candidate resource sources.")
        return parsed

    @staticmethod
    def _safe_filename(filename: str, url: str) -> tuple[str, str]:
        proposed = str(filename or Path(urllib.parse.urlparse(url).path).name or "resource.dat")
        if proposed != Path(proposed).name or proposed in {".", ".."} or "\x00" in proposed:
            raise ValueError("Resource filename must be one plain relative filename.")
        suffix = Path(proposed).suffix.lower()
        url_suffix = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).suffix.lower()
        blocked = suffix if suffix in _BLOCKED_SUFFIXES else url_suffix
        if blocked in _BLOCKED_SUFFIXES:
            raise PermissionError(f"Executable or installer resources are blocked: {blocked}")
        return proposed[:180], suffix

    @staticmethod
    def _looks_executable(head: bytes, mime_type: str) -> bool:
        lowered = mime_type.lower()
        return (
            any(token in lowered for token in _BLOCKED_MIME_PARTS)
            or head.startswith(b"MZ")
            or head.startswith(b"\x7fELF")
            or head[:4] in {b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe"}
            or head.startswith(b"#!")
        )

    def usage(self) -> tuple[int, int]:
        total = 0
        members = 0
        for path in self.files_root.iterdir():
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
                members += 1
        return total, members

    def resolve_relative(self, relative_path: str) -> Path:
        raw = Path(str(relative_path))
        if raw.is_absolute():
            raise ValueError("Absolute resource paths are not allowed.")
        if any(part == ".." for part in raw.parts):
            raise ValueError("Resource path traversal is not allowed.")
        current = self.files_root
        for part in raw.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError("Symlinks are not allowed in candidate resources.")
        path = (self.files_root / raw).resolve()
        if path == self.files_root or self.files_root not in path.parents:
            raise ValueError("Path escapes candidate resource store.")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != self.files_root):
            raise ValueError("Symlinks are not allowed in candidate resources.")
        return path

    def download(
        self,
        url: str,
        filename: str,
        *,
        candidate_branch: str,
        task_id: str,
        cycle_id: int,
    ) -> CandidateResource:
        requested = self._validate_https_url(url, self._resolver)
        clean_name, requested_suffix = self._safe_filename(filename, url)
        self.governor_check()
        request = urllib.request.Request(
            requested.geturl(),
            headers={"User-Agent": "LocalPilot/0.2 candidate-resource"},
            method="GET",
        )
        part = self.root / f".{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        mime_type = "application/octet-stream"
        final_url = requested.geturl()
        head = b""
        complete = False
        try:
            with self._opener(request, timeout=20) as response:
                final_url = str(getattr(response, "geturl", lambda: requested.geturl())())
                self._validate_https_url(final_url, self._resolver)
                self._safe_filename(clean_name, final_url)
                headers = getattr(response, "headers", {})
                mime_type = str(headers.get("Content-Type", "application/octet-stream")).split(";", 1)[0].strip().lower()
                content_length = headers.get("Content-Length")
                if content_length and int(content_length) > self.max_file_bytes:
                    raise RuntimeError("Candidate resource exceeds the per-file byte limit.")
                with part.open("wb") as handle:
                    while True:
                        self.governor_check()
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        if not head:
                            head = chunk[:16]
                            if self._looks_executable(head, mime_type):
                                raise PermissionError("Executable or installer payloads are blocked.")
                        size += len(chunk)
                        if size > self.max_file_bytes:
                            raise RuntimeError("Candidate resource exceeds the per-file byte limit.")
                        handle.write(chunk)
                        digest.update(chunk)
                complete = True
        except PermissionError:
            raise
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Candidate resource download failed: {exc}") from exc
        finally:
            if part.exists() and not complete:
                part.unlink()

        if self._looks_executable(head, mime_type):
            part.unlink(missing_ok=True)
            raise PermissionError("Executable or installer payloads are blocked.")
        sha256 = digest.hexdigest()
        inferred_suffix = requested_suffix or mimetypes.guess_extension(mime_type) or ".dat"
        if inferred_suffix.lower() in _BLOCKED_SUFFIXES:
            part.unlink(missing_ok=True)
            raise PermissionError("Executable or installer resource extension is blocked.")
        stored_name = f"{sha256}{inferred_suffix.lower()}"
        destination = self.resolve_relative(stored_name)
        used, _ = self.usage()
        additional = 0 if destination.exists() else size
        if used + additional > self.quota_bytes:
            part.unlink(missing_ok=True)
            raise RuntimeError("Candidate resource-store quota would be exceeded.")
        if destination.exists():
            part.unlink(missing_ok=True)
        else:
            os.replace(part, destination)

        fetched_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE candidate_resources SET stale = 1
                WHERE requested_url = ? AND task_id = ? AND candidate_branch = ?
                  AND sha256 != ? AND stale = 0
                """,
                (requested.geturl(), str(task_id), str(candidate_branch), sha256),
            )
            cursor = connection.execute(
                """
                INSERT INTO candidate_resources (
                    stored_path, requested_url, final_url, fetched_at, sha256,
                    mime_type, extension, size_bytes, candidate_branch, task_id,
                    cycle_id, stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    stored_name, requested.geturl(), final_url, fetched_at, sha256,
                    mime_type, inferred_suffix.lower(), size, str(candidate_branch),
                    str(task_id), int(cycle_id),
                ),
            )
            resource_id = int(cursor.lastrowid)
        record = CandidateResource(
            resource_id, destination, requested.geturl(), final_url, fetched_at,
            sha256, mime_type, inferred_suffix.lower(), size, str(candidate_branch),
            str(task_id), int(cycle_id), False,
        )
        if self.audit:
            self.audit.write(
                "candidate_resource_downloaded",
                resource_id=resource_id,
                requested_url=requested.geturl(),
                final_url=final_url,
                sha256=sha256,
                mime_type=mime_type,
                extension=inferred_suffix.lower(),
                size_bytes=size,
                candidate_branch=candidate_branch,
                task_id=task_id,
                cycle_id=cycle_id,
                executable=False,
            )
        return record

    def records(self) -> list[CandidateResource]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM candidate_resources ORDER BY id"
            ).fetchall()
        return [
            CandidateResource(
                int(row["id"]), self.resolve_relative(str(row["stored_path"])),
                str(row["requested_url"]), str(row["final_url"]), str(row["fetched_at"]),
                str(row["sha256"]), str(row["mime_type"]), str(row["extension"]),
                int(row["size_bytes"]), str(row["candidate_branch"]), str(row["task_id"]),
                int(row["cycle_id"]), bool(row["stale"]),
            )
            for row in rows
        ]
