"""Stage hash-pinned Windows x64 runtime dependencies without installing them.

Only the owner-reviewed JSON lock controls downloads. Archives use a verified,
content-addressed cache. The optional wheel resolver invokes installed pip in
download-only, binary-only mode for CPython 3.12 / win_amd64. It never imports or
executes downloaded packages. No model files belong in this lock.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from email.parser import BytesParser
import gzip
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

CHUNK = 1024 * 1024
MAX_LOCK_BYTES = 1024 * 1024
HARD_DOWNLOAD_BYTES = 8 * 1024**3
HARD_UNPACKED_BYTES = 16 * 1024**3
HARD_WHEEL_BYTES = 2 * 1024**3
HARD_ZIP_ENTRIES = 100_000
TARGET = {"platform": "win_amd64", "python_version": "312", "implementation": "cp", "abi": "cp312"}
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_PIN = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9_,.-]+\])?==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)}, *{f"LPT{i}" for i in range(1, 10)}}


def _limit(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer between 1 and {maximum}.")
    return value


def _filename(value: object) -> str:
    if (not isinstance(value, str) or not value or len(value) > 180
            or value != value.strip() or value.endswith((".", " "))
            or value in {".", ".."} or any(c in value for c in '/\\:<>"|?*')
            or any(ord(c) < 32 for c in value) or value.split(".", 1)[0].upper() in _RESERVED):
        raise ValueError("Dependency and archive names must be plain Windows-compatible filenames.")
    value.encode("utf-8")
    return value


def validate_url(url: object) -> str:
    if not isinstance(url, str) or len(url) > 4096 or any(ord(c) < 32 for c in url) or "\\" in url:
        raise ValueError("Dependency URLs must be public HTTPS URLs.")
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:
        raise ValueError("Dependency URL is malformed.") from exc
    if (parts.scheme != "https" or not host or parts.username is not None or parts.password is not None
            or parts.fragment or port not in {None, 443} or host.lower() == "localhost"
            or host.lower().endswith((".localhost", ".local")) or "." not in host):
        raise ValueError("Dependency URLs must use public HTTPS without credentials or fragments.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Private or local dependency URLs are not permitted.")
    return url


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_lock(raw: object) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Dependency lock schema_version must be 1.")
    if raw.get("target", TARGET) != TARGET:
        raise ValueError("This bundle supports only Windows x64 CPython 3.12 (cp312).")
    policy = raw.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("Dependency lock policy must be an object.")
    limits = {
        "max_total_download_bytes": _limit(policy.get("max_total_download_bytes", HARD_DOWNLOAD_BYTES), "Total archive byte limit", HARD_DOWNLOAD_BYTES),
        "max_total_unpacked_bytes": _limit(policy.get("max_total_unpacked_bytes", HARD_UNPACKED_BYTES), "Total extracted byte limit", HARD_UNPACKED_BYTES),
        "max_zip_entries": _limit(policy.get("max_zip_entries", HARD_ZIP_ENTRIES), "ZIP entry limit", HARD_ZIP_ENTRIES),
    }
    packages = raw.get("packages")
    if not isinstance(packages, list) or not 1 <= len(packages) <= 32:
        raise ValueError("Dependency lock requires 1 to 32 packages.")
    normalized = []
    names = set()
    for entry in packages:
        if not isinstance(entry, dict):
            raise ValueError("Every package must be an object.")
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Package names must use lowercase letters, digits, dot, dash or underscore.")
        _filename(name)
        if name.casefold() in names:
            raise ValueError("Package names must be unique.")
        names.add(name.casefold())
        url = validate_url(entry.get("url"))
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name}: SHA256 must be exactly 64 lowercase hexadecimal characters.")
        size = _limit(entry.get("max_bytes"), f"{name} max_bytes", HARD_DOWNLOAD_BYTES)
        kind = entry.get("kind")
        if kind not in {"zip", "tar.gz", "exe", "file"}:
            raise ValueError(f"{name}: kind must be zip, tar.gz, exe or file.")
        filename = _filename(entry.get("filename", PurePosixPath(unquote(urlsplit(url).path)).name))
        if Path(filename).suffix.lower() in {".gguf", ".safetensors", ".pt", ".pth", ".ckpt"}:
            raise ValueError("Model weight files must not be part of the runtime dependency lock.")
        if kind == "zip" and not filename.lower().endswith(".zip"):
            raise ValueError(f"{name}: ZIP packages must have a .zip filename.")
        if kind == "tar.gz" and not filename.lower().endswith(".tar.gz"):
            raise ValueError(f"{name}: tar.gz packages must have a .tar.gz filename.")
        if kind == "exe" and not filename.lower().endswith(".exe"):
            raise ValueError(f"{name}: installer packages must have an .exe filename.")
        item = {"name": name, "url": url, "sha256": digest, "max_bytes": size, "kind": kind, "filename": filename}
        if kind in {"zip", "tar.gz"}:
            item["max_unpacked_bytes"] = _limit(entry.get("max_unpacked_bytes", min(size * 20, HARD_UNPACKED_BYTES)), f"{name} max_unpacked_bytes", HARD_UNPACKED_BYTES)
        normalized.append(item)
    if sum(item["max_bytes"] for item in normalized) > limits["max_total_download_bytes"]:
        raise ValueError("Declared package download sizes exceed the total policy limit.")
    if sum(item.get("max_unpacked_bytes", 0) for item in normalized) > limits["max_total_unpacked_bytes"]:
        raise ValueError("Declared ZIP expansion sizes exceed the total policy limit.")
    wheel_spec = raw.get("wheels", {})
    if not isinstance(wheel_spec, dict):
        raise ValueError("wheels must be an object.")
    requirements = wheel_spec.get("requirements", [])
    if not isinstance(requirements, list) or len(requirements) > 128:
        raise ValueError("Wheel requirements must be a bounded list of exact version pins.")
    distributions = set()
    for requirement in requirements:
        match = _PIN.fullmatch(requirement) if isinstance(requirement, str) else None
        if not match:
            raise ValueError("Wheel requirements must be exact name==version pins without paths, markers or wildcards.")
        distribution = re.sub(r"[-_.]+", "-", match["name"]).lower()
        if distribution in distributions:
            raise ValueError("Wheel requirements must not repeat a distribution.")
        distributions.add(distribution)
    wheels = {"requirements": requirements,
              "index_url": validate_url(wheel_spec.get("index_url", "https://pypi.org/simple")),
              "max_total_bytes": _limit(wheel_spec.get("max_total_bytes", 1024**3), "Wheelhouse byte limit", HARD_WHEEL_BYTES)}
    prebuilt = wheel_spec.get("prebuilt", [])
    if not isinstance(prebuilt, list) or len(prebuilt) > 16:
        raise ValueError("Prebuilt wheel list must contain at most 16 reviewed wheels.")
    seen_prebuilt = set()
    wheels["prebuilt"] = []
    for item in prebuilt:
        if not isinstance(item, dict):
            raise ValueError("Invalid prebuilt wheel identity.")
        name = _filename(item.get("filename"))
        digest = item.get("sha256")
        if not name.endswith(".whl") or name.casefold() in seen_prebuilt or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Prebuilt wheels require unique filenames and exact SHA256 pins.")
        seen_prebuilt.add(name.casefold())
        wheels["prebuilt"].append({"filename": name, "sha256": digest, "max_bytes": _limit(item.get("max_bytes"), "Prebuilt wheel size", 50 * CHUNK)})
    return {"schema_version": 1, "target": dict(TARGET), "packages": normalized, "policy": limits, "wheels": wheels}


def load_lock(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > MAX_LOCK_BYTES:
        raise ValueError("Dependency lock exceeds 1 MiB.")
    data = path.read_bytes()
    lock = validate_lock(json.loads(data))
    if lock["wheels"]["prebuilt"]:
        lock["wheels"]["prebuilt_root"] = str(path.absolute().parent / "wheels")
    return lock, hashlib.sha256(data).hexdigest()


def _plain(path: Path) -> None:
    for candidate in (path.absolute(), *path.absolute().parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Staging paths must not be links or junctions: {candidate}")


def _mkdir(path: Path) -> None:
    _plain(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"Staging directory is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def file_identity(path: Path, max_bytes: int) -> dict:
    _plain(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"File is absent or exceeds its byte limit: {path.name}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"File exceeds its byte limit: {path.name}")
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def cached_download(package: dict, cache: Path) -> Path:
    directory = cache / package["sha256"]
    _mkdir(directory)
    destination = directory / package["filename"]
    _plain(destination)
    if destination.exists():
        identity = file_identity(destination, package["max_bytes"])
        if identity["sha256"] != package["sha256"]:
            raise ValueError(f"Cached {package['name']} failed its hash check; existing evidence was preserved.")
        return destination
    partial = directory / (package["filename"] + ".part-" + uuid.uuid4().hex)
    opener = build_opener(_HTTPSRedirects())
    request = Request(package["url"], headers={"User-Agent": "NestraDependencyStager/1", "Accept-Encoding": "identity"})
    digest = hashlib.sha256()
    size = 0
    try:
        with opener.open(request, timeout=60) as response, partial.open("xb") as output:
            validate_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > package["max_bytes"]):
                raise ValueError(f"{package['name']}: server content length exceeds the pinned byte limit.")
            while chunk := response.read(CHUNK):
                size += len(chunk)
                if size > package["max_bytes"]:
                    raise ValueError(f"{package['name']}: download exceeds the pinned byte limit.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not size or digest.hexdigest() != package["sha256"]:
            raise ValueError(f"{package['name']}: downloaded SHA256 does not match the lock.")
        # No overwrite of a pre-existing cache entry, even under concurrent builds.
        try:
            os.link(partial, destination)
        except FileExistsError:
            if file_identity(destination, package["max_bytes"])["sha256"] != package["sha256"]:
                raise ValueError("Concurrent cache entry failed its hash check.")
        except OSError:
            # Some removable/cache filesystems lack hard links. Exclusive copy
            # preserves the same no-overwrite contract.
            with partial.open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, CHUNK)
        return destination
    finally:
        if partial.exists():
            partial.unlink()


def _zip_member(info: zipfile.ZipInfo) -> tuple[str, ...]:
    # ZipInfo normalizes backslashes and truncates NULs on Windows. Validate
    # the original central-directory name before using its normalized alias.
    name = info.orig_filename
    if not name or "\\" in name or name.startswith("/") or "\x00" in name:
        raise ValueError("ZIP contains an invalid or absolute path.")
    path = name[:-1] if name.endswith("/") else name
    parts = tuple(path.split("/"))
    for part in parts:
        _filename(part)
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or info.external_attr & 0x400:
        raise ValueError("ZIP links, reparse points and special files are not permitted.")
    if info.flag_bits & 1:
        raise ValueError("Encrypted dependency ZIP entries are not supported.")
    return parts


def extract_zip(archive_path: Path, destination: Path, *, max_bytes: int, max_entries: int) -> list[dict]:
    if destination.exists():
        raise ValueError("ZIP destination already exists; refusing to overwrite it.")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries or sum(item.file_size for item in infos) > max_bytes:
            raise ValueError("ZIP expansion exceeds its declared entry or byte limit.")
        seen = set()
        entries = []
        for info in infos:
            parts = _zip_member(info)
            normalized = "/".join(parts).casefold()
            if normalized in seen:
                raise ValueError("ZIP contains duplicate paths on Windows.")
            seen.add(normalized)
            entries.append((info, parts))
        _mkdir(destination)
        total = 0
        receipt = []
        for info, parts in entries:
            target = destination.joinpath(*parts)
            cursor = destination
            for part in parts[:-1]:
                cursor /= part
                _mkdir(cursor)
            if info.is_dir():
                _mkdir(target)
                continue
            digest = hashlib.sha256()
            written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                while chunk := source.read(CHUNK):
                    written += len(chunk)
                    total += len(chunk)
                    if written > info.file_size or total > max_bytes:
                        raise ValueError("ZIP expanded bytes exceed the declared limit.")
                    digest.update(chunk)
                    output.write(chunk)
            if written != info.file_size:
                raise ValueError("ZIP entry length did not match its declaration.")
            receipt.append({"path": "/".join(parts), "size": written, "sha256": digest.hexdigest()})
        return receipt


class _BoundedTarInfo(tarfile.TarInfo):
    # tarfile processes extended headers before returning a member to callers.
    # Cap those metadata allocations before they can bypass member-size checks.
    def _proc_pax(self, archive):
        if self.size > CHUNK:
            raise ValueError("TAR extended metadata exceeds 1 MiB.")
        return super()._proc_pax(archive)

    def _proc_gnulong(self, archive):
        if self.size > 16_384:
            raise ValueError("TAR long-name metadata exceeds its limit.")
        return super()._proc_gnulong(archive)

    def _proc_sparse(self, archive):
        raise ValueError("Sparse dependency TAR members are not permitted.")


class _ReadBudget:
    def __init__(self, stream, limit: int):
        self.stream = stream
        self.remaining = limit

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(min(CHUNK, self.remaining + 1, CHUNK if size < 0 else size))
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError("TAR decompressed data and metadata exceed the expansion budget.")
        return data


def extract_tar(archive_path: Path, destination: Path, *, max_bytes: int, max_entries: int) -> list[dict]:
    if destination.exists():
        raise ValueError("TAR destination already exists; refusing to overwrite it.")
    _mkdir(destination)
    receipt = []
    seen = set()
    total = 0
    count = 0
    # Count decompressed headers as well as files. Repeated PAX/global metadata
    # cannot evade the file-byte/count checks by expanding discarded headers.
    stream_budget = max_bytes + min(max_entries * 1024, 32 * CHUNK) + 20_480
    with archive_path.open("rb") as compressed, gzip.GzipFile(fileobj=compressed) as uncompressed, tarfile.open(
            fileobj=_ReadBudget(uncompressed, stream_budget), mode="r|", tarinfo=_BoundedTarInfo) as archive:
        for member in archive:
            count += 1
            if count > max_entries:
                raise ValueError("TAR expansion exceeds its declared entry limit.")
            if not (member.isfile() or member.isdir()) or member.sparse is not None:
                raise ValueError("TAR links, sparse files, devices and special entries are not permitted.")
            name = member.name[:-1] if member.name.endswith("/") else member.name
            if not name or "\\" in name or name.startswith("/") or "\x00" in name:
                raise ValueError("TAR contains an invalid or absolute path.")
            parts = tuple(name.split("/"))
            for part in parts:
                _filename(part)
            normalized = "/".join(parts).casefold()
            if normalized in seen:
                raise ValueError("TAR contains duplicate paths on Windows.")
            seen.add(normalized)
            if member.size < 0 or total + member.size > max_bytes:
                raise ValueError("TAR expansion exceeds its declared byte limit.")
            target = destination.joinpath(*parts)
            cursor = destination
            for part in parts[:-1]:
                cursor /= part
                _mkdir(cursor)
            if member.isdir():
                if member.size:
                    raise ValueError("TAR directories must not contain file data.")
                _mkdir(target)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("TAR regular file could not be read.")
            written = 0
            digest = hashlib.sha256()
            with source, target.open("xb") as output:
                while chunk := source.read(CHUNK):
                    written += len(chunk)
                    total += len(chunk)
                    if written > member.size or total > max_bytes:
                        raise ValueError("TAR expanded bytes exceed the declared limit.")
                    digest.update(chunk)
                    output.write(chunk)
            if written != member.size:
                raise ValueError("TAR file length did not match its declaration.")
            receipt.append({"path": "/".join(parts), "size": written, "sha256": digest.hexdigest()})
    return receipt


def _tree_bytes(root: Path) -> int:
    total = 0
    count = 0
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        for name in subdirectories + files:
            path = Path(directory) / name
            _plain(path)
            count += 1
            if count > 20_000:
                raise ValueError("pip download working files exceed the entry limit.")
        for name in files:
            total += (Path(directory) / name).stat().st_size
    return total


def wheel_manifest(wheelhouse: Path, max_bytes: int) -> tuple[list[dict], str]:
    files = sorted(wheelhouse.iterdir())
    if not files or len(files) > 256:
        raise ValueError("The wheelhouse must contain 1 to 256 binary wheel files.")
    receipt = []
    lines = []
    names = set()
    total = 0
    for path in files:
        _filename(path.name)
        if path.suffix != ".whl":
            raise ValueError("The wheelhouse contains a non-wheel file.")
        identity = file_identity(path, max_bytes)
        total += identity["size"]
        if total > max_bytes:
            raise ValueError("The wheelhouse exceeds its byte limit.")
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if info.filename.endswith(".dist-info/METADATA")]
            if len(infos) != 1 or infos[0].file_size > 1024 * 1024:
                raise ValueError("A wheel has absent, duplicate or oversized metadata.")
            metadata = BytesParser().parsebytes(archive.read(infos[0]), headersonly=True)
        name = metadata.get("Name", "")
        version = metadata.get("Version", "")
        pin = _PIN.fullmatch(name + "==" + version)
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if not pin or canonical in names:
            raise ValueError("Wheel metadata identity is invalid or duplicated.")
        names.add(canonical)
        receipt.append({"file": path.name, "name": name, "version": version, **identity})
        lines.append(f"{name}=={version} --hash=sha256:{identity['sha256']}")
    return receipt, "\n".join(lines) + "\n"


def download_wheels(spec: dict, output: Path) -> dict:
    wheelhouse = output / "wheelhouse"
    work = output / "pip-work"
    _mkdir(wheelhouse)
    _mkdir(work)
    prebuilt = spec.get("prebuilt", [])
    for item in prebuilt:
        filename = _filename(item["filename"])
        root = Path(spec["prebuilt_root"])
        _plain(root)
        source = root / filename
        _plain(source)
        if source.resolve().parent != root.resolve():
            raise ValueError("Reviewed prebuilt wheel is outside its locked wheel directory.")
        identity = file_identity(source, item["max_bytes"])
        if identity["sha256"] != item["sha256"]:
            raise ValueError("Reviewed prebuilt wheel hash differs from the dependency lock.")
        destination = wheelhouse / filename
        with source.open("rb") as original, destination.open("xb") as target:
            shutil.copyfileobj(original, target, CHUNK)
        if file_identity(destination, item["max_bytes"]) != identity:
            raise ValueError("Reviewed prebuilt wheel changed while it was staged.")
    if _tree_bytes(wheelhouse) > spec["max_total_bytes"]:
        raise ValueError("Reviewed prebuilt wheels exceed the wheelhouse byte limit.")
    requirements = output / "requirements-download.txt"
    requirements.write_text("\n".join(spec["requirements"]) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input", "download",
               "--no-cache-dir", "--platform", TARGET["platform"], "--python-version", TARGET["python_version"],
               "--implementation", TARGET["implementation"], "--abi", TARGET["abi"], "--only-binary=:all:",
               "--index-url", spec["index_url"], "--dest", str(wheelhouse), "--requirement", str(requirements)]
    if prebuilt:
        command.extend(["--find-links", str(wheelhouse)])
    log = output / "pip-download.log"
    env = {**os.environ, "TMPDIR": str(work), "TEMP": str(work), "TMP": str(work)}
    started = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=str(output), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                if time.monotonic() - started > 900:
                    raise ValueError("pip download exceeded its 15-minute timeout.")
                if _tree_bytes(work) + _tree_bytes(wheelhouse) > spec["max_total_bytes"] or log.stat().st_size > 2 * CHUNK:
                    raise ValueError("pip download exceeded its bounded working byte limit.")
                time.sleep(0.25)
            if process.returncode:
                raise ValueError(f"pip binary download failed with exit {process.returncode}; inspect {log}.")
            if _tree_bytes(work) + _tree_bytes(wheelhouse) > spec["max_total_bytes"] or log.stat().st_size > 2 * CHUNK:
                raise ValueError("pip download exceeded its bounded working byte limit.")
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            raise
    receipt, offline = wheel_manifest(wheelhouse, spec["max_total_bytes"])
    (output / "requirements-offline.txt").write_text(offline, encoding="utf-8")
    # pip's temporary directory is intentionally retained for build diagnostics;
    # it is not part of the runtime payload or wheelhouse identity.
    return {"target": dict(TARGET), "download_command": command, "requested_requirements": list(spec["requirements"]), "files": receipt,
            "requirements": "requirements-offline.txt",
            "offline_install_arguments": ["-m", "pip", "install", "--no-index", "--find-links", "wheelhouse", "--require-hashes", "--only-binary=:all:", "-r", "requirements-offline.txt"]}


def stage(lock_path: Path, cache: Path, output: Path, *, skip_wheels: bool = False) -> dict:
    lock, lock_hash = load_lock(lock_path)
    _plain(cache)
    _plain(output)
    if output.exists():
        raise ValueError("Output must be a new directory; existing build evidence is preserved.")
    if output.resolve() == cache.resolve() or cache.resolve() in output.resolve().parents or output.resolve() in cache.resolve().parents:
        raise ValueError("Cache and output directories must be separate, non-nested paths.")
    _mkdir(cache)
    _mkdir(output)
    receipt = {"schema_version": 1, "lock_sha256": lock_hash, "target": dict(TARGET),
               "created_at": datetime.now(UTC).isoformat(), "packages": [], "wheels": {"status": "skipped"}}
    try:
        for package in lock["packages"]:
            source = cached_download(package, cache)
            category = "archives" if package["kind"] in {"zip", "tar.gz"} else "installers"
            destination = output / category / package["name"] / package["filename"]
            _mkdir(destination.parent)
            with source.open("rb") as original, destination.open("xb") as target:
                shutil.copyfileobj(original, target, CHUNK)
            identity = file_identity(destination, package["max_bytes"])
            if identity["sha256"] != package["sha256"]:
                raise ValueError("Staged archive hash does not match the lock.")
            item = {"name": package["name"], "kind": package["kind"], "url": package["url"],
                    "path": destination.relative_to(output).as_posix(), **identity}
            if package["kind"] in {"zip", "tar.gz"}:
                unpack = output / "runtime-deps" / package["name"]
                _mkdir(unpack.parent)
                item["extracted_root"] = unpack.relative_to(output).as_posix()
                extract = extract_zip if package["kind"] == "zip" else extract_tar
                item["files"] = extract(destination, unpack, max_bytes=package["max_unpacked_bytes"], max_entries=lock["policy"]["max_zip_entries"])
            receipt["packages"].append(item)
        if not skip_wheels and lock["wheels"]["requirements"]:
            receipt["wheels"] = {"status": "downloaded", **download_wheels(lock["wheels"], output)}
        receipt["status"] = "staged"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = str(exc)[:2000]
        (output / "dependency-stage-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    (output / "dependency-stage-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-wheels", action="store_true")
    args = parser.parse_args()
    try:
        receipt = stage(args.lock.absolute(), args.cache.absolute(), args.output.absolute(), skip_wheels=args.skip_wheels)
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f"Dependency staging failed: {exc}\n")
    print(json.dumps({"status": receipt["status"], "packages": len(receipt["packages"]), "receipt": str(args.output / "dependency-stage-receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
