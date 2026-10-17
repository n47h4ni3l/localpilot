"""Stream the reviewed main source and staged payload into a Nestra ZIP64 bundle.

This builder does not enumerate the owner's application data, load a model,
install anything or execute downloaded runtime binaries. Every payload member
is hashed while it is written. The accepted model descriptors are checked in
that same pass, avoiding a redundant full GGUF verification read.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile

CHUNK = 1024 * 1024
MAX_METADATA = 32 * CHUNK
MAX_FILES = 200_000
MAX_TOTAL_BYTES = 512 * 1024**3
ORIGIN = "https://github.com/n47h4ni3l/localpilot.git"
MAIN_REF = "refs/heads/main"
_RESERVED = {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)}, *{f"LPT{i}" for i in range(1, 10)}}
_ROOT_FILES = {
    "Install Nestra.cmd": "Install Nestra.cmd",
    "scripts/install-nestra-bundle.ps1": "scripts/install-nestra-bundle.ps1",
    "scripts/start-nestra-setup.ps1": "scripts/start-nestra-setup.ps1",
    "README.md": "docs/NESTRA_BUNDLE.md",
    "THIRD_PARTY_NOTICES.md": "packaging/nestra-third-party-notices.md",
}


def plain_path(path: Path) -> None:
    for candidate in (path.absolute(), *path.absolute().parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked or reparse-point paths are not allowed: {candidate}")


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or value.startswith("/"):
        raise ValueError("Bundle paths must be bounded relative filenames.")
    for part in value.split("/"):
        if (not part or part in {".", ".."} or part.endswith((".", " "))
                or any(char in part for char in '\\:<>"|?*') or any(ord(char) < 32 for char in part)
                or part.split(".", 1)[0].upper() in _RESERVED):
            raise ValueError(f"Unsafe Windows bundle path: {value}")
    value.encode("utf-8")
    return value


def existing_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    plain_path(root)
    if not root.is_dir():
        raise ValueError(f"Expected an existing directory: {root}")
    return root.resolve()


def _json(path: Path) -> dict:
    plain_path(path)
    if not path.is_file() or path.stat().st_size > MAX_METADATA:
        raise ValueError(f"Missing or oversized build metadata: {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate build metadata key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Build metadata must be an object.")
    return value


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or len(result.stdout) > MAX_METADATA or len(result.stderr) > CHUNK:
        raise ValueError(f"Git source verification failed: {' '.join(arguments[:2])}")
    return result.stdout.decode("utf-8").strip()


def verify_source(root: Path) -> str:
    if git(root, "rev-parse", "--show-toplevel").replace("\\", "/").casefold() != root.as_posix().casefold():
        raise ValueError("Source root must be the canonical Git repository root.")
    if git(root, "symbolic-ref", "HEAD") != MAIN_REF:
        raise ValueError("Build from the canonical main branch.")
    sha = git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or git(root, "rev-parse", MAIN_REF) != sha:
        raise ValueError("Source main and HEAD identities differ.")
    if git(root, "remote", "get-url", "origin") != ORIGIN:
        raise ValueError("Source origin must be the canonical LocalPilot repository URL.")
    if git(root, "rev-parse", "refs/remotes/origin/main") != sha:
        raise ValueError("Local main must exactly match the recorded origin/main release revision.")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Source main must be clean, including untracked files.")
    return sha


def walk_files(root: Path) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    total = 0
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        for name in subdirectories:
            path = Path(directory) / name
            plain_path(path)
            relative_path(path.relative_to(root).as_posix())
        for name in files:
            path = Path(directory) / name
            plain_path(path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Bundle input is not a regular file: {path}")
            relative = relative_path(path.relative_to(root).as_posix())
            normalized = relative.casefold()
            if normalized in seen:
                raise ValueError("Bundle contains duplicate Windows path identities.")
            seen.add(normalized)
            result.append((relative, path))
            total += info.st_size
            if len(result) > MAX_FILES or total > MAX_TOTAL_BYTES:
                raise ValueError("Bundle input exceeds its file or byte limit.")
    return sorted(result)


def model_spec(source_root: Path, model_root: Path) -> dict:
    # Import only the committed, stdlib-only metadata helper. _read_bundle
    # validates accepted identity/descriptor sizes without reading model blobs.
    path = source_root / "localpilot" / "nestra_bundle_model.py"
    plain_path(path)
    module_spec = importlib.util.spec_from_file_location("nestra_release_model_metadata", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    _, _, result = module._read_bundle(model_root)
    return result


def _expected(item: dict) -> dict:
    if (not isinstance(item, dict) or type(item.get("size")) is not int or not 0 <= item["size"] <= MAX_TOTAL_BYTES
            or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
        raise ValueError("Invalid staged file identity.")
    return {"size": item["size"], "sha256": item["sha256"]}


def dependency_files(root: Path) -> list[tuple[str, Path, dict | None]]:
    receipt = _json(root / "dependency-stage-receipt.json")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "staged" or receipt.get("wheels", {}).get("status") != "downloaded":
        raise ValueError("Dependencies require a successful stage receipt and complete offline wheelhouse.")
    expected = {}
    for package in receipt.get("packages", []):
        path = relative_path(package["path"])
        if path in expected:
            raise ValueError("Duplicate dependency receipt paths.")
        expected[path] = _expected(package)
        if package.get("kind") in {"zip", "tar.gz"}:
            prefix = relative_path(package["extracted_root"])
            for item in package["files"]:
                path = relative_path(prefix + "/" + relative_path(item["path"]))
                if path in expected:
                    raise ValueError("Duplicate extracted dependency paths.")
                expected[path] = _expected(item)
    for item in receipt["wheels"].get("files", []):
        path = relative_path("wheelhouse/" + relative_path(item["file"]))
        if path in expected:
            raise ValueError("Duplicate wheelhouse paths.")
        expected[path] = _expected(item)
    if not any(path.startswith("wheelhouse/") for path in expected):
        raise ValueError("Dependency receipt has no pinned wheel files.")
    metadata = {"dependency-stage-receipt.json", "requirements-download.txt", "requirements-offline.txt"}
    found = set()
    result = []
    for relative, path in walk_files(root):
        if relative == "pip-download.log" or relative.startswith("pip-work/"):
            continue
        if relative not in expected and re.fullmatch(r"runtime-deps/python/python/Lib/(?:[^/]+/)*__pycache__/[^/]+\.pyc", relative):
            # Running the staged interpreter for offline checks can create
            # local caches. Preserve them locally, but do not distribute them.
            # Archive-listed bytecode remains pinned and is never skipped.
            continue
        if relative not in expected and relative not in metadata:
            raise ValueError(f"Unlisted file in staged dependencies: {relative}")
        found.add(relative)
        result.append(("payload/dependencies/" + relative, path, expected.get(relative)))
    if not set(expected).issubset(found) or "requirements-offline.txt" not in found:
        raise ValueError("Staged dependencies are missing files named in their receipt.")
    return result


def distribution_dependency_receipt(root: Path) -> bytes:
    """Keep public dependency provenance without machine-local build paths.

    The original receipt remains untouched in the local staging directory.
    Whitelisting also prevents a future diagnostic field from silently adding
    owner directories or environment data to a public release.
    """
    original = _json(root / "dependency-stage-receipt.json")
    def only(item: dict, fields: tuple[str, ...]) -> dict:
        return {field: item[field] for field in fields if field in item}
    target_fields = ("platform", "python_version", "implementation", "abi")
    public = only(original, ("schema_version", "lock_sha256", "created_at", "status"))
    if "target" in original:
        public["target"] = only(original["target"], target_fields)
    public["packages"] = []
    for package in original.get("packages", []):
        item = only(package, ("name", "kind", "url", "path", "size", "sha256", "extracted_root"))
        if "files" in package:
            item["files"] = [only(file, ("path", "size", "sha256")) for file in package["files"]]
        public["packages"].append(item)
    original_wheels = original.get("wheels", {})
    wheels = only(original_wheels, ("status", "requested_requirements"))
    if "target" in original_wheels:
        wheels["target"] = only(original_wheels["target"], target_fields)
    wheels["files"] = [only(file, ("file", "name", "version", "size", "sha256")) for file in original_wheels.get("files", [])]
    wheels["requirements"] = "requirements-offline.txt"
    wheels["offline_install_arguments"] = ["-m", "pip", "install", "--no-index", "--find-links", "wheelhouse", "--require-hashes", "--only-binary=:all:", "-r", "requirements-offline.txt"]
    command = original_wheels.get("download_command")
    if isinstance(command, list):
        # Reconstruct the portable command from the known staging contract;
        # no unknown command argument or executable directory is distributed.
        index_url = "https://pypi.org/simple"
        if "--index-url" in command:
            position = command.index("--index-url") + 1
            if position < len(command):
                index_url = command[position]
        from urllib.parse import urlsplit
        parts = urlsplit(index_url) if isinstance(index_url, str) else None
        if parts is None or parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError("Staged wheel index must be a credential-free HTTPS URL.")
        wheels["download_command"] = ["python", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input", "download",
                                       "--no-cache-dir", "--platform", "win_amd64", "--python-version", "312", "--implementation", "cp", "--abi", "cp312",
                                       "--only-binary=:all:", "--index-url", index_url, "--dest", "wheelhouse", "--requirement", "requirements-download.txt"]
        if "--find-links" in command:
            wheels["download_command"].extend(["--find-links", "wheelhouse"])
    public["wheels"] = wheels
    return (json.dumps(public, indent=2) + "\n").encode("utf-8")


class _HashWriter:
    """Unseekable output lets ZIP use descriptors and hash final bytes once."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        result = self.stream.write(data)
        if result != len(data):
            raise OSError("Short write while building ZIP.")
        self.digest.update(data)
        self.size += result
        return result

    def tell(self):
        return self.size

    def seek(self, *_):
        raise io.UnsupportedOperation("Streaming ZIP output is not seekable")

    def flush(self):
        self.stream.flush()


def write_member(archive: zipfile.ZipFile, name: str, source, *, expected: dict | None = None, stored: bool = False) -> dict:
    relative_path(name)
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info._compresslevel = 1
    before = None
    try:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Bundle member is not a regular file.")
    except (AttributeError, io.UnsupportedOperation):
        pass
    digest = hashlib.sha256()
    size = 0
    with archive.open(info, "w", force_zip64=True) as output:
        while chunk := source.read(CHUNK):
            size += len(chunk)
            if size > MAX_TOTAL_BYTES or expected is not None and size > expected["size"]:
                raise ValueError(f"Bundle member grew beyond its expected size: {name}")
            digest.update(chunk)
            output.write(chunk)
    if before is not None:
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or size != before.st_size:
            raise ValueError(f"Bundle input changed while streaming: {name}")
    identity = {"path": name, "size": size, "sha256": digest.hexdigest()}
    if expected is not None and {"size": size, "sha256": identity["sha256"]} != expected:
        raise ValueError(f"Bundle member failed its pinned identity check: {name}")
    return identity


def _committed_file(root: Path, sha: str, relative: str) -> bytes:
    # Only the five small release wrappers/docs are read into memory, directly
    # from committed Git objects. Owner working/ignored files are never read.
    tree = git(root, "ls-tree", sha, "--", relative)
    if not re.fullmatch(r"100(?:644|755) blob [0-9a-f]{40}\t" + re.escape(relative), tree):
        raise ValueError(f"Committed release wrapper/doc is not a regular tracked file: {relative}")
    size = git(root, "cat-file", "-s", f"{sha}:{relative}")
    if not size.isdigit() or not 0 < int(size) <= CHUNK:
        raise ValueError(f"Committed release wrapper/doc is absent or exceeds 1 MiB: {relative}")
    result = subprocess.run(["git", "-C", str(root), "show", f"{sha}:{relative}"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or len(result.stdout) > CHUNK:
        raise ValueError(f"Committed release wrapper/doc is absent or exceeds 1 MiB: {relative}")
    return result.stdout


def release_root_files(root: Path, sha: str) -> dict[str, bytes]:
    files = {name: _committed_file(root, sha, path) for name, path in _ROOT_FILES.items()}
    licenses = git(root, "ls-tree", "-r", "--name-only", sha, "--", "packaging/licenses").splitlines()
    if not licenses or len(licenses) > 128:
        raise ValueError("Committed third-party license texts and provenance are missing or excessive.")
    for relative in licenses:
        relative_path(relative)
        prefix = "packaging/licenses/"
        if not relative.startswith(prefix):
            raise ValueError("License file is outside the committed license directory.")
        name = relative_path("licenses/" + relative[len(prefix):])
        files[name] = _committed_file(root, sha, relative)
    return files


def build_bundle(source_root: Path, dependencies: Path, model: Path, hardware: Path, output_directory: Path, version: str) -> dict:
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+[A-Za-z0-9.+-]{0,48}", version):
        raise ValueError("Version must be a plain dotted release version.")
    source_root, dependencies, model, hardware = [existing_root(path) for path in (source_root, dependencies, model, hardware)]
    output = Path(output_directory).absolute()
    plain_path(output)
    if output.exists():
        raise ValueError("Output must be a new directory; existing release evidence was preserved.")
    for source in (source_root, dependencies, model, hardware):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Bundle output must be outside and separate from every input directory.")
    sha = verify_source(source_root)
    root_files = release_root_files(source_root, sha)
    # Published wheel identities are reviewed independently of pip's resolver.
    # When a release inventory is committed, a later compatible transitive
    # version cannot silently enter a rebuild of that release.
    wheel_lock_path = "packaging/nestra-alpha2-wheels.json"
    committed_paths = git(source_root, "ls-tree", "--name-only", sha, "--", wheel_lock_path)
    if committed_paths:
        wheel_lock = json.loads(_committed_file(source_root, sha, wheel_lock_path))
        staged = _json(dependencies / "dependency-stage-receipt.json")
        if wheel_lock.get("files") != staged.get("wheels", {}).get("files") or wheel_lock.get("target") != staged.get("target"):
            raise ValueError("Staged wheels differ from the reviewed release wheel inventory.")
    spec = model_spec(source_root, model)
    expected_model = {f"blobs/sha256-{item['sha256']}": _expected(item) for item in spec["blobs"]}
    expected_model["manifest.json"] = {"size": (model / "manifest.json").stat().st_size, "sha256": spec["manifest_sha256"]}
    stored_model = {f"payload/model/blobs/sha256-{item['sha256']}" for item in spec["blobs"] if item["media_type"] == "application/vnd.ollama.image.model"}
    payload = dependency_files(dependencies)
    public_dependency_receipt = distribution_dependency_receipt(dependencies)
    model_names = set()
    for relative, path in walk_files(model):
        if relative not in expected_model and relative not in {"manifest.json", "bundle-model.json"}:
            raise ValueError(f"Unexpected file in staged model: {relative}")
        model_names.add(relative)
        payload.append(("payload/model/" + relative, path, expected_model.get(relative)))
    if not (set(expected_model) | {"manifest.json", "bundle-model.json"}).issubset(model_names):
        raise ValueError("Staged model is incomplete.")
    hardware_files = walk_files(hardware)
    required_hardware = {"win-x64/LocalPilot.SystemSense.HardwareProvider.exe", "win-x64/THIRD_PARTY_NOTICES.md"}
    if not required_hardware.issubset({name for name, _ in hardware_files}):
        raise ValueError("The staged Windows x64 hardware provider or its notices are missing.")
    payload.extend(("payload/hardware/" + relative, path, None) for relative, path in hardware_files)
    member_names = [*root_files, "source.bundle", *(name for name, _, _ in payload)]
    if len(member_names) > MAX_FILES or len({name.casefold() for name in member_names}) != len(member_names):
        raise ValueError("Bundle inventory exceeds its limit or contains duplicate paths.")
    if sum(path.stat().st_size for _, path, _ in payload) > MAX_TOTAL_BYTES:
        raise ValueError("Bundle payload exceeds the total byte limit.")
    output.mkdir(parents=True, exist_ok=False)
    zip_path = output / f"Nestra-{version}-Windows-x64.zip"
    receipt = {"schema_version": 1, "source_sha": sha, "source_ref": MAIN_REF, "origin": ORIGIN, "version": version,
               "created_at": datetime.now(UTC).isoformat(), "status": "building", "zip": zip_path.name}
    try:
        source_bundle = output / "source.bundle"
        git(source_root, "bundle", "create", str(source_bundle), MAIN_REF)
        advertised = git(source_root, "bundle", "list-heads", str(source_bundle))
        if advertised != f"{sha} {MAIN_REF}":
            raise ValueError("Source bundle does not advertise exactly the selected main revision.")
        manifest = {"schema_version": 1, "format_version": 1, "source_sha": sha, "source_ref": MAIN_REF,
                    "origin": ORIGIN, "version": version, "files": []}
        with zip_path.open("xb") as raw:
            hashed = _HashWriter(raw)
            with zipfile.ZipFile(hashed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for name, content in root_files.items():
                    manifest["files"].append(write_member(archive, name, io.BytesIO(content)))
                with source_bundle.open("rb") as stream:
                    manifest["files"].append(write_member(archive, "source.bundle", stream))
                for name, path, expected in sorted(payload):
                    if name == "payload/dependencies/dependency-stage-receipt.json":
                        manifest["files"].append(write_member(archive, name, io.BytesIO(public_dependency_receipt)))
                        continue
                    plain_path(path)
                    with path.open("rb") as stream:
                        manifest["files"].append(write_member(archive, name, stream, expected=expected, stored=name in stored_model))
                content = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
                if len(content) > MAX_METADATA:
                    raise ValueError("Bundle manifest exceeds the installer metadata limit.")
                write_member(archive, "bundle-manifest.json", io.BytesIO(content))
            raw.flush()
            os.fsync(raw.fileno())
            receipt.update(size=hashed.size, sha256=hashed.digest.hexdigest(), files=len(manifest["files"]))
        if verify_source(source_root) != sha:
            raise ValueError("Canonical source changed during the build.")
        (output / "bundle-manifest.json").write_bytes(content)
        (output / (zip_path.name + ".sha256")).write_text(f"{receipt['sha256']}  {zip_path.name}\n", encoding="ascii")
        receipt["status"] = "built"
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc)[:2000])
        (output / "bundle-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    (output / "bundle-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "dependencies", "model", "hardware", "output-directory"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        receipt = build_bundle(args.source_root, args.dependencies, args.model, args.hardware, args.output_directory, args.version)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Nestra bundle build failed: {exc}\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
