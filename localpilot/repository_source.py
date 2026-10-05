"""The shared file contract for repository authority and its live source index.

Tracked files are source inputs unless their path has a generated/private role.
Nonignored untracked source formats are included too, so new code invalidates
truth before it is committed. Without Git, the same source formats are used.
There is deliberately no file-size limit.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from localpilot.process import hidden_process_creation_flags


# Prune these roles before enumerating files, including in source distributions
# without Git. Training inputs (configs, schemas, fixtures, manifests, baselines,
# datasets and evaluation cases) are not generated output directories.
_GENERATED_PARTS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", "node_modules",
    "unsloth_compiled_cache", "localpilot-data",
})
_GENERATED_ROOTS = frozenset({
    "output", "outputs", "reports", "logs", "build", "dist", "htmlcov",
    "models", "adapters", "checkpoints",
})
_GENERATED_PREFIXES = (
    ("training", "outputs"), ("training", "reports"), ("localpilot", "_hardware"),
)
_PAYLOAD_SUFFIXES = frozenset({
    ".safetensors", ".gguf", ".ggml", ".ckpt", ".pt", ".pth", ".onnx",
    ".pyc", ".pyo", ".log", ".sqlite", ".sqlite3", ".db",
})
_PRIVATE_ROOT_FILES = frozenset({
    "localpilot.toml", ".localpilot-release.json", ".coverage", "thumbs.db", ".ds_store",
})
# Formats that can supply code, configuration, documentation, schemas, fixtures,
# or source assets. Unknown formats remain eligible when tracked by Git.
_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".json", ".jsonl",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".xml", ".csv", ".tsv", ".ipynb",
    ".ps1", ".psm1", ".psd1", ".sh", ".cmd", ".bat",
    ".js", ".cjs", ".mjs", ".jsx", ".ts", ".tsx", ".html", ".css",
    ".cs", ".csproj", ".sln", ".props", ".targets", ".manifest",
    ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".sql", ".proto",
    ".svg", ".png", ".jpg", ".jpeg", ".ico", ".pdf",
})
_SOURCE_NAMES = frozenset({
    ".gitignore", ".gitattributes", ".editorconfig", "dockerfile", "makefile",
    "license", "notice",
})


def _generated(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    return bool(parts) and (
        any(part in _GENERATED_PARTS or part.startswith(".venv.unusable-") for part in parts)
        or parts[0] in _GENERATED_ROOTS
        or any(parts[:len(prefix)] == prefix for prefix in _GENERATED_PREFIXES)
        or (parts[0] == "tools" and any(part in {"bin", "obj"} for part in parts[1:]))
        or (len(parts) == 1 and parts[0] in _PRIVATE_ROOT_FILES)
        or any(part == ".env" or part.startswith(".env.") for part in parts)
        or relative.suffix.lower() in _PAYLOAD_SUFFIXES
    )


def _path_key(relative: str) -> str:
    return os.path.normcase(relative.replace("/", os.sep))


def _git_paths(root: Path) -> tuple[set[str], set[str]] | None:
    """Use Git's tracked/ignore metadata only; never read artifact contents."""
    groups = []
    try:
        for arguments in (("--cached",), ("--others", "--exclude-standard")):
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", *arguments],
                check=True, capture_output=True, timeout=10,
                creationflags=hidden_process_creation_flags(),
            )
            groups.append({
                _path_key(os.fsdecode(name))
                for name in result.stdout.split(b"\0") if name
            })
    except (OSError, subprocess.SubprocessError):
        return None
    return groups[0], groups[1]


def repository_source_files(root: Path) -> tuple[Path, ...]:
    """Return deterministic, root-relative source membership without following links."""
    root = root.resolve()
    git_paths = _git_paths(root)
    selected: list[Path] = []
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = sorted(
            name for name in children
            if not _generated(Path(directory, name).relative_to(root))
            and not Path(directory, name).is_symlink()
            # resolve also catches Windows junctions on Python 3.11.
            and Path(directory, name).resolve() == Path(directory, name)
        )
        for name in names:
            path = Path(directory, name)
            relative = path.relative_to(root)
            if _generated(relative) or path.is_symlink() or not path.is_file():
                continue
            source_format = (
                relative.suffix.lower() in _SOURCE_SUFFIXES
                or name.lower() in _SOURCE_NAMES
            )
            if git_paths is not None:
                key = _path_key(relative.as_posix())
                tracked, untracked = git_paths
                if key not in tracked and not (key in untracked and source_format):
                    continue
            elif not source_format:
                continue
            selected.append(path)
    return tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))
