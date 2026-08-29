from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Iterator

from localpilot.process import hidden_process_creation_flags
from localpilot.runtime_evidence import RuntimeEvidence


_BLOCKED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "localpilot-data",
    "node_modules",
}
_BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
}
_BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_MAX_READ_LINES = 300
_MAX_READ_CHARS = 50_000
_MAX_SEARCH_FILE_BYTES = 1_000_000


class RepositoryReader:
    """Bounded, read-only inspection of LocalPilot's trusted repository checkout."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        data_dir: str | Path | None = None,
        main_branch: str = "main",
    ) -> None:
        self.root = Path(project_root).resolve()
        self.runtime_evidence = RuntimeEvidence(
            self.root,
            Path(data_dir).resolve() / "audit.jsonl"
            if data_dir is not None
            else self.root / "localpilot-data" / "audit.jsonl",
            main_branch=main_branch,
        )

    @staticmethod
    def _is_sensitive(relative: Path) -> bool:
        parts = {part.lower() for part in relative.parts}
        name = relative.name.lower()
        if parts & _BLOCKED_PARTS:
            return True
        if name in _BLOCKED_NAMES or name.startswith(".env."):
            return True
        return relative.suffix.lower() in _BLOCKED_SUFFIXES

    def _resolve(self, path: str = ".", *, must_exist: bool = True) -> Path:
        raw = Path(path or ".")
        if raw.is_absolute():
            raise ValueError("Repository paths must be relative to the trusted project root.")
        cursor = self.root
        for part in raw.parts:
            if part in {"", "."}:
                continue
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                raise ValueError("Symlink repository paths are not allowed.")
        candidate = (self.root / raw).resolve(strict=False)
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Repository path escapes the trusted project root.") from exc
        if self._is_sensitive(relative):
            raise ValueError("Repository path is excluded from operator inspection.")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Repository path does not exist: {relative.as_posix()}")
        return candidate

    def _relative(self, path: Path) -> Path:
        return path.resolve(strict=False).relative_to(self.root)

    def _iter_files(self, base: Path) -> Iterator[Path]:
        if base.is_file():
            yield base
            return
        for directory, dirnames, filenames in os.walk(base, followlinks=False):
            current = Path(directory)
            retained_dirs: list[str] = []
            for dirname in dirnames:
                candidate = current / dirname
                relative = candidate.relative_to(self.root)
                if self._is_sensitive(relative) or candidate.is_symlink():
                    continue
                retained_dirs.append(dirname)
            dirnames[:] = retained_dirs
            for filename in filenames:
                candidate = current / filename
                relative = candidate.relative_to(self.root)
                if self._is_sensitive(relative) or candidate.is_symlink():
                    continue
                yield candidate

    def list_repository_tree(
        self,
        path: str = ".",
        depth: int = 2,
        max_entries: int = 200,
    ) -> str:
        """List a bounded repository tree without following symlinks or reading file bodies."""
        base = self._resolve(path)
        if not base.is_dir():
            raise ValueError("Tree inspection requires a directory path.")
        depth = max(0, min(int(depth), 6))
        max_entries = max(1, min(int(max_entries), 500))
        root_relative = self._relative(base)
        rows: list[str] = []

        def visit(directory: Path, remaining: int) -> None:
            if len(rows) >= max_entries:
                return
            try:
                entries = sorted(
                    directory.iterdir(),
                    key=lambda item: (
                        item.is_symlink() or not item.is_dir(),
                        item.name.lower(),
                    ),
                )
            except OSError as exc:
                rows.append(f"[unreadable] {exc}")
                return
            for item in entries:
                if len(rows) >= max_entries:
                    return
                relative = item.relative_to(self.root)
                if self._is_sensitive(relative):
                    continue
                indent_level = len(relative.parts) - len(root_relative.parts)
                indent = "  " * max(0, indent_level - 1)
                if item.is_symlink():
                    rows.append(f"{indent}{item.name} -> [symlink not followed]")
                    continue
                if item.is_dir():
                    rows.append(f"{indent}{item.name}/")
                    if remaining > 0:
                        visit(item, remaining - 1)
                else:
                    rows.append(f"{indent}{item.name}")

        visit(base, depth)
        if len(rows) >= max_entries:
            rows.append(f"... output limited to {max_entries} entries")
        heading = "." if root_relative == Path(".") else root_relative.as_posix()
        return f"Repository tree: {heading}\n" + "\n".join(rows)

    def read_repository_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        """Read a bounded UTF-8 text slice from a verified file inside the repository."""
        target = self._resolve(path)
        if not target.is_file():
            raise ValueError("Repository file inspection requires a regular file.")
        start_line = max(1, int(start_line))
        end_line = max(start_line, int(end_line))
        end_line = min(end_line, start_line + _MAX_READ_LINES - 1)
        output: list[str] = []
        chars = 0
        try:
            with target.open("r", encoding="utf-8", errors="strict") as handle:
                for number, line in enumerate(handle, start=1):
                    if number < start_line:
                        continue
                    if number > end_line:
                        break
                    rendered = f"{number}: {line.rstrip()}"
                    if chars + len(rendered) + 1 > _MAX_READ_CHARS:
                        output.append("... output truncated at character limit")
                        break
                    output.append(rendered)
                    chars += len(rendered) + 1
        except UnicodeDecodeError as exc:
            raise ValueError("Repository file is not UTF-8 text.") from exc
        relative = target.relative_to(self.root).as_posix()
        return f"Repository file: {relative} lines {start_line}-{end_line}\n" + "\n".join(output)

    def search_repository(
        self,
        query: str,
        path: str = ".",
        max_results: int = 40,
    ) -> str:
        """Search repository text for a literal case-insensitive string and return bounded line matches."""
        needle = str(query).strip()
        if not needle:
            raise ValueError("Repository search query must not be empty.")
        if len(needle) > 200:
            raise ValueError("Repository search query is too long.")
        base = self._resolve(path)
        max_results = max(1, min(int(max_results), 100))
        matches: list[str] = []
        lower_needle = needle.lower()
        for candidate in self._iter_files(base):
            if len(matches) >= max_results:
                break
            relative = candidate.relative_to(self.root)
            try:
                if candidate.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue
                with candidate.open("r", encoding="utf-8", errors="strict") as handle:
                    for number, line in enumerate(handle, start=1):
                        if lower_needle not in line.lower():
                            continue
                        snippet = " ".join(line.strip().split())[:240]
                        matches.append(f"{relative.as_posix()}:{number}: {snippet}")
                        if len(matches) >= max_results:
                            break
            except (OSError, UnicodeDecodeError):
                continue
        suffix = "\n... result limit reached" if len(matches) >= max_results else ""
        return (
            f"Repository search: {needle!r}\n"
            + ("\n".join(matches) if matches else "No matches found.")
            + suffix
        )

    def inspect_project_dependencies(self) -> str:
        """Read declared Python/build dependencies from repository packaging files."""
        pyproject = self._resolve("pyproject.toml")
        if not pyproject.is_file():
            return "No pyproject.toml exists in the trusted repository."
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        project = data.get("project", {}) if isinstance(data, dict) else {}
        build = data.get("build-system", {}) if isinstance(data, dict) else {}
        result = {
            "requires_python": project.get("requires-python"),
            "dependencies": project.get("dependencies", []),
            "optional_dependencies": project.get("optional-dependencies", {}),
            "scripts": project.get("scripts", {}),
            "build_requires": build.get("requires", []),
            "build_backend": build.get("build-backend"),
        }
        return json.dumps(result, indent=2, sort_keys=True)

    def get_repository_status(self) -> str:
        """Return current branch, HEAD, and working-tree status using read-only Git commands."""
        executable = shutil.which("git")
        if not executable:
            return "Git is not available."
        commands = [
            [executable, "-C", str(self.root), "branch", "--show-current"],
            [executable, "-C", str(self.root), "rev-parse", "HEAD"],
            [executable, "-C", str(self.root), "status", "--short"],
        ]
        labels = ["branch", "head", "changes"]
        result: dict[str, str] = {}
        for label, argv in zip(labels, commands, strict=True):
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
                creationflags=hidden_process_creation_flags(),
            )
            if completed.returncode != 0:
                result[label] = f"git error: {completed.stderr.strip()}"
            else:
                result[label] = completed.stdout.strip() or "(clean)"
        return json.dumps(result, indent=2)

    def get_runtime_lifecycle(self, limit: int = 8) -> str:
        """Read recent runtime transitions, current process evidence, and checkout state."""
        limit = max(1, min(int(limit), 20))
        return json.dumps(self.runtime_evidence.snapshot(limit=limit), indent=2)
