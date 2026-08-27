from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from localpilot.config import LibraryConfig


_SUPPORTED_SUFFIXES = {".md", ".pdf", ".rst", ".txt"}
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_QUERY_CHARS = 300
_PASSAGE_CHARS = 1800
_PASSAGE_OVERLAP = 200
_MAX_TOOL_OUTPUT_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    relative: str
    size: int
    mtime_ns: int


class LocalLibrary:
    """Bounded full-text access to owner-managed local reference files.

    Source files are never written. SQLite contains only a disposable derived
    index under LocalPilot's private data directory.
    """

    def __init__(self, config: LibraryConfig, index_path: str | Path) -> None:
        self.config = config
        self.root = Path(config.root).expanduser().resolve(strict=False)
        self.index_path = Path(index_path).resolve(strict=False)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS library_documents (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    source_digest TEXT NOT NULL DEFAULT '',
                    indexed_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS library_passages USING fts5(
                    path UNINDEXED,
                    page UNINDEXED,
                    passage UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(library_documents)")
            }
            if "source_digest" not in columns:
                connection.execute(
                    "ALTER TABLE library_documents ADD COLUMN source_digest TEXT NOT NULL DEFAULT ''"
                )

    def _availability_error(self) -> str | None:
        if not self.config.enabled:
            return "Local library is disabled in configuration."
        if not self.root.exists():
            return f"Local library root does not exist: {self.root}"
        if not self.root.is_dir():
            return f"Local library root is not a directory: {self.root}"
        if self.root.is_symlink():
            return "Local library root cannot be a symlink."
        return None

    @staticmethod
    def _hidden(relative: Path) -> bool:
        return any(part.startswith(".") for part in relative.parts)

    def _discover(self) -> tuple[list[_Source], bool]:
        sources: list[_Source] = []
        limit = int(self.config.max_documents)
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            current = Path(directory)
            retained: list[str] = []
            for name in sorted(dirnames, key=str.casefold):
                candidate = current / name
                relative = candidate.relative_to(self.root)
                if candidate.is_symlink() or self._hidden(relative):
                    continue
                retained.append(name)
            dirnames[:] = retained
            for name in sorted(filenames, key=str.casefold):
                candidate = current / name
                relative_path = candidate.relative_to(self.root)
                if (
                    candidate.is_symlink()
                    or self._hidden(relative_path)
                    or candidate.suffix.lower() not in _SUPPORTED_SUFFIXES
                ):
                    continue
                try:
                    stat = candidate.stat()
                except OSError:
                    continue
                sources.append(
                    _Source(
                        candidate,
                        relative_path.as_posix(),
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                    )
                )
                if len(sources) > limit:
                    return sources[:limit], True
        sources.sort(key=lambda item: item.relative.casefold())
        return sources, False

    def _resolve_source(self, path: str) -> tuple[Path, str]:
        raw = Path(str(path).strip())
        if not raw.parts or raw.is_absolute():
            raise ValueError("Library paths must be relative to the configured library root.")
        cursor = self.root
        for part in raw.parts:
            if part in {"", "."}:
                continue
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                raise ValueError("Symlink library paths are not allowed.")
        candidate = (self.root / raw).resolve(strict=False)
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Library path escapes the configured library root.") from exc
        if self._hidden(relative):
            raise ValueError("Hidden library paths are excluded.")
        if candidate.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError("Unsupported library file type.")
        if not candidate.is_file():
            raise FileNotFoundError(f"Library file does not exist: {relative.as_posix()}")
        return candidate, relative.as_posix()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(str(value).replace("\x00", " ").split())

    @staticmethod
    def _passages(text: str) -> Iterator[str]:
        normalized = LocalLibrary._normalize_text(text)
        cursor = 0
        while cursor < len(normalized):
            end = min(len(normalized), cursor + _PASSAGE_CHARS)
            if end < len(normalized):
                boundary = normalized.rfind(" ", cursor + (_PASSAGE_CHARS // 2), end)
                if boundary > cursor:
                    end = boundary
            passage = normalized[cursor:end].strip()
            if passage:
                yield passage
            if end >= len(normalized):
                break
            cursor = max(cursor + 1, end - _PASSAGE_OVERLAP)

    def _extract_pages(self, source: _Source) -> list[str]:
        maximum_bytes = int(self.config.max_file_size_mb) * 1024 * 1024
        if source.size > maximum_bytes:
            raise ValueError(
                f"file exceeds the configured {self.config.max_file_size_mb} MiB limit"
            )
        suffix = source.path.suffix.lower()
        if suffix != ".pdf":
            text = source.path.read_text(encoding="utf-8", errors="strict")
            return text.split("\f")[: int(self.config.max_pages_per_document)]

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires the pypdf package.") from exc

        reader = PdfReader(str(source.path), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("encrypted PDF requires a password")
        maximum_pages = int(self.config.max_pages_per_document)
        pages: list[str] = []
        for page in reader.pages[:maximum_pages]:
            extracted = page.extract_text() or ""
            pages.append(extracted[: int(self.config.max_chars_per_page)])
        return pages

    @staticmethod
    def _source_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def refresh_index(self) -> dict[str, int | bool | str]:
        unavailable = self._availability_error()
        if unavailable:
            return {"available": False, "message": unavailable}
        sources, truncated = self._discover()
        source_paths = {source.relative for source in sources}
        indexed = unchanged = failed = removed = processed = 0
        updates_remaining = int(self.config.max_refresh_files)
        with self._connect() as connection:
            prior = {
                str(row["path"]): row
                for row in connection.execute(
                    "SELECT path, size, mtime_ns FROM library_documents"
                )
            }
            if not truncated:
                for missing in sorted(set(prior) - source_paths):
                    connection.execute(
                        "DELETE FROM library_passages WHERE path = ?", (missing,)
                    )
                    connection.execute(
                        "DELETE FROM library_documents WHERE path = ?", (missing,)
                    )
                    removed += 1
            for source in sources:
                row = prior.get(source.relative)
                if (
                    row is not None
                    and int(row["size"]) == source.size
                    and int(row["mtime_ns"]) == source.mtime_ns
                ):
                    unchanged += 1
                    continue
                if updates_remaining <= 0:
                    continue
                updates_remaining -= 1
                processed += 1
                connection.execute(
                    "DELETE FROM library_passages WHERE path = ?", (source.relative,)
                )
                try:
                    pages = self._extract_pages(source)
                    source_digest = self._source_digest(source.path)
                    passage_rows: list[tuple[str, int, int, str]] = []
                    for page_number, page_text in enumerate(pages, start=1):
                        for passage_number, passage in enumerate(
                            self._passages(page_text), start=1
                        ):
                            passage_rows.append(
                                (source.relative, page_number, passage_number, passage)
                            )
                    if passage_rows:
                        connection.executemany(
                            "INSERT INTO library_passages(path, page, passage, text) VALUES (?, ?, ?, ?)",
                            passage_rows,
                        )
                    error = "" if passage_rows else "no extractable text"
                    connection.execute(
                        """
                        INSERT INTO library_documents(path, size, mtime_ns, kind, page_count, source_digest, indexed_at, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            size = excluded.size,
                            mtime_ns = excluded.mtime_ns,
                            kind = excluded.kind,
                            page_count = excluded.page_count,
                            source_digest = excluded.source_digest,
                            indexed_at = excluded.indexed_at,
                            error = excluded.error
                        """,
                        (
                            source.relative,
                            source.size,
                            source.mtime_ns,
                            source.path.suffix.lower().lstrip("."),
                            len(pages),
                            source_digest,
                            datetime.now(UTC).isoformat(),
                            error,
                        ),
                    )
                    indexed += 1
                    if error:
                        failed += 1
                except Exception as exc:
                    connection.execute(
                        """
                        INSERT INTO library_documents(path, size, mtime_ns, kind, page_count, source_digest, indexed_at, error)
                        VALUES (?, ?, ?, ?, 0, '', ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            size = excluded.size,
                            mtime_ns = excluded.mtime_ns,
                            kind = excluded.kind,
                            page_count = 0,
                            source_digest = '',
                            indexed_at = excluded.indexed_at,
                            error = excluded.error
                        """,
                        (
                            source.relative,
                            source.size,
                            source.mtime_ns,
                            source.path.suffix.lower().lstrip("."),
                            datetime.now(UTC).isoformat(),
                            f"{type(exc).__name__}: {exc}"[:500],
                        ),
                    )
                    failed += 1
        return {
            "available": True,
            "discovered": len(sources),
            "indexed": indexed,
            "unchanged": unchanged,
            "failed": failed,
            "removed": removed,
            "document_limit_reached": truncated,
            "updates_deferred": max(0, len(sources) - unchanged - processed),
        }

    @staticmethod
    def _fts_query(query: str) -> str:
        value = str(query).strip()
        if not value:
            raise ValueError("Library search query must not be empty.")
        if len(value) > _MAX_QUERY_CHARS:
            raise ValueError("Library search query is too long.")
        tokens = [token.casefold() for token in _WORD.findall(value) if len(token) > 1]
        if not tokens:
            raise ValueError("Library search query needs at least one searchable word.")
        return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))

    def get_library_summary(self) -> str:
        refresh = self.refresh_index()
        if not refresh.get("available"):
            return str(refresh["message"])
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS documents,
                       COALESCE(SUM(page_count), 0) AS pages,
                       COALESCE(SUM(CASE WHEN error <> '' THEN 1 ELSE 0 END), 0) AS errors
                FROM library_documents
                """
            ).fetchone()
            passages = connection.execute(
                "SELECT COUNT(*) AS count FROM library_passages"
            ).fetchone()["count"]
            kinds = connection.execute(
                "SELECT kind, COUNT(*) AS count FROM library_documents GROUP BY kind ORDER BY kind"
            ).fetchall()
        kind_text = ", ".join(f"{row['kind']}={row['count']}" for row in kinds) or "none"
        return (
            f"Local library: {totals['documents']} documents, {totals['pages']} pages, "
            f"{passages} searchable passages, extraction_errors={totals['errors']}.\n"
            f"Formats: {kind_text}. Refresh: indexed={refresh['indexed']}, "
            f"unchanged={refresh['unchanged']}, removed={refresh['removed']}, "
            f"updates_deferred={refresh['updates_deferred']}.\n"
            "Library contents are owner-managed source material and remain separate from durable memory."
        )

    def search_library(self, query: str, max_results: int = 6) -> str:
        refresh = self.refresh_index()
        if not refresh.get("available"):
            return str(refresh["message"])
        expression = self._fts_query(query)
        limit = max(1, min(int(max_results), int(self.config.max_search_results)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT path, page, passage,
                       snippet(library_passages, 3, '[', ']', ' … ', 32) AS excerpt,
                       bm25(library_passages) AS score
                FROM library_passages
                WHERE library_passages MATCH ?
                ORDER BY score, path, CAST(page AS INTEGER), CAST(passage AS INTEGER)
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        if not rows:
            return f"Local library search: {query!r}\nNo indexed library passages matched."
        output = [f"Local library search: {query!r}"]
        for index, row in enumerate(rows, start=1):
            citation = f"library://{row['path']}#page={row['page']}&passage={row['passage']}"
            excerpt = self._normalize_text(str(row["excerpt"]))[:900]
            output.append(f"[{index}] {citation}\n{excerpt}")
        rendered = "\n\n".join(output)
        return rendered[:_MAX_TOOL_OUTPUT_CHARS]

    def read_library_passage(
        self,
        path: str,
        page: int = 1,
        start_passage: int = 1,
        max_passages: int = 3,
    ) -> str:
        unavailable = self._availability_error()
        if unavailable:
            return unavailable
        _, relative = self._resolve_source(path)
        self.refresh_index()
        page = max(1, int(page))
        start_passage = max(1, int(start_passage))
        max_passages = max(1, min(int(max_passages), 6))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT passage, text FROM library_passages
                WHERE path = ? AND page = ? AND passage >= ?
                ORDER BY CAST(passage AS INTEGER)
                LIMIT ?
                """,
                (relative, page, start_passage, max_passages),
            ).fetchall()
            document = connection.execute(
                "SELECT error FROM library_documents WHERE path = ?", (relative,)
            ).fetchone()
        if not rows:
            if document is not None and str(document["error"]):
                return f"Library extraction failed for {relative}: {document['error']}"
            return (
                f"No indexed library passage exists at {relative} page {page}, "
                f"starting at passage {start_passage}."
            )
        output = [f"Library source: library://{relative}#page={page}"]
        for row in rows:
            output.append(f"Passage {row['passage']}:\n{row['text']}")
        return "\n\n".join(output)[:_MAX_TOOL_OUTPUT_CHARS]
