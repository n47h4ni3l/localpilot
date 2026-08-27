from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from localpilot.config import Config, LibraryConfig, load_config
from localpilot.cli import build_parser
from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.library import LocalLibrary


def _library(tmp_path, **overrides):
    root = tmp_path / "library"
    root.mkdir()
    values = {
        "enabled": True,
        "root": str(root),
        "max_documents": 20,
        "max_refresh_files": 20,
        "max_file_size_mb": 2,
        "max_pages_per_document": 20,
        "max_chars_per_page": 10_000,
        "max_search_results": 8,
    }
    values.update(overrides)
    config = LibraryConfig(**values)
    return root, LocalLibrary(config, tmp_path / "data" / "library.sqlite3")


def test_text_library_indexes_searches_and_reads_with_source_citations(tmp_path):
    root, library = _library(tmp_path)
    (root / "manuals").mkdir()
    (root / "manuals" / "power.txt").write_text(
        "A reversible power change should preserve the original scheme. "
        "Restoration must verify the current state before applying the saved value.",
        encoding="utf-8",
    )

    results = library.search_library("reversible power restoration")
    passage = library.read_library_passage("manuals/power.txt", page=1)
    summary = library.get_library_summary()

    assert "library://manuals/power.txt#page=1&passage=1" in results
    assert "preserve the original scheme" in passage
    assert "1 documents" in summary
    assert "durable memory" in summary


def test_refresh_reindexes_changed_files_and_removes_deleted_sources(tmp_path):
    root, library = _library(tmp_path)
    source = root / "notes.md"
    source.write_text("The original observation is amber.", encoding="utf-8")
    assert "notes.md" in library.search_library("amber")

    source.write_text("The replacement observation is cobalt blue.", encoding="utf-8")
    assert "notes.md" in library.search_library("cobalt")
    assert "No indexed library passages matched" in library.search_library("amber")

    source.unlink()
    summary = library.get_library_summary()
    assert "0 documents" in summary


def test_refresh_budget_reports_only_unprocessed_updates_as_deferred(tmp_path):
    root, library = _library(tmp_path, max_refresh_files=1)
    (root / "a.txt").write_text("alpha reference", encoding="utf-8")
    (root / "b.txt").write_text("beta reference", encoding="utf-8")

    first = library.refresh_index()
    second = library.refresh_index()

    assert first["indexed"] == 1
    assert first["updates_deferred"] == 1
    assert second["indexed"] == 1
    assert second["updates_deferred"] == 0


def test_library_rejects_traversal_hidden_and_symlink_sources(tmp_path):
    root, library = _library(tmp_path)
    (root / "safe.txt").write_text("safe reference", encoding="utf-8")
    (root / ".private.txt").write_text("hidden secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        library.read_library_passage("../outside.txt")
    with pytest.raises(ValueError, match="Hidden"):
        library.read_library_passage(".private.txt")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    link = root / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform/account.")
    with pytest.raises(ValueError, match="Symlink"):
        library.read_library_passage("linked.txt")
    assert "outside secret" not in library.search_library("secret")


def test_pdf_pages_are_bounded_and_keep_page_citations(tmp_path, monkeypatch):
    root, library = _library(tmp_path, max_pages_per_document=2)
    (root / "book.pdf").write_bytes(b"%PDF-fake-for-bounded-reader-test")

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        is_encrypted = False

        def __init__(self, path, strict=False):
            self.pages = [
                Page("First page discusses curiosity and exploration."),
                Page("Second page discusses evidence and restraint."),
                Page("Third page must remain outside the configured page bound."),
            ]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))

    first = library.search_library("curiosity")
    second = library.search_library("restraint")
    omitted = library.search_library("outside configured bound")

    assert "library://book.pdf#page=1" in first
    assert "library://book.pdf#page=2" in second
    assert "No indexed library passages matched" in omitted


def test_registry_exposes_library_only_when_enabled(tmp_path):
    config = Config()
    config.agent.data_dir = "data"
    config.library.enabled = True
    config.library.root = str(tmp_path / "library")
    (tmp_path / "library").mkdir()

    tools = registry(tmp_path, config=config)

    names = {"get_library_summary", "search_library", "read_library_passage"}
    assert names <= tools.keys()
    assert all(tools[name].risk is RiskLevel.READ_ONLY for name in names)
    assert names.isdisjoint(registry(tmp_path).keys())


def test_library_config_is_bounded_and_uses_a_separate_cache_database(tmp_path):
    path = tmp_path / "localpilot.toml"
    path.write_text(
        "[library]\n"
        "enabled = true\n"
        f"root = {str(tmp_path / 'books')!r}\n"
        "index_database = 'references.sqlite3'\n"
        "max_search_results = 4\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.library.enabled is True
    assert config.library.index_database == "references.sqlite3"
    assert config.library.max_search_results == 4

    collision = tmp_path / "collision.toml"
    collision.write_text(
        "[library]\nindex_database = 'learning.sqlite3'\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="separate"):
        load_config(collision)

    excessive = tmp_path / "excessive.toml"
    excessive.write_text(
        "[library]\nmax_search_results = 50000\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="between"):
        load_config(excessive)


def test_library_cli_has_status_index_and_search_modes():
    parser = build_parser()

    status = parser.parse_args(["library", "status"])
    index = parser.parse_args(["library", "index"])
    search = parser.parse_args(["library", "search", "power management"])

    assert status.library_action == "status"
    assert index.library_action == "index"
    assert search.query == "power management"


def test_agent_function_surface_can_hide_owner_prohibited_web_tools(tmp_path):
    config = Config()
    config.agent.data_dir = "data"
    config.library.enabled = True
    config.library.root = str(tmp_path / "library")
    (tmp_path / "library").mkdir()

    from localpilot.agent import LocalPilotAgent

    agent = LocalPilotAgent(config, tmp_path)
    functions = agent._functions(
        excluded_tools=frozenset({"search_public_web", "fetch_public_https"})
    )
    names = {
        function.__name__ if callable(function) else function["function"]["name"]
        for function in functions
    }

    assert "search_library" in names
    assert "search_public_web" not in names
    assert "fetch_public_https" not in names

    after_discovery = agent._functions(excluded_tools=frozenset({"search_library"}))
    after_discovery_names = {
        function.__name__ if callable(function) else function["function"]["name"]
        for function in after_discovery
    }
    assert "search_library" not in after_discovery_names
    assert "read_library_passage" in after_discovery_names


def test_explicit_library_turn_routes_first_observation_to_library(tmp_path, monkeypatch):
    config = Config()
    config.agent.data_dir = "data"
    config.library.enabled = True
    config.library.root = str(tmp_path / "library")
    (tmp_path / "library").mkdir()

    from localpilot.agent import LocalPilotAgent

    agent = LocalPilotAgent(config, tmp_path)

    def inspect_first_call(**kwargs):
        names = {
            function.__name__ if callable(function) else function["function"]["name"]
            for function in kwargs["tools"]
        }
        assert {"get_library_summary", "search_library", "read_library_passage"} <= names
        assert "search_repository" not in names
        assert "fetch_public_https" not in names
        raise RuntimeError("first tool surface inspected")

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=inspect_first_call))

    with pytest.raises(RuntimeError, match="first tool surface inspected"):
        agent.ask("Consult the local library before answering, and do not use the public web.")
