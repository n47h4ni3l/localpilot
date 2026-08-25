from types import SimpleNamespace

from rich.console import Console
from rich.table import Table

from localpilot.cli import _ConsoleSafeWriter, _console_safe_text, _console_safe_value


class _LegacyStream:
    encoding = "cp1252"

    def __init__(self):
        self.value = ""

    def write(self, value):
        value.encode(self.encoding)
        self.value += value
        return len(value)

    def flush(self):
        pass

    def isatty(self):
        return False


def test_model_answer_is_safe_for_legacy_windows_console_encoding():
    console = SimpleNamespace(file=SimpleNamespace(encoding="cp1252"))

    rendered = _console_safe_text(console, "lines\u202f1–4 and snowman ☃")

    assert rendered == "lines 1-4 and snowman ?"
    rendered.encode("cp1252")


def test_utf8_console_keeps_general_unicode_after_spacing_normalization():
    console = SimpleNamespace(file=SimpleNamespace(encoding="utf-8"))

    assert _console_safe_text(console, "lines\u202f1–4 and snowman ☃") == "lines 1–4 and snowman ☃"


def test_safe_writer_covers_rich_tables_and_stored_unicode_text():
    stream = _LegacyStream()
    console = Console(file=_ConsoleSafeWriter(stream), force_terminal=False, width=80)
    table = Table()
    table.add_column("State")
    table.add_row("deferred — user busy; policy—not candidate‑branch; snowman ☃")

    console.print(table)

    assert "deferred - user busy; policy - not candidate-branch; snowman ?" in stream.value
    stream.value.encode("cp1252")


def test_lone_em_dash_placeholder_stays_compact():
    assert _console_safe_value("—", "cp1252") == "-"
