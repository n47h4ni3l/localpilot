from types import SimpleNamespace

from localpilot.cli import _console_safe_text


def test_model_answer_is_safe_for_legacy_windows_console_encoding():
    console = SimpleNamespace(file=SimpleNamespace(encoding="cp1252"))

    rendered = _console_safe_text(console, "lines\u202f1–4 and snowman ☃")

    assert rendered == "lines 1-4 and snowman ?"
    rendered.encode("cp1252")


def test_utf8_console_keeps_general_unicode_after_spacing_normalization():
    console = SimpleNamespace(file=SimpleNamespace(encoding="utf-8"))

    assert _console_safe_text(console, "lines\u202f1–4 and snowman ☃") == "lines 1–4 and snowman ☃"
