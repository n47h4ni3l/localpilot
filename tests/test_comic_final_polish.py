from __future__ import annotations

from localpilot import webview_app


def _read(name: str) -> str:
    return (webview_app.WEBVIEW_DIR / name).read_text(encoding="utf-8")


def test_final_polish_stylesheet_loads_after_comic_shell():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    assert 'href="comic-final-polish.css"' in index
    assert index.index('href="comic-shell.css"') < index.index('href="comic-final-polish.css"')


def test_final_polish_reasserts_transparent_host_surface():
    polish = _read("comic-final-polish.css")
    assert ".app.is-expanded" in polish
    assert "background: transparent !important;" in polish
    assert "box-shadow: none !important;" in polish


def test_settings_actions_and_switches_use_comic_palette():
    polish = _read("comic-final-polish.css")
    assert ".settings-popover button.action" in polish
    assert "color: var(--comic-ink);" in polish
    assert ".switch.is-on" in polish
    assert "background: var(--comic-accent-soft);" in polish
    assert ".switch.is-on::after" in polish
    assert "background: var(--comic-accent);" in polish


def test_systemsense_metric_details_wrap_instead_of_ellipsis():
    polish = _read("comic-final-polish.css")
    assert ".system-metric small" in polish
    assert "white-space: normal;" in polish
    assert "text-overflow: clip;" in polish


def test_comic_scrollbars_are_styled_but_code_scrolling_remains_separate():
    polish = _read("comic-final-polish.css")
    chrome = _read("chrome-polish.css")
    assert ".message-stream::-webkit-scrollbar-thumb" in polish
    assert ".system-content::-webkit-scrollbar-thumb" in polish
    assert "background-clip: content-box;" in polish
    assert ".app.is-expanded .reveal-text pre" in chrome
    assert "overflow-x: auto;" in chrome
