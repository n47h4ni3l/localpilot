from __future__ import annotations

from localpilot import webview_app


def _css(name: str) -> str:
    return (webview_app.WEBVIEW_DIR / name).read_text(encoding="utf-8")


def test_legacy_chrome_no_longer_overrides_comic_panel_geometry():
    chrome = _css("chrome-polish.css")
    comic = _css("comic-shell.css")

    # The screenshot regression came from the old 100%-wide panel being moved
    # 34px left by the comic right inset, clipping the first characters of
    # LocalPilot's labels and replies. comic-shell.css must own the bubble size.
    assert ".app.is-expanded .panel {\n  width: 100%;" not in chrome
    assert "width: var(--chat-panel-width);" in comic
    assert "right: var(--chat-right-inset);" in comic
    assert "--chat-panel-width: 458px;" in comic
    assert "--chat-right-inset: 34px;" in comic


def test_closed_history_sheet_cannot_bleed_into_speech_tail_space():
    chrome = _css("chrome-polish.css")

    # The panel deliberately overflows so the speech tail can protrude. The old
    # translated history drawer therefore appeared as a cream strip with dots
    # beside the bubble even while history was closed.
    assert ".history-sheet:not(.is-open)" in chrome
    assert "visibility: hidden;" in chrome
    assert "pointer-events: none;" in chrome
    assert ".history-sheet.is-open" in chrome
    assert "visibility: visible;" in chrome


def test_message_stream_never_gains_a_window_level_horizontal_scrollbar():
    chrome = _css("chrome-polish.css")

    assert ".app.is-expanded .message-stream" in chrome
    assert "overflow-x: hidden;" in chrome
    assert ".app.is-expanded .reveal-text pre" in chrome
    assert "overflow-x: auto;" in chrome


def test_comic_toolbar_exposes_only_the_real_close_control():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    chrome = _css("chrome-polish.css")

    assert index.count('aria-label="Close chat bubble"') == 1
    assert 'id="collapse-btn"' in index
    assert "#close-app-btn" in chrome
    assert "display: none !important;" in chrome
