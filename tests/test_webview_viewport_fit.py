from pathlib import Path

from localpilot import webview_app


def test_expanded_panel_fills_actual_webview_viewport():
    css = (webview_app.WEBVIEW_DIR / "chrome-polish.css").read_text(encoding="utf-8")
    assert ".app.is-expanded .panel" in css
    assert "width: 100%;" in css
    assert "height: 100%;" in css
    assert "max-width: none;" in css
    assert "max-height: none;" in css


def test_polish_stylesheet_is_loaded_after_base_stylesheet():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    assert index.index('href="app.css"') < index.index('href="chrome-polish.css"')
