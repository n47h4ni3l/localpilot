from pathlib import Path

from localpilot import webview_app


def test_comic_shell_owns_expanded_panel_geometry_inside_webview_viewport():
    chrome = (webview_app.WEBVIEW_DIR / "chrome-polish.css").read_text(encoding="utf-8")
    comic = (webview_app.WEBVIEW_DIR / "comic-shell.css").read_text(encoding="utf-8")

    # The old full-viewport rule shifted the comic panel left by its 34px
    # speech-tail inset and clipped LocalPilot's first characters. The host
    # still fills the viewport, but the bubble itself now keeps explicit comic
    # geometry with room for its tail.
    assert ".app.is-expanded .panel {\n  width: 100%;" not in chrome
    assert "--chat-panel-width: 458px;" in comic
    assert "width: var(--chat-panel-width);" in comic
    assert "right: var(--chat-right-inset);" in comic
    assert "top: var(--surface-top-inset);" in comic
    assert "bottom: var(--surface-bottom-inset);" in comic


def test_comic_shell_is_loaded_after_base_and_chrome_stylesheets():
    index = webview_app.INDEX_HTML.read_text(encoding="utf-8")
    assert index.index('href="app.css"') < index.index('href="chrome-polish.css"')
    assert index.index('href="chrome-polish.css"') < index.index('href="comic-shell.css"')
