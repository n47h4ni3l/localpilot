from __future__ import annotations

from pathlib import Path

import pytest

from localpilot.selfdev import CandidateTools


ROOT = Path(__file__).resolve().parents[1]
WEBVIEW = ROOT / "localpilot" / "webview"


@pytest.mark.parametrize("name", ["index.html", "app.css", "app.js"])
def test_current_frontend_assets_are_valid_candidate_writes(tmp_path, name):
    tools = CandidateTools(tmp_path)
    content = (WEBVIEW / name).read_text(encoding="utf-8")
    result = tools.write_project_file(f"localpilot/webview/{name}", content)
    assert result.startswith("Wrote localpilot/webview/")


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("site/index.html", "<!doctype html>"),
        ("assets/app.css", "body {}"),
        ("scripts/app.js", "'use strict';"),
    ],
)
def test_frontend_suffixes_are_confined_to_the_companion_directory(tmp_path, path, content):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError, match="only inside localpilot/webview"):
        tools.write_project_file(path, content)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'">'
            '<script src="https://example.com/app.js"></script>',
            "local relative files",
        ),
        (
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'">'
            '<button onclick="steal()">x</button>',
            "Inline frontend HTML attribute",
        ),
        ("<!doctype html><title>Missing policy</title>", "must declare a Content-Security-Policy"),
    ],
)
def test_html_candidate_validation_rejects_remote_active_or_unprotected_pages(tmp_path, content, message):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError, match=message):
        tools.write_project_file("localpilot/webview/candidate.html", content)


@pytest.mark.parametrize(
    "content",
    [
        '@import url("https://example.com/theme.css");',
        'body { background-image: url("../../broker-token"); }',
    ],
)
def test_css_candidate_validation_rejects_remote_or_escaping_resources(tmp_path, content):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError):
        tools.write_project_file("localpilot/webview/candidate.css", content)


@pytest.mark.parametrize(
    "content",
    [
        'fetch("https://example.com/collect");',
        'localStorage.setItem("token", TOKEN);',
        'node.innerHTML = untrusted;',
        'bridge("read_conversations");',
        'window.pywebview.api.expand();',
    ],
)
def test_javascript_candidate_validation_preserves_token_and_bridge_boundaries(tmp_path, content):
    tools = CandidateTools(tmp_path)
    with pytest.raises(ValueError):
        tools.write_project_file("localpilot/webview/candidate.js", content)


def test_recovered_frontend_candidate_is_revalidated(tmp_path):
    path = tmp_path / "localpilot" / "webview" / "app.js"
    path.parent.mkdir(parents=True)
    path.write_text('fetch("https://example.com/collect");', encoding="utf-8")
    with pytest.raises(ValueError, match="remote URL"):
        CandidateTools(tmp_path, existing_changed_paths=["localpilot/webview/app.js"])


def test_static_checks_validate_frontend_files_without_executing_them(tmp_path):
    tools = CandidateTools(tmp_path)
    for name in ("index.html", "app.css", "app.js"):
        tools.write_project_file(
            f"localpilot/webview/{name}",
            (WEBVIEW / name).read_text(encoding="utf-8"),
        )
    result = tools.run_candidate_static_checks()
    assert result.startswith("static_checks=passed")
    assert "frontend_files=3" in result
    assert "javascript_files=1" in result
