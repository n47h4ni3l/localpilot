from __future__ import annotations

from types import SimpleNamespace

import pytest

from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.web import fetch_public_https, search_public_web


class _Headers:
    def __init__(self, content_type: str = "text/html") -> None:
        self.content_type = content_type

    def get_content_type(self):
        return self.content_type

    def get_content_charset(self):
        return "utf-8"

    def get(self, name, default=None):
        return default


class _Response:
    def __init__(self, url: str, body: str, content_type: str = "text/html") -> None:
        self._url = url
        self._body = body.encode("utf-8")
        self.headers = _Headers(content_type)

    def geturl(self):
        return self._url

    def read(self, limit=-1):
        return self._body if limit < 0 else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Opener:
    def __init__(self, search_html: str, article_html: str = "") -> None:
        self.search_html = search_html
        self.article_html = article_html
        self.urls: list[str] = []

    def open(self, request, timeout=15):
        url = request.full_url
        self.urls.append(url)
        if "duckduckgo.com" in url:
            return _Response(url, self.search_html)
        if url == "https://example.com/article":
            return _Response(url, self.article_html)
        raise AssertionError(f"unexpected URL: {url}")


def _public_dns(*args, **kwargs):
    return [(None, None, None, None, ("93.184.216.34", 443))]


def test_web_search_returns_only_bounded_https_leads(monkeypatch):
    html = """
    <html><body>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">Useful result</a>
      <a class="result__a" href="//duckduckgo.com/y.js?ad_domain=example.invalid">Tracking ad</a>
      <a class="result__a" href="http://insecure.example/item">Insecure result</a>
      <a class="result__a" href="https://127.0.0.1/private">Private literal</a>
      <a class="result__a" href="https://second.example/path">Second result</a>
    </body></html>
    """
    opener = _Opener(html)
    monkeypatch.setattr("localpilot.tools.web.socket.getaddrinfo", _public_dns)
    monkeypatch.setattr("localpilot.tools.web.urllib.request.build_opener", lambda *args: opener)

    result = search_public_web("LocalPilot architecture", max_results=2)

    assert "Useful result" in result
    assert "https://example.com/article" in result
    assert "Second result" in result
    assert "http://insecure.example" not in result
    assert "127.0.0.1" not in result
    assert "Tracking ad" not in result
    assert "untrusted discovery leads" in result


def test_search_to_read_flow_keeps_discovery_separate_from_evidence(monkeypatch):
    search_html = (
        '<a class="result__a" href="https://example.com/article">Authoritative-looking result</a>'
    )
    article_html = "<html><body><h1>Observed page</h1><p>Verified after an explicit page read.</p></body></html>"
    opener = _Opener(search_html, article_html)
    monkeypatch.setattr("localpilot.tools.web.socket.getaddrinfo", _public_dns)
    monkeypatch.setattr("localpilot.tools.web.urllib.request.build_opener", lambda *args: opener)

    discovered = search_public_web("test query", max_results=1)
    assert "https://example.com/article" in discovered
    assert "Verified after an explicit page read" not in discovered

    page = fetch_public_https("https://example.com/article")
    assert "HTTPS source: https://example.com/article" in page
    assert "Verified after an explicit page read." in page
    assert len(opener.urls) == 2


def test_web_search_validates_query_and_limits(monkeypatch):
    with pytest.raises(ValueError, match="must not be empty"):
        search_public_web("   ")
    with pytest.raises(ValueError, match="too long"):
        search_public_web("x" * 501)

    many = "".join(
        f'<a class="result__a" href="https://example{i}.com/path">Result {i}</a>'
        for i in range(20)
    )
    opener = _Opener(many)
    monkeypatch.setattr("localpilot.tools.web.socket.getaddrinfo", _public_dns)
    monkeypatch.setattr("localpilot.tools.web.urllib.request.build_opener", lambda *args: opener)
    result = search_public_web("bounded", max_results=50)
    assert result.count("\n   https://") == 10


def test_registry_exposes_web_discovery_as_read_only(tmp_path):
    tools = registry(tmp_path)
    assert tools["search_public_web"].risk is RiskLevel.READ_ONLY
    assert tools["fetch_public_https"].risk is RiskLevel.READ_ONLY
