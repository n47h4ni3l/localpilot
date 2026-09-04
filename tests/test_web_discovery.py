from __future__ import annotations

import gzip
import urllib.error

import pytest

from localpilot.safety import RiskLevel
from localpilot.tools import registry
from localpilot.tools.web import fetch_public_https, search_public_web


class _Headers:
    def __init__(self, content_type: str = "text/html", content_encoding: str = "identity") -> None:
        self.content_type = content_type
        self.content_encoding = content_encoding

    def get_content_type(self):
        return self.content_type

    def get_content_charset(self):
        return "utf-8"

    def get(self, name, default=None):
        if str(name).lower() == "content-encoding":
            return self.content_encoding
        return default


class _Response:
    def __init__(
        self,
        url: str,
        body: str,
        content_type: str = "text/html",
        content_encoding: str = "identity",
    ) -> None:
        self._url = url
        raw = body.encode("utf-8")
        self._body = gzip.compress(raw) if content_encoding == "gzip" else raw
        self.headers = _Headers(content_type, content_encoding)

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


def test_web_search_retries_transient_socket_denial(monkeypatch):
    html = '<a class="result__a" href="https://example.com/article">Recovered result</a>'

    class TransientOpener:
        def __init__(self):
            self.calls = 0

        def open(self, request, timeout=15):
            self.calls += 1
            if self.calls == 1:
                raise urllib.error.URLError(
                    PermissionError(10013, "socket access was temporarily denied")
                )
            return _Response(request.full_url, html)

    opener = TransientOpener()
    delays = []
    monkeypatch.setattr("localpilot.tools.web.socket.getaddrinfo", _public_dns)
    monkeypatch.setattr("localpilot.tools.web.urllib.request.build_opener", lambda *args: opener)
    monkeypatch.setattr("localpilot.tools.web.time.sleep", delays.append)

    result = search_public_web("recover transient search", max_results=1)

    assert "Recovered result" in result
    assert opener.calls == 2
    assert delays == [0.25]


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


def test_https_reader_decodes_bounded_gzip_text(monkeypatch):
    article_html = "<html><body><h1>Python 3.14.7</h1><p>Latest stable release.</p></body></html>"

    class GzipOpener:
        def open(self, request, timeout=15):
            assert request.headers["Accept-encoding"] == "gzip, deflate, identity"
            return _Response(
                request.full_url,
                article_html,
                content_encoding="gzip",
            )

    monkeypatch.setattr("localpilot.tools.web.socket.getaddrinfo", _public_dns)
    monkeypatch.setattr(
        "localpilot.tools.web.urllib.request.build_opener",
        lambda *args: GzipOpener(),
    )

    page = fetch_public_https("https://example.com/article")

    assert "Python 3.14.7" in page
    assert "Latest stable release." in page


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


def test_resolve_validated_address_rejects_private_and_accepts_public(monkeypatch):
    from localpilot.tools.web import _resolve_validated_address

    monkeypatch.setattr(
        "localpilot.tools.web.socket.getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("172.16.0.9", port))],
    )
    with pytest.raises(OSError, match="private, reserved"):
        _resolve_validated_address("attacker.example", 443)

    monkeypatch.setattr(
        "localpilot.tools.web.socket.getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("93.184.216.34", port))],
    )
    assert _resolve_validated_address("example.com", 443) == "93.184.216.34"


def test_validated_https_connection_rejects_rebinding_at_connect_time(monkeypatch):
    """DNS-rebinding regression test. A naive implementation validates a
    hostname once and lets the real TCP connection re-resolve it later --
    an attacker's DNS server can answer differently between those two
    lookups. This asserts the connection performs its own resolution and
    validation *inside* connect(), so whatever the real connection would
    use is exactly what gets checked, with no separate answer to spoof."""
    from localpilot.tools.web import _ValidatedHTTPSConnection

    monkeypatch.setattr(
        "localpilot.tools.web.socket.getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("10.0.0.5", port))],
    )
    def _must_not_connect(*args, **kwargs):
        raise AssertionError("must not connect to a rebound private address")

    monkeypatch.setattr("localpilot.tools.web.socket.create_connection", _must_not_connect)

    conn = _ValidatedHTTPSConnection("attacker.example", timeout=15)
    with pytest.raises(OSError, match="private, reserved"):
        conn.connect()


def test_validated_https_connection_uses_the_freshly_resolved_public_address(monkeypatch):
    from localpilot.tools.web import _ValidatedHTTPSConnection

    monkeypatch.setattr(
        "localpilot.tools.web.socket.getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("93.184.216.34", port))],
    )
    connect_calls = []

    def _fake_create_connection(address, timeout=None):
        connect_calls.append(address)
        return object()

    monkeypatch.setattr("localpilot.tools.web.socket.create_connection", _fake_create_connection)

    class _FakeContext:
        def __init__(self):
            self.wrapped_with_hostname = None

        def wrap_socket(self, sock, server_hostname=None):
            self.wrapped_with_hostname = server_hostname
            return sock

    context = _FakeContext()
    conn = _ValidatedHTTPSConnection("example.com", timeout=15, context=context)
    conn.connect()

    assert connect_calls == [("93.184.216.34", 443)]
    # TLS still verifies the certificate against the real hostname, not the
    # resolved IP -- pinning the address must not weaken certificate checks.
    assert context.wrapped_with_hostname == "example.com"
