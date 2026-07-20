"""Local FCC transport must never escape through an outbound proxy."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.cli.local_http import with_local_proxy_bypass


@contextmanager
def _status_server(status_code: int) -> Iterator[tuple[str, list[str]]]:
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(status_code)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_proxy_preflight_connects_directly_when_http_proxy_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _status_server(502) as (forward_proxy_url, forward_proxy_hits):
        monkeypatch.setenv("HTTP_PROXY", forward_proxy_url)
        monkeypatch.setenv("http_proxy", forward_proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        with _status_server(200) as (fcc_url, fcc_hits):
            assert preflight_proxy(fcc_url) is None

    assert fcc_hits == ["/health"]
    assert forward_proxy_hits == []


def test_child_proxy_bypass_preserves_existing_proxy_policy() -> None:
    base_env = {
        "HTTP_PROXY": "http://proxy.example:3128",
        "NO_PROXY": "example.com, localhost",
        "no_proxy": "10.0.0.0/8,EXAMPLE.COM",
        "KEEP_ME": "yes",
    }

    env = with_local_proxy_bypass(
        base_env,
        proxy_root_url="http://fcc.internal:8082",
    )

    assert env["HTTP_PROXY"] == "http://proxy.example:3128"
    assert env["KEEP_ME"] == "yes"
    assert env["NO_PROXY"] == (
        "example.com,localhost,10.0.0.0/8,127.0.0.1,::1,fcc.internal"
    )
    assert env["no_proxy"] == env["NO_PROXY"]
    assert base_env["NO_PROXY"] == "example.com, localhost"
    assert base_env["no_proxy"] == "10.0.0.0/8,EXAMPLE.COM"


def test_child_proxy_bypass_uses_all_loopback_spellings_without_duplicates() -> None:
    env = with_local_proxy_bypass(
        {"NO_PROXY": "127.0.0.1"},
        proxy_root_url="http://127.0.0.1:8082",
    )

    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]
