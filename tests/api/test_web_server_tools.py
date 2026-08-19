from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import free_claude_code.api.web_tools.constants as web_tool_constants
from free_claude_code.api.web_tools import egress as web_egress
from free_claude_code.api.web_tools.egress import (
    WebDomainPolicy,
    WebDomainRule,
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
    parse_web_domain_rule,
)
from free_claude_code.api.web_tools.outbound import (
    _drain_response_body_capped,
    _read_response_body_capped,
    _run_web_fetch,
    _run_web_search,
)
from free_claude_code.core.version import package_version

_STRICT_EGRESS = WebFetchEgressPolicy(
    allow_private_network_targets=False,
    allowed_schemes=frozenset({"http", "https"}),
)


def test_web_tool_user_agent_reports_installed_package_version() -> None:
    assert {
        "User-Agent": f"Mozilla/5.0 compatible; free-claude-code/{package_version()}"
    } == web_tool_constants._WEB_TOOL_HTTP_HEADERS


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://localhost/foo",
        "http://mybox.local/",
        "file:///etc/passwd",
        "http://user:secret@example.com/",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_enforce_web_fetch_egress_blocks_internal_or_disallowed(url: str) -> None:
    with pytest.raises(WebFetchEgressViolation):
        enforce_web_fetch_egress(url, _STRICT_EGRESS)


def test_enforce_web_fetch_egress_allows_global_literal_ip() -> None:
    enforce_web_fetch_egress("http://8.8.8.8/", _STRICT_EGRESS)


def test_enforce_web_fetch_egress_skips_private_checks_when_opted_in() -> None:
    enforce_web_fetch_egress(
        "http://127.0.0.1/",
        WebFetchEgressPolicy(
            allow_private_network_targets=True,
            allowed_schemes=frozenset({"http", "https"}),
        ),
    )


def test_domain_policy_matches_parent_subdomains_and_search_paths() -> None:
    policy = WebDomainPolicy(
        allowed=(WebDomainRule("example.com", "/docs"),),
        blocked=(WebDomainRule("private.example.com"),),
    )

    assert policy.allows("https://www.example.com/docs/start")
    assert not policy.allows("https://www.example.com/blog")
    assert not policy.allows("https://private.example.com/docs/start")
    assert not policy.allows("https://example.net/docs/start")


@pytest.mark.parametrize("rule_path", ["/api", "/api/"])
def test_domain_rule_path_matches_only_exact_or_descendant_paths(
    rule_path: str,
) -> None:
    rule = WebDomainRule("example.com", rule_path)

    assert rule.matches("https://example.com/api")
    assert rule.matches("https://example.com/api/")
    assert rule.matches("https://example.com/api/child")
    assert not rule.matches("https://example.com/apiary")
    assert not rule.matches("https://example.com/api-v2")


def test_domain_policy_search_hints_preserve_blocked_paths() -> None:
    policy = WebDomainPolicy(blocked=(WebDomainRule("example.com", "/private"),))

    assert policy.search_query("docs") == "docs -site:example.com/private"


@pytest.mark.parametrize(
    ("value", "allow_path"),
    [
        ("https://example.com", True),
        ("example.com:443", True),
        ("*.example.com", True),
        ("tést.example", True),
        ("example.com/docs", False),
    ],
)
def test_domain_rule_rejects_unsupported_shapes(value: str, allow_path: bool) -> None:
    with pytest.raises(ValueError):
        parse_web_domain_rule(value, allow_path=allow_path)


def test_enforce_web_fetch_egress_documents_connect_time_pinning() -> None:
    assert enforce_web_fetch_egress.__doc__ and "resolved addresses" in (
        enforce_web_fetch_egress.__doc__ or ""
    )
    assert web_egress.get_validated_stream_addrinfos_for_egress.__doc__
    assert "pinning" in (
        web_egress.get_validated_stream_addrinfos_for_egress.__doc__ or ""
    )
    assert "DNS-pinned" in (_run_web_fetch.__doc__ or "")


def _aiohttp_response(
    status: int,
    *,
    url: str = "http://8.8.8.8/",
    location: str | None = None,
    body: bytes = b"hello world",
) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.url = url
    headers = {"content-type": "text/plain"}
    if location is not None:
        headers["location"] = location
    response.headers = headers
    response.get_encoding = MagicMock(return_value="utf-8")
    response.raise_for_status = MagicMock()
    response.request_info = MagicMock()
    response.history = ()

    async def iter_chunked(_size: int) -> AsyncIterator[bytes]:
        yield body

    response.content.iter_chunked = MagicMock(side_effect=iter_chunked)
    return response


def _aiohttp_client_session_patch(
    *responses: MagicMock,
) -> tuple[MagicMock, MagicMock]:
    queue = list(responses)
    index = 0

    def get_side(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal index
        response = queue[index] if index < len(queue) else queue[-1]
        index += 1
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=None)
        return context

    session = MagicMock()
    session.get = MagicMock(side_effect=get_side)
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=session)
    client_context.__aexit__ = AsyncMock(return_value=None)
    return client_context, session


@pytest.mark.asyncio
async def test_run_web_fetch_follows_redirect_when_each_hop_is_allowed() -> None:
    redirect = _aiohttp_response(
        302, url="http://8.8.8.8/start", location="/final", body=b""
    )
    success = _aiohttp_response(200, url="http://8.8.8.8/final")
    client_context, session = _aiohttp_client_session_patch(redirect, success)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession",
        return_value=client_context,
    ):
        result = await _run_web_fetch("http://8.8.8.8/start", _STRICT_EGRESS)

    assert result["data"] == "hello world"
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_run_web_fetch_applies_domain_policy_to_redirects() -> None:
    redirect = _aiohttp_response(
        302,
        url="http://8.8.8.8/start",
        location="http://1.1.1.1/secret",
        body=b"",
    )
    client_context, session = _aiohttp_client_session_patch(redirect)
    policy = WebFetchEgressPolicy(
        allow_private_network_targets=False,
        allowed_schemes=frozenset({"http", "https"}),
        domains=WebDomainPolicy(allowed=(WebDomainRule("8.8.8.8"),)),
    )
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_context,
        ),
        pytest.raises(WebFetchEgressViolation, match="domain policy"),
    ):
        await _run_web_fetch("http://8.8.8.8/start", policy)

    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_run_web_fetch_truncates_large_body_to_byte_cap(monkeypatch) -> None:
    success = _aiohttp_response(
        200,
        url="http://8.8.8.8/big",
        body=b"x" * 5_000,
    )
    client_context, _ = _aiohttp_client_session_patch(success)
    monkeypatch.setattr(web_tool_constants, "_MAX_WEB_FETCH_RESPONSE_BYTES", 100)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession",
        return_value=client_context,
    ):
        result = await _run_web_fetch("http://8.8.8.8/big", _STRICT_EGRESS)

    assert result["data"] == "x" * 100


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_without_location_raises() -> None:
    response = _aiohttp_response(302, url="http://8.8.8.8/here", body=b"")
    client_context, _ = _aiohttp_client_session_patch(response)
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_context,
        ),
        pytest.raises(WebFetchEgressViolation, match="missing Location"),
    ):
        await _run_web_fetch("http://8.8.8.8/here", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_run_web_fetch_excess_redirects_raises() -> None:
    first = _aiohttp_response(302, url="http://8.8.8.8/a", location="/b", body=b"")
    second = _aiohttp_response(302, url="http://8.8.8.8/b", location="/c", body=b"")
    client_context, _ = _aiohttp_client_session_patch(first, second)
    with (
        patch("free_claude_code.api.web_tools.constants._MAX_WEB_FETCH_REDIRECTS", 1),
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_context,
        ),
        pytest.raises(WebFetchEgressViolation, match="exceeded maximum redirects"),
    ):
        await _run_web_fetch("http://8.8.8.8/a", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_run_web_search_postfilters_backend_results(monkeypatch) -> None:
    html = b"""
    <a href="/?uddg=https%3A%2F%2Fdocs.example.com%2Fguide">Allowed</a>
    <a href="/?uddg=https%3A%2F%2Fevil.example%2Fguide">Blocked</a>
    """
    response = httpx.Response(
        200,
        content=html,
        request=httpx.Request("GET", "https://lite.duckduckgo.com/lite/"),
    )
    stream_context = MagicMock()
    stream_context.__aenter__ = AsyncMock(return_value=response)
    stream_context.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_context)
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=client)
    client_context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=client_context))

    results = await _run_web_search(
        "docs",
        WebDomainPolicy(allowed=(WebDomainRule("example.com"),)),
    )

    assert results == [{"title": "Allowed", "url": "https://docs.example.com/guide"}]
    assert client.stream.call_args.kwargs["params"] == {"q": "docs site:example.com"}


@pytest.mark.asyncio
async def test_read_response_body_capped_truncates_oversized_chunk() -> None:
    response = MagicMock()

    async def chunks(*, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 65_536
        yield b"x" * 200

    response.aiter_bytes = chunks
    assert await _read_response_body_capped(response, 10) == b"x" * 10


@pytest.mark.asyncio
async def test_drain_response_body_capped_stops_after_cap() -> None:
    response = MagicMock()
    consumed = 0

    async def chunks(*, chunk_size: int) -> AsyncIterator[bytes]:
        nonlocal consumed
        assert chunk_size == 65_536
        consumed += 1
        yield b"x" * 200
        consumed += 1
        yield b"y" * 200

    response.aiter_bytes = chunks
    await _drain_response_body_capped(response, 10)
    assert consumed == 1
