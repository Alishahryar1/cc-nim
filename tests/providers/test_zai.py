"""Tests for Z.ai native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import ZAI_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.zai import ZaiProvider


@pytest.fixture
def zai_config():
    return ProviderConfig(
        api_key="test_zai_key",
        base_url=ZAI_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def zai_provider(zai_config):
    return ZaiProvider(zai_config)


def test_init(zai_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = ZaiProvider(zai_config)
    assert provider._api_key == "test_zai_key"
    assert provider._base_url == ZAI_DEFAULT_BASE
    assert mock_client.called


def test_request_headers(zai_provider):
    h = zai_provider._request_headers()
    assert h["x-api-key"] == "test_zai_key"
    assert h["anthropic-version"] == "2023-06-01"


def test_model_list_headers(zai_provider):
    h = zai_provider._model_list_headers()
    assert h["x-api-key"] == "test_zai_key"


def test_build_request_body_native(zai_provider):
    request = MessagesRequest(
        model="glm-5.2",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    body = zai_provider._build_request_body(request)
    assert body["model"] == "glm-5.2"
    assert body["stream"] is True
    assert body["max_tokens"] == 100


def test_build_request_body_default_max_tokens(zai_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = zai_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_rejects_extra_body(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"x": 1},
        }
    )
    with pytest.raises(InvalidRequestError, match="does not support extra_body"):
        zai_provider._build_request_body(request)


def test_build_request_body_injects_zai_mcp_servers_by_default(zai_provider):
    """The three HTTP z.ai MCP servers are injected as native mcp_servers."""
    request = MessagesRequest(
        model="glm-5.2",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    body = zai_provider._build_request_body(request)

    names = {server["name"] for server in body["mcp_servers"]}
    assert {"web-reader", "web-search-prime", "zread"}.issubset(names)
    assert body["mcp_servers"][0]["type"] == "url"
    assert body["mcp_servers"][0]["authorization_token"] == "test_zai_key"


def test_build_request_body_mcp_injection_disabled(monkeypatch, zai_config):
    """ZAI_INJECT_MCP_SERVERS=false suppresses injection entirely."""
    monkeypatch.setenv("ZAI_INJECT_MCP_SERVERS", "false")
    provider = ZaiProvider(zai_config)
    request = MessagesRequest(
        model="glm-5.2",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    body = provider._build_request_body(request)
    assert "mcp_servers" not in body


def test_build_request_body_mcp_injection_skipped_without_api_key(zai_config):
    """No injection when the provider has no API key to authenticate with."""
    zai_config.api_key = ""
    provider = ZaiProvider(zai_config)
    request = MessagesRequest(
        model="glm-5.2",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    body = provider._build_request_body(request)
    assert "mcp_servers" not in body


def test_build_request_body_mcp_injection_preserves_client_servers(zai_provider):
    """Client-provided mcp_servers are kept and win on name collisions."""
    request = MessagesRequest.model_validate(
        {
            "model": "glm-5.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [
                {"type": "url", "url": "https://x/mcp", "name": "mine"},
                {"type": "url", "url": "https://y/mcp", "name": "web-reader"},
            ],
        }
    )
    body = zai_provider._build_request_body(request)

    names = [server["name"] for server in body["mcp_servers"]]
    assert names[:2] == ["mine", "web-reader"]
    assert {"web-search-prime", "zread"}.issubset(set(names))
    assert names.count("web-reader") == 1
    assert body["mcp_servers"][1]["url"] == "https://y/mcp"


@pytest.mark.asyncio
async def test_cleanup_aclose(zai_provider):
    zai_provider._client = AsyncMock()

    await zai_provider.cleanup()

    zai_provider._client.aclose.assert_awaited_once()
