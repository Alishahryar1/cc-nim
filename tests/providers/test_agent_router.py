"""Tests for AgentRouter native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import AGENTROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.agent_router import AgentRouterProvider
from free_claude_code.providers.base import ProviderConfig


@pytest.fixture
def agent_router_config():
    return ProviderConfig(
        api_key="test_agent_router_key",
        base_url=AGENTROUTER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "free_claude_code.providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def agent_router_provider(agent_router_config):
    return AgentRouterProvider(agent_router_config)


def test_init(agent_router_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = AgentRouterProvider(agent_router_config)
    assert provider._api_key == "test_agent_router_key"
    assert provider._base_url == AGENTROUTER_DEFAULT_BASE
    assert mock_client.called


def test_request_headers(agent_router_provider):
    h = agent_router_provider._request_headers()
    assert h["x-api-key"] == "test_agent_router_key"
    assert h["Authorization"] == "Bearer test_agent_router_key"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["Accept"] == "application/json"
    assert "claude-cli" in h["User-Agent"]
    assert "claude-code" in h["anthropic-beta"]
    assert h["x-app"] == "cli"


def test_request_headers_forward_client_fingerprint(agent_router_provider):
    request = MessagesRequest(
        model="claude-opus-5",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        client_headers={
            "x-claude-code-session-id": "session-123",
            "x-stainless-runtime-version": "v26.3.0",
            "anthropic-beta": "claude-code-20250219,fallback-credit-2026-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
            "x-arbitrary-header": "must-not-forward",
            "authorization": "must-not-forward",
        },
    )

    headers = agent_router_provider._request_headers_for_request(request)
    headers_lower = {k.lower(): v for k, v in headers.items()}
    assert headers_lower["x-claude-code-session-id"] == "session-123"
    assert headers_lower["x-stainless-runtime-version"] == "v26.3.0"
    assert (
        headers_lower["anthropic-beta"]
        == "claude-code-20250219,fallback-credit-2026-06-01"
    )
    assert headers_lower["anthropic-dangerous-direct-browser-access"] == "true"
    assert "x-arbitrary-header" not in headers_lower
    assert headers["Authorization"] == "Bearer test_agent_router_key"
    assert headers["x-api-key"] == "test_agent_router_key"


def test_messages_path_matches_claude_code_beta_transport(agent_router_provider):
    assert agent_router_provider._messages_path() == "/messages?beta=true"


def test_model_list_headers(agent_router_provider):
    h = agent_router_provider._model_list_headers()
    assert h["x-api-key"] == "test_agent_router_key"
    assert h["Authorization"] == "Bearer test_agent_router_key"
    assert "claude-cli" in h["User-Agent"]
    assert h["x-app"] == "cli"


def test_build_request_body_native(agent_router_provider):
    request = MessagesRequest(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    body = agent_router_provider._build_request_body(request)
    assert body["model"] == "claude-opus-4-6"
    assert body["stream"] is True
    assert body["max_tokens"] == 100


def test_build_request_body_default_max_tokens(agent_router_provider):
    request = MessagesRequest(
        model="claude-opus-4-6",
        messages=[Message(role="user", content="x")],
    )
    body = agent_router_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_cleanup_aclose(agent_router_provider):
    agent_router_provider._client = AsyncMock()

    await agent_router_provider.cleanup()

    agent_router_provider._client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_model_infos_fallback(agent_router_provider):
    with patch(
        "free_claude_code.providers.transports.anthropic_messages.transport.AnthropicMessagesTransport.list_model_infos",
        side_effect=RuntimeError("Endpoint /models not supported"),
    ):
        infos = await agent_router_provider.list_model_infos()
        model_ids = {info.model_id for info in infos}
        assert "claude-fable-5" in model_ids
        assert "claude-opus-5" in model_ids
        assert "claude-opus-4-8" in model_ids
        assert "claude-opus-4-6" in model_ids
        assert "glm-5.2" in model_ids
        assert "gpt-5.5" in model_ids
        assert "gpt-5.6-sol" in model_ids
        assert "kimi-k3" in model_ids
