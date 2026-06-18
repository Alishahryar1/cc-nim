"""Tests for Fireworks AI native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.fireworks import FIREWORKS_BASE_URL, FireworksProvider


@pytest.fixture
def fireworks_config():
    return ProviderConfig(
        api_key="test_fireworks_key",
        base_url=FIREWORKS_BASE_URL,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def fireworks_provider(fireworks_config):
    return FireworksProvider(fireworks_config)


def test_init(fireworks_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = FireworksProvider(fireworks_config)
    assert provider._api_key == "test_fireworks_key"
    assert provider._base_url == FIREWORKS_BASE_URL
    assert mock_client.called


def test_base_url_constant():
    assert FIREWORKS_BASE_URL == "https://api.fireworks.ai/inference/v1"


def test_request_headers(fireworks_provider):
    h = fireworks_provider._request_headers()
    assert h["Authorization"] == "Bearer test_fireworks_key"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["Accept"] == "text/event-stream"


def test_build_request_body_native_shape(fireworks_provider):
    request = MessagesRequest(
        model="accounts/fireworks/models/glm-5p1",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="System prompt",
    )
    body = fireworks_provider._build_request_body(request)
    assert body["model"] == "accounts/fireworks/models/glm-5p1"
    assert body["stream"] is True
    assert body["max_tokens"] == 100
    assert body["system"] == "System prompt"
    assert body["messages"][0]["role"] == "user"


def test_build_request_body_default_max_tokens(fireworks_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = fireworks_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_global_disable_blocks_thinking():
    provider = FireworksProvider(
        ProviderConfig(
            api_key="k",
            base_url=FIREWORKS_BASE_URL,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "thinking" not in body


def test_build_request_body_request_disable_blocks_thinking(fireworks_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"enabled": False},
        }
    )
    body = fireworks_provider._build_request_body(request)
    assert "thinking" not in body


def test_build_request_body_merges_safe_extra_body(fireworks_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"custom_param": "value"},
        }
    )
    body = fireworks_provider._build_request_body(request)
    assert body["custom_param"] == "value"


def test_build_request_body_rejects_reserved_extra_body_keys(fireworks_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"temperature": 0.1},
        }
    )
    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        fireworks_provider._build_request_body(request)


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(fireworks_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    called: dict[str, str] = {}

    async def fake_send(request, *args, **kwargs):
        called["path"] = request.url.path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_closed = False
        mock_resp.raise_for_status = lambda: None

        async def aiter():
            if False:  # pragma: no cover
                yield ""

        mock_resp.aiter_lines = aiter
        mock_resp.aclose = AsyncMock()
        return mock_resp

    fireworks_provider._client.send = fake_send
    _ = [x async for x in fireworks_provider.stream_response(request, request_id="r1")]

    assert called["path"].endswith("/messages")


@pytest.mark.asyncio
async def test_cleanup_aclose(fireworks_provider):
    fireworks_provider._client = AsyncMock()

    await fireworks_provider.cleanup()

    fireworks_provider._client.aclose.assert_awaited_once()
