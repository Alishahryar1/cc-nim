"""Regression tests for authenticated requests in the shared Anthropic Messages transport."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig


class _FakeProvider(AnthropicMessagesTransport):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config,
            provider_name="TEST_NATIVE",
            default_base_url="https://example.test/v1",
        )


@pytest.mark.asyncio
async def test_request_headers_includes_bearer_auth() -> None:
    """When api_key is configured, Authorization must be sent."""
    config = ProviderConfig(
        api_key="sk-test-123",
        base_url="https://example.test/v1/",
    )
    provider = _FakeProvider(config)

    assert provider._request_headers().get("Authorization") == "Bearer sk-test-123"


@pytest.mark.asyncio
async def test_request_headers_omits_auth_when_key_is_empty() -> None:
    """Empty api_key must not add an Authorization header."""
    config = ProviderConfig(
        api_key="",
        base_url="https://example.test/v1/",
    )
    provider = _FakeProvider(config)

    assert "Authorization" not in provider._request_headers()


@pytest.mark.asyncio
async def test_stream_request_uses_auth_header() -> None:
    """The streaming path must pass headers containing the configured api_key."""
    config = ProviderConfig(
        api_key="sk-stream",
        base_url="https://example.test/v1/",
    )
    provider = _FakeProvider(config)
    fake_response = MagicMock()
    fake_response.is_closed = False
    request_obj = httpx.Request("POST", "https://example.test/v1/messages")

    with (
        patch.object(provider._client, "build_request", return_value=request_obj) as mock_build,
        patch.object(provider._client, "send", new_callable=AsyncMock, return_value=fake_response) as mock_send,
    ):
        await provider._send_stream_request({"model": "m"})

    assert mock_build.call_args.kwargs["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-stream",
    }
    mock_send.assert_awaited_once_with(request_obj, stream=True)
