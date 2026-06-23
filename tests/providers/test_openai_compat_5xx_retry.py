"""OpenAI-compat transports: upstream 5xx uses the same execute_with_retry path as 429."""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from core.anthropic import VisionCapabilityProtocol
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider
from providers.rate_limit import GlobalRateLimiter
from tests.providers.test_nvidia_nim import MockRequest


def _internal_5xx(code: int) -> openai.InternalServerError:
    return openai.InternalServerError(
        "unavailable",
        response=Response(code, request=Request("POST", "http://x")),
        body={},
    )


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@pytest.mark.asyncio
async def test_nim_stream_retries_on_openai_5xx_then_streams(status_code):
    GlobalRateLimiter.reset_instance()
    try:
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://test.api.nvidia.com/v1",
            rate_limit=100,
            rate_window=60,
            http_read_timeout=600.0,
            http_write_timeout=15.0,
            http_connect_timeout=5.0,
        )
        provider = NvidiaNimProvider(config, nim_settings=NimSettings())
        req = MockRequest()

        mock_chunk = MagicMock()
        mock_chunk.choices = [
            MagicMock(
                delta=MagicMock(content="Hi", reasoning_content=""),
                finish_reason="stop",
            )
        ]
        mock_chunk.usage = None

        async def mock_stream():
            yield mock_chunk

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_create.side_effect = [_internal_5xx(status_code), mock_stream()]
            events = [e async for e in provider.stream_response(req)]

        assert mock_create.await_count == 2
        assert any("Hi" in e for e in events)
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.parametrize(
    ("status_code", "expect_substr"),
    [
        (500, "provider api request failed"),
        (502, "temporarily unavailable"),
        (503, "temporarily unavailable"),
        (504, "temporarily unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_nim_stream_openai_5xx_exhausted_emits_user_message(
    status_code,
    expect_substr,
):
    GlobalRateLimiter.reset_instance()
    try:
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://test.api.nvidia.com/v1",
            rate_limit=100,
            rate_window=60,
            http_read_timeout=600.0,
            http_write_timeout=15.0,
            http_connect_timeout=5.0,
        )
        provider = NvidiaNimProvider(config, nim_settings=NimSettings())
        req = MockRequest()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_create.side_effect = _internal_5xx(status_code)
            events = [e async for e in provider.stream_response(req)]

        assert mock_create.await_count == 5
        blob = "".join(events)
        assert expect_substr in blob.lower()
    finally:
        GlobalRateLimiter.reset_instance()


def test_default_vision_capability_is_no_vision():
    """Base OpenAIChatTransport._vision_capability() returns NO_VISION by default."""
    # Use a minimal concrete subclass since OpenAIChatTransport is abstract
    from providers.base import ProviderConfig
    from providers.transports.openai_chat import OpenAIChatTransport

    class _StubTransport(OpenAIChatTransport):
        def _build_request_body(self, request, thinking_enabled=None):
            return {}

    config = ProviderConfig(
        base_url="http://localhost",
        api_key="test",
    )
    transport = _StubTransport(
        config,
        provider_name="stub",
        base_url="http://localhost",
        api_key="test",
    )
    capability = transport._vision_capability()
    assert isinstance(capability, VisionCapabilityProtocol)
    assert capability.enabled is False
