"""OpenAI-compat transports: upstream 5xx uses the same execute_with_retry path as 429."""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.key_rotation import ApiKeyRotationMode
from providers.nvidia_nim import NvidiaNimProvider
from providers.rate_limit import GlobalRateLimiter
from tests.providers.test_nvidia_nim import MockRequest


def _internal_5xx(code: int) -> openai.InternalServerError:
    return openai.InternalServerError(
        "unavailable",
        response=Response(code, request=Request("POST", "http://x")),
        body={},
    )


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limited",
        response=Response(429, request=Request("POST", "http://x")),
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


@pytest.mark.asyncio
async def test_openai_chat_failover_mode_retries_429_with_next_key():
    GlobalRateLimiter.reset_instance()
    try:
        config = ProviderConfig(
            api_key="key-a,key-b",
            api_key_rotation_mode=ApiKeyRotationMode.FAILOVER_ON_LIMIT,
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
                delta=MagicMock(content="Recovered", reasoning_content=""),
                finish_reason="stop",
            )
        ]
        mock_chunk.usage = None

        async def mock_stream():
            yield mock_chunk

        first_client = MagicMock()
        first_create = AsyncMock(side_effect=_rate_limit_error())
        first_client.chat.completions.create = first_create
        second_client = MagicMock()
        second_create = AsyncMock(return_value=mock_stream())
        second_client.chat.completions.create = second_create

        with (
            patch.object(
                provider._client,
                "with_options",
                side_effect=[first_client, second_client],
            ) as mock_with_options,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            events = [e async for e in provider.stream_response(req)]

        assert first_create.await_count == 1
        assert second_create.await_count == 1
        assert [
            call.kwargs["api_key"] for call in mock_with_options.call_args_list
        ] == [
            "key-a",
            "key-b",
        ]
        assert any("Recovered" in event for event in events)
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_openai_chat_failover_mode_preserves_5xx_retry_budget():
    GlobalRateLimiter.reset_instance()
    try:
        config = ProviderConfig(
            api_key="key-a,key-b",
            api_key_rotation_mode=ApiKeyRotationMode.FAILOVER_ON_LIMIT,
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
                delta=MagicMock(content="Recovered", reasoning_content=""),
                finish_reason="stop",
            )
        ]
        mock_chunk.usage = None

        async def mock_stream():
            yield mock_chunk

        fake_client = MagicMock()
        fake_create = AsyncMock(
            side_effect=[
                _internal_5xx(500),
                _internal_5xx(500),
                _internal_5xx(500),
                mock_stream(),
            ]
        )
        fake_client.chat.completions.create = fake_create

        with (
            patch.object(provider._client, "with_options", return_value=fake_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            events = [e async for e in provider.stream_response(req)]

        assert fake_create.await_count == 4
        assert any("Recovered" in event for event in events)
    finally:
        GlobalRateLimiter.reset_instance()
