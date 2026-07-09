"""OpenAI-compat transports: upstream 5xx uses the same execute_with_retry path as 429."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.exceptions import ProviderError
from providers.nvidia_nim import NvidiaNimProvider
from providers.rate_limit import GlobalRateLimiter
from tests.providers.test_nvidia_nim import MockRequest


def _internal_5xx(code: int) -> openai.InternalServerError:
    return openai.InternalServerError(
        "unavailable",
        response=Response(code, request=Request("POST", "http://x")),
        body={},
    )


def _connection_error(message: str = "connect failed") -> openai.APIConnectionError:
    error = openai.APIConnectionError(
        request=Request("POST", "https://test.api.nvidia.com/v1/chat/completions")
    )
    error.__cause__ = httpx.ConnectError(message)
    return error


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


@pytest.mark.asyncio
async def test_nim_stream_retries_on_pre_stream_connection_error_then_streams():
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
                delta=MagicMock(content="Recovered", reasoning_content=""),
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
            mock_create.side_effect = [_connection_error(), mock_stream()]
            events = [e async for e in provider.stream_response(req)]

        assert mock_create.await_count == 2
        assert any("Recovered" in e for e in events)
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_nim_stream_connection_error_exhausted_emits_cause_chain():
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
        error = _connection_error("upstream disconnected")

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=error,
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("providers.transports.openai_chat.stream.trace_event") as trace,
            pytest.raises(ProviderError) as exc_info,
        ):
            [e async for e in provider.stream_response(req, request_id="req_conn")]

        assert mock_create.await_count == 5
        error_traces = [
            call.kwargs
            for call in trace.call_args_list
            if call.kwargs.get("event") == "provider.response.error"
        ]
        assert error_traces[-1]["request_id"] == "req_conn"
        assert error_traces[-1]["exc_type"] == "APIConnectionError"
        assert (
            "Caused by:\nConnectError: upstream disconnected" in exc_info.value.message
        )
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
            with pytest.raises(ProviderError) as exc_info:
                [e async for e in provider.stream_response(req)]

        assert mock_create.await_count == 5
        assert expect_substr in exc_info.value.message.lower()
    finally:
        GlobalRateLimiter.reset_instance()
