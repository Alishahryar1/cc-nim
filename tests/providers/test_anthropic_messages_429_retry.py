"""Native Anthropic transport: HTTP 429 and upstream 5xx are retried inside execute_with_retry."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providers.base import ProviderConfig
from providers.rate_limit import GlobalRateLimiter
from tests.providers.test_anthropic_messages import (
    FakeResponse,
    MockRequest,
    NativeProvider,
)
from tests.stream_contract import assert_canonical_stream_error_envelope


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test-key",
        base_url="https://custom.test/v1/",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )


@pytest.mark.asyncio
async def test_native_stream_retries_on_http_429_then_streams(provider_config):
    """First response 429 (closed), second 200 streams; send is called twice."""
    GlobalRateLimiter.reset_instance()
    try:
        provider = NativeProvider(provider_config)
        req = MockRequest()
        request_obj = httpx.Request("POST", "https://custom.test/v1/messages")
        ok_lines = [
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        ok_response = FakeResponse(lines=ok_lines)
        too_many = FakeResponse(status_code=429, text="rate limited")

        send_calls = {"n": 0}

        async def send_side_effect(*_a, **_kw):
            send_calls["n"] += 1
            if send_calls["n"] == 1:
                return too_many
            return ok_response

        with (
            patch.object(provider._client, "build_request", return_value=request_obj),
            patch.object(
                provider._client,
                "send",
                new_callable=AsyncMock,
                side_effect=send_side_effect,
            ),
            patch(
                "asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            events = [e async for e in provider.stream_response(req)]

        assert send_calls["n"] == 2
        assert too_many.is_closed
        assert ok_response.is_closed
        assert events == [
            "event: message_start\n",
            'data: {"type":"message_start"}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ]
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@pytest.mark.asyncio
async def test_native_stream_retries_on_http_5xx_then_streams(
    provider_config, status_code
):
    """First response is retryable 5xx (closed); second 200 streams; send twice."""
    GlobalRateLimiter.reset_instance()
    try:
        provider = NativeProvider(provider_config)
        req = MockRequest()
        request_obj = httpx.Request("POST", "https://custom.test/v1/messages")
        ok_lines = [
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        ok_response = FakeResponse(lines=ok_lines)
        bad = FakeResponse(status_code=status_code, text="upstream error")

        send_calls = {"n": 0}

        async def send_side_effect(*_a, **_kw):
            send_calls["n"] += 1
            if send_calls["n"] == 1:
                return bad
            return ok_response

        with (
            patch.object(provider._client, "build_request", return_value=request_obj),
            patch.object(
                provider._client,
                "send",
                new_callable=AsyncMock,
                side_effect=send_side_effect,
            ),
            patch(
                "asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            events = [e async for e in provider.stream_response(req)]

        assert send_calls["n"] == 2
        assert bad.is_closed
        assert ok_response.is_closed
        assert events == [
            "event: message_start\n",
            'data: {"type":"message_start"}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ]
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.parametrize(
    ("status_code", "substr"),
    [
        (500, "Provider API request failed"),
        (502, "Provider is currently overloaded"),
        (503, "Provider is currently overloaded"),
        (504, "Provider is currently overloaded"),
    ],
)
@pytest.mark.asyncio
async def test_native_stream_5xx_retry_exhausted(provider_config, status_code, substr):
    """Repeated upstream 5xx exhausts execute_with_retry; user message matches mapping."""
    GlobalRateLimiter.reset_instance()
    try:

        @asynccontextmanager
        async def _slot():
            yield

        with patch(
            "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
        ) as mock_gl:
            instance = mock_gl.get_scoped_instance.return_value
            real = GlobalRateLimiter(
                rate_limit=100,
                rate_window=60,
                max_concurrency=5,
            )
            instance.wait_if_blocked = real.wait_if_blocked
            instance.execute_with_retry = real.execute_with_retry
            instance.set_blocked = real.set_blocked
            instance.concurrency_slot.side_effect = _slot

            provider = NativeProvider(provider_config)
            req = MockRequest()

            bad = FakeResponse(status_code=status_code, text="upstream error")

            with (
                patch.object(
                    provider._client, "build_request", return_value=MagicMock()
                ),
                patch.object(
                    provider._client,
                    "send",
                    new_callable=AsyncMock,
                    return_value=bad,
                ) as mock_send,
                patch("asyncio.sleep", new_callable=AsyncMock),
            ):
                events = [e async for e in provider.stream_response(req)]

            assert mock_send.await_count == 5
            assert bad.is_closed
            assert_canonical_stream_error_envelope(
                events,
                user_message_substr=substr,
            )
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_non_retryable_4xx_http_error_not_retried(provider_config):
    """HTTP 400 from upstream is not retried; single send (passthrough limiter)."""
    GlobalRateLimiter.reset_instance()
    try:

        @asynccontextmanager
        async def _slot():
            yield

        with patch(
            "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
        ) as mock_gl:
            instance = mock_gl.get_scoped_instance.return_value

            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)

            instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
            instance.concurrency_slot.side_effect = _slot

            provider = NativeProvider(provider_config)
            req = MockRequest()
            err = FakeResponse(status_code=400, text="Bad Request")

            with (
                patch.object(
                    provider._client, "build_request", return_value=MagicMock()
                ),
                patch.object(
                    provider._client,
                    "send",
                    new_callable=AsyncMock,
                    return_value=err,
                ) as mock_send,
            ):
                events = [e async for e in provider.stream_response(req)]

            mock_send.assert_awaited_once()
            assert err.is_closed
            assert_canonical_stream_error_envelope(
                events, user_message_substr="Invalid request sent to provider"
            )
    finally:
        GlobalRateLimiter.reset_instance()
