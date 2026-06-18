"""Tests for metadata-only provider transport logging by default."""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.constants import NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES
from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider
from providers.transports.anthropic_messages import stream as native_stream
from tests.provider_request_mocks import make_openai_compat_stream_request
from tests.providers.test_anthropic_messages import (
    FakeResponse,
    MockRequest,
    NativeProvider,
)


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test-key",
        base_url="https://custom.test/v1/",
        proxy="socks5://127.0.0.1:9999",
        rate_limit=10,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
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


@pytest.mark.asyncio
async def test_native_non_200_logs_exclude_body_text_by_default(
    caplog, provider_config
):
    provider = NativeProvider(provider_config)
    req = MockRequest()
    response = FakeResponse(status_code=500, text="SECRET_UPSTREAM_BODY")

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_UPSTREAM_BODY" not in messages
    assert "HTTP 500" in messages
    assert "body_preview_bytes=" not in messages


@pytest.mark.asyncio
async def test_native_non_200_logs_body_when_verbose(caplog, provider_config):
    provider_config.log_api_error_tracebacks = True
    provider = NativeProvider(provider_config)
    req = MockRequest()
    response = FakeResponse(status_code=500, text="SECRET_UPSTREAM_BODY")

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_UPSTREAM_BODY" in messages
    assert "truncated=False" in messages


@pytest.mark.asyncio
async def test_native_non_200_verbose_logs_only_capped_error_body(
    caplog, provider_config
):
    provider_config.log_api_error_tracebacks = True
    provider = NativeProvider(provider_config)
    req = MockRequest()
    tail = "SECRET_TAIL_NOT_LOGGED"
    huge = f"{'A' * (NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES + 50)}{tail}"
    response = FakeResponse(status_code=500, text=huge)

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_TAIL_NOT_LOGGED" not in messages
    assert "truncated=True" in messages
    assert f"body_preview_bytes={NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES}" in messages


@pytest.mark.asyncio
async def test_native_non_200_default_does_not_read_oversized_body(
    caplog, provider_config
):
    provider = NativeProvider(provider_config)
    req = MockRequest()
    huge = f"{'Z' * 500_000}LEAK_MARKER"
    response = FakeResponse(status_code=500, text=huge)

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "LEAK_MARKER" not in messages
    assert "ZZZ" not in messages
    assert "HTTP 500" in messages


@pytest.mark.asyncio
async def test_native_stream_failure_logs_exclude_exception_str_by_default(
    caplog, provider_config
):
    provider = NativeProvider(provider_config)
    req = MockRequest()
    response = FakeResponse(
        lines=[
            "event: ping",
            'data: {"type":"ping"}',
            "",
        ]
    )

    async def boom(_response):
        raise RuntimeError("SECRET_DETAIL")
        if False:
            yield ""

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        patch.object(native_stream, "iter_sse_events", boom),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_DETAIL" not in messages
    assert "exc_type=RuntimeError" in messages
    assert "http_status=None" in messages


@pytest.mark.asyncio
async def test_openai_compat_stream_failure_default_logs_exclude_exception_str(caplog):
    config = ProviderConfig(
        api_key="k",
        base_url="http://localhost:1/v1",
        log_api_error_tracebacks=False,
    )
    provider = NvidiaNimProvider(config, nim_settings=NimSettings())
    req = make_openai_compat_stream_request()

    @asynccontextmanager
    async def _noop_slot():
        yield

    with (
        patch.object(
            provider,
            "_create_stream",
            new_callable=AsyncMock,
            side_effect=RuntimeError("SECRET_OPENAI_COMPAT"),
        ),
        patch.object(
            provider._global_rate_limiter,
            "concurrency_slot",
            _noop_slot,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_OPENAI_COMPAT" not in messages
    assert "exc_type=RuntimeError" in messages


@pytest.mark.asyncio
async def test_openai_compat_stream_failure_respects_verbose_flag(caplog):
    config = ProviderConfig(
        api_key="k",
        base_url="http://localhost:1/v1",
        log_api_error_tracebacks=True,
    )
    provider = NvidiaNimProvider(config, nim_settings=NimSettings())
    req = make_openai_compat_stream_request()

    @asynccontextmanager
    async def _noop_slot():
        yield

    with (
        patch.object(
            provider,
            "_create_stream",
            new_callable=AsyncMock,
            side_effect=RuntimeError("SECRET_OPENAI_COMPAT"),
        ),
        patch.object(
            provider._global_rate_limiter,
            "concurrency_slot",
            _noop_slot,
        ),
        caplog.at_level(logging.ERROR),
    ):
        _ = [e async for e in provider.stream_response(req)]

    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_OPENAI_COMPAT" in messages
