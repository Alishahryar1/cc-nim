"""Tests for Llama.cpp native Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.exceptions import ProviderError
from free_claude_code.providers.llamacpp import LlamaCppProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter

LLAMACPP_MODEL = "llamacpp-community/qwen2.5-7b-instruct"


async def _minimal_native_lines():
    yield "event: message_start"
    yield 'data: {"type":"message_start"}'
    yield ""
    yield "event: message_stop"
    yield 'data: {"type":"message_stop"}'
    yield ""


@pytest.fixture
def llamacpp_config():
    return ProviderConfig(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def llamacpp_provider(llamacpp_config):
    return LlamaCppProvider(llamacpp_config, rate_limiter=passthrough_rate_limiter())


def test_init(llamacpp_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient"):
        provider = LlamaCppProvider(
            llamacpp_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._base_url == "http://localhost:8080/v1"
        assert provider._provider_name == "LLAMACPP"


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("httpx.AsyncClient") as mock_client:
        LlamaCppProvider(config, rate_limiter=passthrough_rate_limiter())
        call_kwargs = mock_client.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("httpx.AsyncClient"):
        provider = LlamaCppProvider(config, rate_limiter=passthrough_rate_limiter())
        assert provider._base_url == "http://localhost:8080/v1"


@pytest.mark.asyncio
async def test_stream_response_omits_thinking_when_globally_disabled(llamacpp_config):
    provider = LlamaCppProvider(
        llamacpp_config.model_copy(update={"enable_thinking": False}),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request(LLAMACPP_MODEL)

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_response.aiter_lines = _minimal_native_lines

    with (
        patch.object(provider._client, "build_request") as mock_build,
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        [e async for e in provider.stream_response(req)]

    _, kwargs = mock_build.call_args
    assert "thinking" not in kwargs["json"]


@pytest.mark.asyncio
async def test_stream_response(llamacpp_provider):
    """Test streaming native Anthropic response."""
    req = make_messages_request(LLAMACPP_MODEL)

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_lines():
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{}}'
        yield ""
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello World"}}'
        yield ""
        yield "event: message_stop"
        yield 'data: {"type":"message_stop"}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    with (
        patch.object(
            llamacpp_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            llamacpp_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        events = [e async for e in llamacpp_provider.stream_response(req)]

        # Verify request construction
        mock_build.assert_called_once()
        args, kwargs = mock_build.call_args
        assert args[0] == "POST"
        assert args[1] == "/messages"
        assert kwargs["json"]["model"] == "llamacpp-community/qwen2.5-7b-instruct"
        # Verify internal fields are popped
        assert "extra_body" not in kwargs["json"]
        assert kwargs["json"]["max_tokens"] == 100

        # Verify internal ThinkingConfig is mapped to Anthropic API format
        assert kwargs["json"]["thinking"] == {"type": "enabled"}

        assert [event.event for event in parse_sse_text("".join(events))] == [
            "message_start",
            "content_block_delta",
            "message_stop",
        ]
        assert "Hello World" in "".join(events)


@pytest.mark.asyncio
async def test_stream_response_adds_max_tokens_if_missing(llamacpp_provider):
    """Fallback max_tokens to 81920 if not present."""
    req = make_messages_request(LLAMACPP_MODEL, max_tokens=None)
    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_response.aiter_lines = _minimal_native_lines

    with (
        patch.object(llamacpp_provider._client, "build_request") as mock_build,
        patch.object(
            llamacpp_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        # Just run the generator to completion
        [e async for e in llamacpp_provider.stream_response(req)]

        _, kwargs = mock_build.call_args
        assert kwargs["json"]["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_stream_error_status_code(llamacpp_provider):
    """Pre-start non-200 status code raises for API-level non-200 handling."""
    req = make_messages_request(LLAMACPP_MODEL)

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response
        )
    )

    with (
        patch.object(
            llamacpp_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            llamacpp_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        with pytest.raises(ProviderError) as exc_info:
            [
                e
                async for e in llamacpp_provider.stream_response(
                    req, request_id="TEST_ID"
                )
            ]

        assert "Provider API request failed" in exc_info.value.message
        assert "TEST_ID" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_network_error(llamacpp_provider):
    """Pre-start network errors raise for API-level non-200 handling."""
    req = make_messages_request(LLAMACPP_MODEL)

    with (
        patch.object(
            llamacpp_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            llamacpp_provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ),
    ):
        with pytest.raises(ProviderError) as exc_info:
            [
                e
                async for e in llamacpp_provider.stream_response(
                    req, request_id="TEST_ID2"
                )
            ]

        assert "Could not connect to provider." in exc_info.value.message
        assert "Provider exception" in exc_info.value.message
        assert "Connection refused" in exc_info.value.message
        assert "TEST_ID2" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_error_405_mentions_upstream_provider(llamacpp_provider):
    req = make_messages_request(LLAMACPP_MODEL)

    mock_response = MagicMock()
    mock_response.status_code = 405
    mock_response.aread = AsyncMock(return_value=b"Method Not Allowed")
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Method Not Allowed", request=MagicMock(), response=mock_response
        )
    )

    with (
        patch.object(
            llamacpp_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            llamacpp_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        [e async for e in llamacpp_provider.stream_response(req, request_id="REQ405")]

    assert (
        "Upstream provider LLAMACPP rejected the request method or endpoint (HTTP 405)."
        in exc_info.value.message
    )
    assert "REQ405" in exc_info.value.message


def test_build_request_body_disabled_thinking_strips_native_thinking_history(
    llamacpp_config,
):
    """With thinking disabled, prior assistant thinking/redacted blocks are omitted."""
    config = llamacpp_config.model_copy(update={"enable_thinking": False})
    provider = LlamaCppProvider(config, rate_limiter=passthrough_rate_limiter())
    req = make_messages_request(
        LLAMACPP_MODEL,
        system=None,
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "p"},
                    {"type": "redacted_thinking", "data": "ZGF0YQ=="},
                ],
            },
        ],
    )
    body = provider._build_request_body(req, thinking_enabled=False)
    asst = body["messages"][1]
    assert asst["content"] == ""
    assert "thinking" not in str(body)
    assert "redacted_thinking" not in str(body)
