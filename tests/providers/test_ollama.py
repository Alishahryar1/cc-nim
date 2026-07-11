"""Tests for Ollama native Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.exceptions import ProviderError
from free_claude_code.providers.ollama import OLLAMA_DEFAULT_BASE, OllamaProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter

OLLAMA_MODEL = "llama3.1:8b"


@pytest.fixture
def ollama_config():
    return ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def ollama_provider(ollama_config):
    return OllamaProvider(ollama_config, rate_limiter=passthrough_rate_limiter())


def test_init(ollama_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(
            ollama_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._base_url == "http://localhost:11434"
        assert provider._provider_name == "OLLAMA"
        assert provider._api_key == "ollama"


def test_init_uses_default_base_url():
    """Test that provider uses default root URL when not configured."""
    config = ProviderConfig(api_key="ollama", base_url=None)
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(config, rate_limiter=passthrough_rate_limiter())
        assert provider._base_url == OLLAMA_DEFAULT_BASE


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("httpx.AsyncClient") as mock_client:
        OllamaProvider(config, rate_limiter=passthrough_rate_limiter())
        call_kwargs = mock_client.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(config, rate_limiter=passthrough_rate_limiter())
        assert provider._base_url == "http://localhost:11434"


def test_init_uses_default_api_key():
    """Test that provider uses default API key when not configured."""
    config = ProviderConfig(
        base_url="http://localhost:11434",
        api_key="",
        rate_limit=10,
        rate_window=60,
    )
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(config, rate_limiter=passthrough_rate_limiter())
        assert provider._api_key == "ollama"


@pytest.mark.asyncio
async def test_stream_response(ollama_provider):
    """Test streaming native Anthropic response."""
    req = make_messages_request(OLLAMA_MODEL)

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
            ollama_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            ollama_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        events = [event async for event in ollama_provider.stream_response(req)]

    mock_build.assert_called_once()
    args, kwargs = mock_build.call_args
    assert args[0] == "POST"
    assert args[1] == "/v1/messages"
    assert kwargs["json"]["model"] == "llama3.1:8b"
    assert kwargs["json"]["stream"] is True
    assert "extra_body" not in kwargs["json"]
    assert kwargs["json"]["thinking"] == {"type": "enabled"}
    assert [event.event for event in parse_sse_text("".join(events))] == [
        "message_start",
        "content_block_delta",
        "message_stop",
    ]
    assert "Hello World" in "".join(events)


@pytest.mark.asyncio
async def test_build_request_body_omits_thinking_when_disabled(ollama_config):
    """Global disable suppresses provider-side thinking."""
    provider = OllamaProvider(
        ollama_config.model_copy(update={"enable_thinking": False}),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request(OLLAMA_MODEL)

    body = provider._build_request_body(req)

    assert "thinking" not in body
    assert body["model"] == "llama3.1:8b"


def test_build_request_body_disabled_thinking_strips_assistant_thinking_blocks(
    ollama_config,
):
    """Prior assistant thinking/redacted blocks are removed when policy is off."""
    provider = OllamaProvider(
        ollama_config.model_copy(update={"enable_thinking": False}),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request(
        OLLAMA_MODEL,
        system=None,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            },
        ],
    )
    body = provider._build_request_body(req, thinking_enabled=False)
    assert body["messages"][1]["content"] == ""


@pytest.mark.asyncio
async def test_stream_error_status_code(ollama_provider):
    """Pre-start non-200 status code raises for API-level non-200 handling."""
    req = make_messages_request(OLLAMA_MODEL)
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
            ollama_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            ollama_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        [
            event
            async for event in ollama_provider.stream_response(req, request_id="REQ")
        ]

    assert "Provider API request failed" in exc_info.value.message
    assert "REQ" in exc_info.value.message


@pytest.mark.asyncio
async def test_cleanup(ollama_provider):
    """Test that cleanup closes the client."""
    ollama_provider._client.aclose = AsyncMock()

    await ollama_provider.cleanup()

    ollama_provider._client.aclose.assert_called_once()
