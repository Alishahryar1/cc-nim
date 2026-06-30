"""Tests for Ollama native Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.anthropic.stream_contracts import parse_sse_text
from providers.base import ProviderConfig
from providers.ollama import OLLAMA_DEFAULT_BASE, OllamaProvider
from tests.stream_contract import assert_canonical_stream_error_envelope


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "llama3.1:8b"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.stream = True
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none=True):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.system,
            "stream": self.stream,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "extra_body": self.extra_body,
            "thinking": {"enabled": self.thinking.enabled} if self.thinking else None,
        }


@pytest.fixture
def ollama_config():
    return ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def ollama_provider(ollama_config):
    return OllamaProvider(ollama_config)


def test_init(ollama_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(ollama_config)
        assert provider._base_url == "http://localhost:11434"
        assert provider._provider_name == "OLLAMA"
        assert provider._api_key == "ollama"


def test_init_uses_default_base_url():
    """Test that provider uses default root URL when not configured."""
    config = ProviderConfig(api_key="ollama", base_url=None)
    with patch("httpx.AsyncClient"):
        provider = OllamaProvider(config)
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
        OllamaProvider(config)
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
        provider = OllamaProvider(config)
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
        provider = OllamaProvider(config)
        assert provider._api_key == "ollama"


@pytest.mark.asyncio
async def test_stream_response(ollama_provider):
    """Test streaming native Anthropic response."""
    req = MockRequest()

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
        ollama_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()

    body = provider._build_request_body(req)

    assert "thinking" not in body
    assert body["model"] == "llama3.1:8b"


def test_build_request_body_disabled_thinking_strips_assistant_thinking_blocks(
    ollama_config,
):
    """Prior assistant thinking/redacted blocks are removed when policy is off."""
    provider = OllamaProvider(
        ollama_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "hi"),
            MockMessage(
                "assistant",
                [
                    {"type": "thinking", "thinking": "t"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            ),
        ],
    )
    body = provider._build_request_body(req, thinking_enabled=False)
    assert body["messages"][1]["content"] == ""


@pytest.mark.asyncio
async def test_stream_error_status_code(ollama_provider):
    """Non-200 status code is yielded as an SSE API error."""
    req = MockRequest()
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
    ):
        events = [
            event
            async for event in ollama_provider.stream_response(req, request_id="REQ")
        ]

    assert_canonical_stream_error_envelope(
        events, user_message_substr="Provider API request failed"
    )
    assert "REQ" in "".join(events)


@pytest.mark.asyncio
async def test_cleanup(ollama_provider):
    """Test that cleanup closes the client."""
    ollama_provider._client.aclose = AsyncMock()

    await ollama_provider.cleanup()

    ollama_provider._client.aclose.assert_called_once()
