"""Tests for Ollama OpenAI-compatible chat completions provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    # Note: OpenAIChatTransport uses GlobalRateLimiter
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
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
    with patch("openai.AsyncOpenAI"):
        provider = OllamaProvider(ollama_config)
        assert provider._base_url == "http://localhost:11434"
        assert provider._provider_name == "OLLAMA"
        assert provider._api_key == "ollama"


def test_init_uses_default_base_url():
    """Test that provider uses default root URL when not configured."""
    config = ProviderConfig(api_key="ollama", base_url=None)
    with patch("openai.AsyncOpenAI"):
        provider = OllamaProvider(config)
        assert provider._base_url == OLLAMA_DEFAULT_BASE


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("openai.AsyncOpenAI"):
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
    with patch("openai.AsyncOpenAI"):
        provider = OllamaProvider(config)
        assert provider._api_key == "ollama"


@pytest.mark.asyncio
async def test_stream_response(ollama_provider):
    """Test streaming OpenAI chat completions response."""
    req = MockRequest()

    async def mock_iter_chunks():
        # OpenAI style chunks
        # Chunk 1: Role
        delta1 = MagicMock()
        delta1.role = "assistant"
        delta1.content = None
        delta1.reasoning_content = None
        delta1.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta1, finish_reason=None)], usage=None
        )

        # Chunk 2: Content
        delta2 = MagicMock()
        delta2.role = None
        delta2.content = "Hello"
        delta2.reasoning_content = None
        delta2.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta2, finish_reason=None)], usage=None
        )

        # Chunk 3: Content
        delta3 = MagicMock()
        delta3.role = None
        delta3.content = " World"
        delta3.reasoning_content = None
        delta3.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta3, finish_reason=None)], usage=None
        )

    # We need to mock the underlying OpenAI client's request execution
    with patch.object(
        ollama_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_iter_chunks(),
    ) as mock_create:
        events = [event async for event in ollama_provider.stream_response(req)]

    mock_create.assert_called_once()
    full_text = "".join(events)
    assert "message_start" in full_text
    assert "Hello" in full_text
    assert "World" in full_text
    assert "message_stop" in full_text


@pytest.mark.asyncio
async def test_stream_error_status_code(ollama_provider):
    """Exception during stream creation is yielded as an SSE API error."""
    req = MockRequest()

    with patch.object(
        ollama_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=Exception("Connection failed"),
    ):
        events = [
            event
            async for event in ollama_provider.stream_response(req, request_id="REQ")
        ]

    full_text = "".join(events)
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Connection failed"
    )
    assert "REQ" in full_text
    assert "Connection failed" in full_text


@pytest.mark.asyncio
async def test_cleanup(ollama_provider):
    """Test that cleanup closes the client."""
    ollama_provider._client.close = AsyncMock()

    await ollama_provider.cleanup()

    ollama_provider._client.close.assert_called_once()
