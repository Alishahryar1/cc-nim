"""Tests for Mistral Codestral provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.codestral import CODESTRAL_DEFAULT_BASE, CodestralProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "devstral-small-latest"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def codestral_config():
    return ProviderConfig(
        api_key="test_codestral_key",
        base_url=CODESTRAL_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def codestral_provider(codestral_config):
    return CodestralProvider(codestral_config)


def test_init(codestral_config):
    """Test provider initialization."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = CodestralProvider(codestral_config)
        assert provider._api_key == "test_codestral_key"
        assert provider._base_url == CODESTRAL_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url():
    assert CODESTRAL_DEFAULT_BASE == "https://codestral.mistral.ai/v1"


def test_build_request_body_basic(codestral_provider):
    """Basic request body conversion works for Codestral."""
    req = MockRequest()
    body = codestral_provider._build_request_body(req)

    assert body["model"] == "devstral-small-latest"
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    """Global disable disables reasoning replay in the converter."""
    provider = CodestralProvider(
        ProviderConfig(
            api_key="test_codestral_key",
            base_url=CODESTRAL_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


@pytest.mark.asyncio
async def test_stream_response_text(codestral_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        codestral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in codestral_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(codestral_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        codestral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in codestral_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(codestral_provider):
    """cleanup closes the OpenAI client."""
    codestral_provider._client = AsyncMock()

    await codestral_provider.cleanup()

    codestral_provider._client.close.assert_called_once()
