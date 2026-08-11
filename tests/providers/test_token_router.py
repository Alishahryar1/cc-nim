"""Tests for TokenRouter (OpenAI-compatible) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import TOKENROUTER_DEFAULT_BASE
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.token_router import TokenRouterProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "moonshotai/kimi-k3-free"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.model_extra = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        self.extra_body = kwargs.get("extra_body")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def tr_config():
    return ProviderConfig(
        api_key="test_tr_key",
        base_url=TOKENROUTER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "free_claude_code.providers.transports.openai_chat.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def tr_provider(tr_config):
    return TokenRouterProvider(tr_config)


def test_init(tr_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.transports.openai_chat.transport.AsyncOpenAI"
    ) as mock_openai:
        provider = TokenRouterProvider(tr_config)
        assert provider._api_key == "test_tr_key"
        assert provider._base_url == TOKENROUTER_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert TOKENROUTER_DEFAULT_BASE == "https://api.tokenrouter.com/v1"


def test_build_request_body_basic(tr_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = MockRequest()
    body = tr_provider._build_request_body(req)

    assert body["model"] == "moonshotai/kimi-k3-free"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


@pytest.mark.asyncio
async def test_stream_response_text(tr_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from TokenRouter!",
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
        tr_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in tr_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello from TokenRouter!" in event
            for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(tr_provider):
    tr_provider._client = AsyncMock()

    await tr_provider.cleanup()

    tr_provider._client.close.assert_called_once()
