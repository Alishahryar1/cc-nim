"""Tests for Cerebras Inference (OpenAI-compatible) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.cerebras import CEREBRAS_DEFAULT_BASE, CerebrasProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter


def make_request(**overrides):
    return make_messages_request("llama3.1-8b", **overrides)


@pytest.fixture
def cerebras_config():
    return ProviderConfig(
        api_key="test_cerebras_key",
        base_url=CEREBRAS_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture
def cerebras_provider(cerebras_config):
    return CerebrasProvider(cerebras_config, rate_limiter=passthrough_rate_limiter())


def test_init(cerebras_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.transports.openai_chat.transport.AsyncOpenAI"
    ) as mock_openai:
        provider = CerebrasProvider(
            cerebras_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._api_key == "test_cerebras_key"
        assert provider._base_url == CEREBRAS_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert CEREBRAS_DEFAULT_BASE == "https://api.cerebras.ai/v1"


def test_build_request_body_basic(cerebras_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = make_request()
    body = cerebras_provider._build_request_body(req)

    assert body["model"] == "llama3.1-8b"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    provider = CerebrasProvider(
        ProviderConfig(
            api_key="test_cerebras_key",
            base_url=CEREBRAS_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_request()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_remaps_max_tokens_preserves_message_name(cerebras_provider):
    """Cerebras does not strip message ``name``; ``max_tokens`` maps to completion field."""
    with patch(
        "free_claude_code.providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
        }
        req = make_request()
        body = cerebras_provider._build_request_body(req)

    assert body["messages"][0].get("name") == "alice"
    assert body.get("max_tokens") is None
    assert body["max_completion_tokens"] == 42


def test_build_request_body_prefers_existing_max_completion_tokens(cerebras_provider):
    with patch(
        "free_claude_code.providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 77,
            "max_tokens": 999,
        }
        body = cerebras_provider._build_request_body(make_request())

    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(cerebras_provider):
    req = make_request(extra_body={"clear_thinking": False})

    body = cerebras_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("clear_thinking") is False


@pytest.mark.asyncio
async def test_stream_response_text(cerebras_provider):
    """Text content deltas are emitted as text blocks."""
    req = make_request()

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
        cerebras_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in cerebras_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(cerebras_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = make_request()

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
        cerebras_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in cerebras_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(cerebras_provider):
    cerebras_provider._client = AsyncMock()

    await cerebras_provider.cleanup()

    cerebras_provider._client.close.assert_called_once()
