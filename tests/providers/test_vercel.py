"""Tests for Vercel AI Gateway provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.vercel import VERCEL_AI_GATEWAY_DEFAULT_BASE, VercelProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "openai/gpt-5.5"
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
def vercel_config():
    return ProviderConfig(
        api_key="test_vercel_key",
        base_url=VERCEL_AI_GATEWAY_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
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
def vercel_provider(vercel_config):
    return VercelProvider(vercel_config)


def test_default_base_url_constant():
    assert VERCEL_AI_GATEWAY_DEFAULT_BASE == "https://ai-gateway.vercel.sh/v1"


def test_init_uses_default_base_url_and_api_key(vercel_config):
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = VercelProvider(vercel_config)

    assert provider._api_key == "test_vercel_key"
    assert provider._base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(vercel_config):
    config = vercel_config.model_copy(
        update={"base_url": f"{VERCEL_AI_GATEWAY_DEFAULT_BASE}/"}
    )

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = VercelProvider(config)

    assert provider._base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE


def test_build_request_body_keeps_max_tokens(vercel_provider):
    with patch(
        "providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
        }

        body = vercel_provider._build_request_body(MockRequest())

    assert body["messages"][0].get("name") == "alice"
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(vercel_provider):
    req = MockRequest(extra_body={"providerOptions": {"openai": {"reasoning": "low"}}})

    body = vercel_provider._build_request_body(req)

    assert body["extra_body"] == {"providerOptions": {"openai": {"reasoning": "low"}}}


@pytest.mark.asyncio
async def test_stream_response_text(vercel_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Vercel",
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
        vercel_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in vercel_provider.stream_response(MockRequest())
        ]

    assert any(
        '"text_delta"' in event and "Hello from Vercel" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(vercel_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via gateway",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        vercel_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in vercel_provider.stream_response(MockRequest())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via gateway" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(vercel_provider):
    vercel_provider._client = AsyncMock()

    await vercel_provider.cleanup()

    vercel_provider._client.close.assert_called_once()
