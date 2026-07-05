"""Tests for Cohere Compatibility API provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.cohere import COHERE_DEFAULT_BASE, CohereProvider
from providers.exceptions import InvalidRequestError


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "command-a-plus-05-2026"
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
def cohere_config():
    return ProviderConfig(
        api_key="test_cohere_key",
        base_url=COHERE_DEFAULT_BASE,
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
def cohere_provider(cohere_config):
    return CohereProvider(cohere_config)


def test_default_base_url_constant():
    assert COHERE_DEFAULT_BASE == "https://api.cohere.ai/compatibility/v1"


def test_init_uses_default_base_url_and_api_key(cohere_config):
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = CohereProvider(cohere_config)

    assert provider._api_key == "test_cohere_key"
    assert provider._base_url == COHERE_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(cohere_config):
    config = cohere_config.model_copy(update={"base_url": f"{COHERE_DEFAULT_BASE}/"})

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = CohereProvider(config)

    assert provider._base_url == COHERE_DEFAULT_BASE


def test_build_request_body_sanitizes_documented_unsupported_fields(cohere_provider):
    with patch(
        "providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "command-a-plus-05-2026",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
            "store": True,
            "metadata": {"trace": "abc"},
            "logit_bias": {"1": -100},
            "top_logprobs": 2,
            "n": 4,
            "modalities": ["text"],
            "prediction": {"type": "content", "content": "x"},
            "audio": {"voice": "alloy"},
            "service_tier": "auto",
            "parallel_tool_calls": True,
        }

        body = cohere_provider._build_request_body(MockRequest())

    assert body["messages"][0].get("name") is None
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body
    for key in (
        "audio",
        "logit_bias",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "service_tier",
        "store",
        "top_logprobs",
    ):
        assert key not in body


def test_build_request_body_maps_thinking_enabled_to_reasoning_high(cohere_provider):
    body = cohere_provider._build_request_body(MockRequest())

    assert body["reasoning_effort"] == "high"


def test_build_request_body_preserves_replayed_reasoning_content(cohere_provider):
    with patch(
        "providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "command-a-plus-05-2026",
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "hidden chain",
                }
            ],
        }

        body = cohere_provider._build_request_body(MockRequest())

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "hidden chain",
        }
    ]
    assert body["reasoning_effort"] == "high"


def test_build_request_body_maps_thinking_disabled_to_reasoning_none():
    provider = CohereProvider(
        ProviderConfig(
            api_key="test_cohere_key",
            base_url=COHERE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )

    body = provider._build_request_body(MockRequest())

    assert body["reasoning_effort"] == "none"


def test_build_request_body_promotes_allowed_extra_body(cohere_provider):
    req = MockRequest(
        extra_body={
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "response_format": {"type": "json_object"},
            "seed": 123,
        }
    )

    body = cohere_provider._build_request_body(req)

    assert body["frequency_penalty"] == 0.1
    assert body["presence_penalty"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 123
    assert "extra_body" not in body


def test_build_request_body_rejects_unsupported_extra_body(cohere_provider):
    req = MockRequest(extra_body={"documents": [{"text": "x"}]})

    with pytest.raises(InvalidRequestError, match="Unsupported"):
        cohere_provider._build_request_body(req)


@pytest.mark.asyncio
async def test_stream_response_text(cohere_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Cohere",
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
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_response(MockRequest())
        ]

    assert any(
        '"text_delta"' in event and "Hello from Cohere" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_tool_call(cohere_provider):
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function = MagicMock()
    mock_tc.function.name = "Read"
    mock_tc.function.arguments = '{"file_path":"a.py"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=[mock_tc]),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_response(MockRequest())
        ]

    assert any(
        '"content_block_start"' in event and '"tool_use"' in event for event in events
    )
    assert any(
        '"input_json_delta"' in event and "file_path" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(cohere_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via Cohere",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_response(MockRequest())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via Cohere" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(cohere_provider):
    cohere_provider._client = AsyncMock()

    await cohere_provider.cleanup()

    cohere_provider._client.close.assert_called_once()
