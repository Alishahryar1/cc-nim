"""Tests for Groq (OpenAI-compatible) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.providers.base import ProviderConfig
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter, profiled_provider


def make_request(**overrides):
    return make_messages_request("llama-3.3-70b-versatile", **overrides)


@pytest.fixture
def groq_config():
    return ProviderConfig(
        api_key="test_groq_key",
        base_url=GROQ_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture
def groq_provider(groq_config):
    return profiled_provider(
        "groq", groq_config, rate_limiter=passthrough_rate_limiter()
    )


def test_init(groq_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "groq", groq_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._api_key == "test_groq_key"
        assert provider._base_url == GROQ_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert GROQ_DEFAULT_BASE == "https://api.groq.com/openai/v1"


def test_build_request_body_basic(groq_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = make_request()
    body = groq_provider._build_request_body(req)

    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    provider = profiled_provider(
        "groq",
        ProviderConfig(
            api_key="test_groq_key",
            base_url=GROQ_DEFAULT_BASE,
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


def test_build_request_body_sanitizes_and_remaps_via_mock_converter(groq_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "name": "bad", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [],
                    "name": "nope",
                    "content": "ok",
                },
            ],
            "logprobs": True,
            "logit_bias": {"1": -100},
            "top_logprobs": 2,
            "max_tokens": 42,
            "n": 4,
        }
        req = make_request()
        body = groq_provider._build_request_body(req)

    msgs = body["messages"]
    assert msgs[0].get("name") is None and msgs[1].get("name") is None
    for key in ("logprobs", "logit_bias", "top_logprobs"):
        assert key not in body
    assert body.get("max_tokens") is None
    assert body["max_completion_tokens"] == 42
    assert body["n"] == 1


def test_build_request_body_prefers_existing_max_completion_tokens(groq_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 77,
            "max_tokens": 999,
        }
        body = groq_provider._build_request_body(make_request())

    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(groq_provider):
    req = make_request(extra_body={"metadata": {"user": "u1"}})

    body = groq_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


@pytest.mark.asyncio
async def test_stream_response_text(groq_provider):
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
        groq_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in groq_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(groq_provider):
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
        groq_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in groq_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(groq_provider):
    groq_provider._client = AsyncMock()

    await groq_provider.cleanup()

    groq_provider._client.close.assert_called_once()
