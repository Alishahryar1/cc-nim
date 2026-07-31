"""Tests for TokenRouter (OpenAI-compatible Chat Completions gateway)."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import TOKENROUTER_DEFAULT_BASE
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OPENAI_CHAT_PROFILES
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


def make_request(model="quimera-3-free", **overrides):
    return make_messages_request(model, **overrides)


@pytest.fixture
def tokenrouter_config():
    return ProviderConfig(
        api_key="test_tokenrouter_key",
        base_url=TOKENROUTER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def tokenrouter_provider(tokenrouter_config):
    return profiled_provider(
        "tokenrouter", tokenrouter_config, admission=immediate_admission()
    )


def test_default_base_url_constant():
    assert TOKENROUTER_DEFAULT_BASE == "https://api.tokenrouter.com/v1"


def test_profile_in_openai_chat_profiles():
    """The tokenrouter profile must be wired into OPENAI_CHAT_PROFILES."""
    assert "tokenrouter" in OPENAI_CHAT_PROFILES
    profile = OPENAI_CHAT_PROFILES["tokenrouter"]
    assert profile.provider_name == "TOKENROUTER"
    assert profile.request_policy.max_tokens_field == "max_completion_tokens"
    assert profile.request_policy.include_extra_body is True


def test_init_uses_default_base_url_and_api_key(tokenrouter_config):
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "tokenrouter", tokenrouter_config, admission=immediate_admission()
        )

    assert provider._api_key == "test_tokenrouter_key"
    assert provider._base_url == TOKENROUTER_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(tokenrouter_config):
    config = replace(tokenrouter_config, base_url=f"{TOKENROUTER_DEFAULT_BASE}/")

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = profiled_provider(
            "tokenrouter", config, admission=immediate_admission()
        )

    assert provider._base_url == TOKENROUTER_DEFAULT_BASE


def test_build_request_body_basic(tokenrouter_provider):
    """Basic request body conversion attaches system message and uses max_completion_tokens."""
    req = make_request()
    body = tokenrouter_provider._build_request_body(req, reasoning=reasoning_for(req))

    assert body["model"] == "quimera-3-free"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(tokenrouter_provider):
    req = make_request(extra_body={"metadata": {"user": "u1"}})

    body = tokenrouter_provider._build_request_body(req, reasoning=reasoning_for(req))

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


def test_build_request_body_rejects_reasoning_in_extra_body(tokenrouter_provider):
    from free_claude_code.application.errors import InvalidRequestError

    req = make_request(extra_body={"reasoning_effort": "high"})

    with pytest.raises(InvalidRequestError, match="reasoning"):
        tokenrouter_provider._build_request_body(req, reasoning=reasoning_for(req))


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(), "medium"),
    ),
)
def test_build_request_body_uses_only_documented_reasoning_efforts(
    tokenrouter_provider, reasoning, expected
):
    body = tokenrouter_provider._build_request_body(make_request(), reasoning=reasoning)

    assert body.get("reasoning_effort") == expected


@pytest.mark.asyncio
async def test_stream_response_text(tokenrouter_provider):
    """Text content deltas are emitted as text blocks."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from TokenRouter",
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
        tokenrouter_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in tokenrouter_provider.stream_response(make_request())
        ]

    assert any(
        '"text_delta"' in event and "Hello from TokenRouter" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_tool_call(tokenrouter_provider):
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
        tokenrouter_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in tokenrouter_provider.stream_response(make_request())
        ]

    assert any(
        '"content_block_start"' in event and '"tool_use"' in event for event in events
    )
    assert any(
        '"input_json_delta"' in event and "file_path" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(tokenrouter_provider):
    """reasoning_content deltas are emitted as thinking blocks (THINK_TAGS replay)."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via TokenRouter",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        tokenrouter_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in tokenrouter_provider.stream_response(make_request())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via TokenRouter" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(tokenrouter_provider):
    tokenrouter_provider._client = AsyncMock()

    await tokenrouter_provider.cleanup()

    tokenrouter_provider._client.close.assert_called_once()
