"""Tests for Google AI Studio Gemini (OpenAI-compatible) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.google_openai import (
    GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter, reasoning_for


def make_request(**overrides):
    return make_messages_request("models/gemini-3.1-flash-lite", **overrides)


def _simulate_openai_sdk_wire_json(body: dict) -> dict:
    wire = {key: value for key, value in body.items() if key != "extra_body"}
    sdk_extra = body.get("extra_body")
    if isinstance(sdk_extra, dict):
        wire.update(sdk_extra)
    return wire


@pytest.fixture
def gemini_config():
    return ProviderConfig(
        api_key="test_gemini_key",
        base_url=GEMINI_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def gemini_provider(gemini_config):
    return GeminiProvider(gemini_config, rate_limiter=passthrough_rate_limiter())


def test_init(gemini_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        provider = GeminiProvider(
            gemini_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._api_key == "test_gemini_key"
        assert (
            provider._base_url
            == "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert GEMINI_DEFAULT_BASE == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def test_build_request_body_basic(gemini_provider):
    """Basic body conversion attaches Gemini thinking fields when thinking is on."""
    req = make_request()
    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    assert body["model"] == "models/gemini-3.1-flash-lite"
    assert body["messages"][0]["role"] == "system"
    assert "reasoning_effort" not in body
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    literal_extra_body = eb.get("extra_body")
    assert isinstance(literal_extra_body, dict)
    gc = literal_extra_body.get("google")
    assert isinstance(gc, dict)
    tc = gc.get("thinking_config")
    assert isinstance(tc, dict)
    assert tc.get("include_thoughts") is True
    assert "google" not in eb


def test_build_request_body_sdk_wire_json_has_literal_extra_body(gemini_provider):
    """Regression for issue #542: SDK merge must not send top-level google."""
    req = make_request()

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))
    wire_json = _simulate_openai_sdk_wire_json(body)

    assert "reasoning_effort" not in wire_json
    assert "google" not in wire_json
    literal_extra_body = wire_json.get("extra_body")
    assert isinstance(literal_extra_body, dict)
    google = literal_extra_body.get("google")
    assert isinstance(google, dict)
    thinking_config = google.get("thinking_config")
    assert isinstance(thinking_config, dict)
    assert thinking_config.get("include_thoughts") is True


def test_build_request_body_reasoning_off_sets_reasoning_none():
    """When thinking is off, Gemini uses reasoning_effort none (Gemini 2.5 convention)."""
    provider = GeminiProvider(
        ProviderConfig(
            api_key="test_gemini_key",
            base_url=GEMINI_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_request(thinking={"type": "disabled"})
    body = provider._build_request_body(req, reasoning=reasoning_for(req))

    assert body["reasoning_effort"] == "none"
    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_preserves_caller_extra_body(gemini_provider):
    req = make_request(extra_body={"metadata": {"user": "u1"}})

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    assert "reasoning_effort" not in body
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}
    literal_extra_body = eb.get("extra_body")
    assert isinstance(literal_extra_body, dict)
    google = literal_extra_body.get("google")
    assert isinstance(google, dict)


def test_build_request_body_merges_caller_nested_google(gemini_provider):
    req = make_request(
        extra_body={
            "metadata": {"user": "u1"},
            "extra_body": {
                "google": {
                    "thinking_config": {"budget_tokens": 128},
                    "cached_content": "cachedContents/example",
                }
            },
        }
    )

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    assert "reasoning_effort" not in body
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}
    literal_extra_body = eb.get("extra_body")
    assert isinstance(literal_extra_body, dict)
    google = literal_extra_body.get("google")
    assert isinstance(google, dict)
    assert google.get("cached_content") == "cachedContents/example"
    thinking_config = google.get("thinking_config")
    assert isinstance(thinking_config, dict)
    assert thinking_config.get("budget_tokens") == 128
    assert thinking_config.get("include_thoughts") is True


def test_build_request_body_preserves_tool_call_extra_content(gemini_provider):
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                        "extra_content": {
                            "google": {"thought_signature": "sig-from-client"}
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-client"}
    }


def test_build_request_body_uses_cached_tool_call_signature(gemini_provider):
    gemini_provider._record_tool_call_extra_content(
        "function-call-1", {"google": {"thought_signature": "sig-from-cache"}}
    )
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-cache"}
    }


def test_build_request_body_adds_current_turn_fallback_signature(
    gemini_provider,
):
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "function-call-2",
                        "name": "Read",
                        "input": {"file_path": "a.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-2",
                        "content": "contents",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._build_request_body(req, reasoning=reasoning_for(req))

    tool_calls = body["messages"][1]["tool_calls"]
    assert tool_calls[0]["extra_content"] == {
        "google": {"thought_signature": GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR}
    }
    assert "extra_content" not in tool_calls[1]


@pytest.mark.asyncio
async def test_stream_response_text(gemini_provider):
    req = make_request(thinking={"type": "enabled"})

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
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in gemini_provider.stream_response(
                req, reasoning=reasoning_for(req)
            )
        ]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )
        kwargs = mock_create.call_args.kwargs
        assert "reasoning_effort" not in kwargs
        extra_body = kwargs.get("extra_body")
        assert isinstance(extra_body, dict)
        literal_extra_body = extra_body.get("extra_body")
        assert isinstance(literal_extra_body, dict)
        google = literal_extra_body.get("google")
        assert isinstance(google, dict)
        thinking_config = google.get("thinking_config")
        assert isinstance(thinking_config, dict)
        assert thinking_config.get("include_thoughts") is True


@pytest.mark.asyncio
async def test_stream_response_preserves_tool_call_extra_content(gemini_provider):
    req = make_request()

    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "function-call-1"
    mock_tc.extra_content = {"google": {"thought_signature": "sig-stream"}}
    mock_tc.function = MagicMock()
    mock_tc.function.name = "Glob"
    mock_tc.function.arguments = '{"pattern":"*.py"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content=None,
                tool_calls=[mock_tc],
            ),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in gemini_provider.stream_response(req)]

    tool_starts = [
        event
        for event in events
        if '"content_block_start"' in event and '"tool_use"' in event
    ]
    assert any(
        '"extra_content"' in event and "sig-stream" in event for event in tool_starts
    )
    assert gemini_provider._tool_call_extra_content_by_id["function-call-1"] == {
        "google": {"thought_signature": "sig-stream"}
    }


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(gemini_provider):
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
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in gemini_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(gemini_provider):
    gemini_provider._client = AsyncMock()

    await gemini_provider.cleanup()

    gemini_provider._client.close.assert_called_once()
