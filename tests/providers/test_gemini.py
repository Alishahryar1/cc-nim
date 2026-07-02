"""Tests for Google AI Studio Gemini (OpenAI-compatible) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.gemini import GEMINI_DEFAULT_BASE, GeminiProvider
from providers.gemini.quirks import GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "models/gemini-3.1-flash-lite"
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
def gemini_provider(gemini_config):
    return GeminiProvider(gemini_config)


def test_init(gemini_config):
    """Test provider initialization."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = GeminiProvider(gemini_config)
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
    req = MockRequest()
    body = gemini_provider._build_request_body(req)

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
    req = MockRequest()

    body = gemini_provider._build_request_body(req)
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


def test_build_request_body_global_disable_sets_reasoning_none():
    """When thinking is off, Gemini uses reasoning_effort none (Gemini 2.5 convention)."""
    provider = GeminiProvider(
        ProviderConfig(
            api_key="test_gemini_key",
            base_url=GEMINI_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    assert body["reasoning_effort"] == "none"
    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_preserves_caller_extra_body(gemini_provider):
    req = MockRequest(extra_body={"metadata": {"user": "u1"}})

    body = gemini_provider._build_request_body(req)

    assert "reasoning_effort" not in body
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}
    literal_extra_body = eb.get("extra_body")
    assert isinstance(literal_extra_body, dict)
    google = literal_extra_body.get("google")
    assert isinstance(google, dict)


def test_build_request_body_merges_caller_nested_google(gemini_provider):
    req = MockRequest(
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

    body = gemini_provider._build_request_body(req)

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
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "Find files"),
            MockMessage(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                        "extra_content": {
                            "google": {"thought_signature": "sig-from-client"}
                        },
                    }
                ],
            ),
            MockMessage(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    }
                ],
            ),
        ],
    )

    body = gemini_provider._build_request_body(req)

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-client"}
    }


def test_build_request_body_uses_cached_tool_call_signature(gemini_provider):
    gemini_provider._record_tool_call_extra_content(
        "function-call-1", {"google": {"thought_signature": "sig-from-cache"}}
    )
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "Find files"),
            MockMessage(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    }
                ],
            ),
            MockMessage(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    }
                ],
            ),
        ],
    )

    body = gemini_provider._build_request_body(req)

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-cache"}
    }


def test_build_request_body_adds_gemini3_current_turn_fallback_signature(
    gemini_provider,
):
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "Find files"),
            MockMessage(
                "assistant",
                [
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
            ),
            MockMessage(
                "user",
                [
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
            ),
        ],
    )

    body = gemini_provider._build_request_body(req)

    tool_calls = body["messages"][1]["tool_calls"]
    assert tool_calls[0]["extra_content"] == {
        "google": {"thought_signature": GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR}
    }
    assert "extra_content" not in tool_calls[1]


@pytest.mark.asyncio
async def test_stream_response_text(gemini_provider):
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
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in gemini_provider.stream_response(req)]

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
    req = MockRequest()

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
