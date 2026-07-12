"""Tests for Mistral La Plateforme provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from free_claude_code.config.provider_catalog import MISTRAL_DEFAULT_BASE
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.mistral import MistralProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter


def make_request(**overrides):
    return make_messages_request("devstral-small-latest", **overrides)


@pytest.fixture
def mistral_config():
    return ProviderConfig(
        api_key="test_mistral_key",
        base_url=MISTRAL_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture
def mistral_provider(mistral_config):
    return MistralProvider(mistral_config, rate_limiter=passthrough_rate_limiter())


def test_init(mistral_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        provider = MistralProvider(
            mistral_config, rate_limiter=passthrough_rate_limiter()
        )
        assert provider._api_key == "test_mistral_key"
        assert provider._base_url == MISTRAL_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url():
    assert MISTRAL_DEFAULT_BASE == "https://api.mistral.ai/v1"


def test_build_request_body_basic(mistral_provider):
    """Basic request body conversion works for Mistral."""
    req = make_request()
    body = mistral_provider._build_request_body(req)

    assert body["model"] == "devstral-small-latest"
    assert body["messages"][0]["role"] == "system"
    assert body["reasoning_effort"] == "high"


def test_build_request_body_replays_prior_thinking_as_mistral_chunks(
    mistral_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {"type": "text", "text": "Calling the tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo",
                        "input": {"value": "x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result",
                    }
                ],
            },
        ],
    )

    body = mistral_provider._build_request_body(req)

    assistant = body["messages"][0]
    assert "reasoning_content" not in assistant
    assert assistant["content"] == [
        {
            "type": "thinking",
            "thinking": [{"type": "text", "text": "Need the tool."}],
        },
        {"type": "text", "text": "Calling the tool."},
    ]
    assert assistant["tool_calls"][0]["id"] == "toolu_1"


def test_build_request_body_preserves_tools_tool_choice_and_params(mistral_provider):
    req = make_request(
        tools=[
            {
                "name": "echo",
                "description": "Echo a value",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "echo"},
        stop_sequences=["STOP"],
    )

    body = mistral_provider._build_request_body(req)

    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["stop"] == ["STOP"]
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "echo"}}


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    """Global disable disables reasoning replay in the converter."""
    provider = MistralProvider(
        ProviderConfig(
            api_key="test_mistral_key",
            base_url=MISTRAL_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_request()
    body = provider._build_request_body(req)

    assert "reasoning_effort" not in body
    assert all("reasoning_content" not in m for m in body.get("messages", []))


def test_build_request_body_thinking_disabled_strips_prior_mistral_thinking():
    provider = MistralProvider(
        ProviderConfig(
            api_key="test_mistral_key",
            base_url=MISTRAL_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hidden."},
                    {"type": "text", "text": "Visible."},
                ],
            },
        ],
    )

    body = provider._build_request_body(req)

    assert "reasoning_effort" not in body
    assert body["messages"][0]["content"] == "Visible."


@pytest.mark.asyncio
async def test_stream_response_text(mistral_provider):
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
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(mistral_provider):
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
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_native_mistral_thinking_chunk(mistral_provider):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "Native thought."}],
                    }
                ],
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

    assert any(
        '"thinking_delta"' in event and "Native thought." in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_native_mistral_text_chunk(mistral_provider):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=[{"type": "text", "text": "Native text."}],
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

    assert any('"text_delta"' in event and "Native text." in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_preserves_native_thinking_and_string_text(
    mistral_provider,
):
    req = make_request()

    mock_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="Visible token.",
                    thinking="Native thought.",
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            ),
        ],
        usage=MagicMock(completion_tokens=2, prompt_tokens=10),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

    event_text = "\n".join(events)
    assert '"thinking_delta"' in event_text
    assert "Native thought." in event_text
    assert '"text_delta"' in event_text
    assert "Visible token." in event_text


@pytest.mark.asyncio
async def test_stream_response_preserves_native_reasoning_and_string_text(
    mistral_provider,
):
    req = make_request()

    mock_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="Visible token.",
                    reasoning="Native reasoning.",
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(completion_tokens=2, prompt_tokens=10),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

    event_text = "\n".join(events)
    assert "Native reasoning." in event_text
    assert "Visible token." in event_text


@pytest.mark.asyncio
async def test_stream_response_preserves_mixed_native_content_array(
    mistral_provider,
):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "Native thought."}],
                    },
                    {"type": "text", "text": "Native text."},
                ],
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in mistral_provider.stream_response(req)]

    event_text = "\n".join(events)
    assert "Native thought." in event_text
    assert "Native text." in event_text


@pytest.mark.asyncio
async def test_stream_response_suppresses_native_mistral_thinking_when_disabled(
    mistral_provider,
):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "Hidden."}],
                    },
                    {"type": "text", "text": "Visible."},
                ],
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in mistral_provider.stream_response(
                req, thinking_enabled=False
            )
        ]

    event_text = "\n".join(events)
    assert "Hidden." not in event_text
    assert "Visible." in event_text


@pytest.mark.asyncio
async def test_stream_response_retries_without_mistral_reasoning_on_rejection(
    mistral_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_reasoning",
                        "name": "echo",
                        "input": {"value": "FCC_TOOL"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_reasoning",
                        "content": "result",
                    }
                ],
            },
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Recovered", reasoning_content=None, tool_calls=None
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_effort")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in mistral_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["reasoning_effort"] == "high"
    assert first_call["messages"][0]["content"][0]["type"] == "thinking"
    assert "reasoning_effort" not in second_call
    assert second_call["messages"][0]["content"] == ""
    assert second_call["messages"][0]["tool_calls"][0]["id"] == "toolu_reasoning"
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_reasoning_retry_preserves_visible_text_and_tools(
    mistral_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {"type": "text", "text": "Visible history."},
                    {
                        "type": "tool_use",
                        "id": "toolu_reasoning",
                        "name": "echo",
                        "input": {"value": "FCC_TOOL"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_reasoning",
                        "content": "result",
                    }
                ],
            },
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_effort")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in mistral_provider.stream_response(req)]

    second_call = mock_create.await_args_list[1].kwargs
    assert second_call["messages"][0]["content"] == "Visible history."
    assert second_call["messages"][0]["tool_calls"][0]["id"] == "toolu_reasoning"
    assert any("Recovered" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_retries_on_mistral_422_reasoning_rejection(
    mistral_provider,
):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    error = _StatusError(
        "body.messages.assistant.thinking extra_forbidden",
        status_code=422,
        body={"detail": [{"loc": ["body", "reasoning_effort"]}]},
    )

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in mistral_provider.stream_response(req)]

    assert mock_create.await_count == 2
    assert "reasoning_effort" not in mock_create.await_args_list[1].kwargs
    assert any("Recovered" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_retries_when_model_disables_reasoning_input(
    mistral_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need context."},
                    {"type": "text", "text": "Visible history."},
                ],
            }
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    error = _StatusError(
        "Reasoning input is not enabled for this model",
        status_code=400,
        body={
            "object": "error",
            "message": "Reasoning input is not enabled for this model",
            "type": "invalid_request_invalid_args",
            "code": "3051",
        },
    )

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in mistral_provider.stream_response(req)]

    assert mock_create.await_count == 2
    second_call = mock_create.await_args_list[1].kwargs
    assert "reasoning_effort" not in second_call
    assert second_call["messages"][0]["content"] == "Visible history."
    assert any("Recovered" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_unrelated_bad_request_does_not_retry(mistral_provider):
    req = make_request()
    error = _make_bad_request_error("Unsupported field: top_k")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in mistral_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_response_generic_thinking_error_does_not_retry(
    mistral_provider,
):
    req = make_request()
    error = _make_bad_request_error("The model was thinking, but top_k is unsupported")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in mistral_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


def test_retry_body_without_reasoning_returns_none(mistral_provider):
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}

    assert (
        mistral_provider._get_retry_request_body(
            _make_bad_request_error("Unsupported field: reasoning_effort"), body
        )
        is None
    )


@pytest.mark.asyncio
async def test_cleanup(mistral_provider):
    """cleanup closes the OpenAI client."""
    mistral_provider._client = AsyncMock()

    await mistral_provider.cleanup()

    mistral_provider._client.close.assert_called_once()


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    request = Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = Response(400, request=request)
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


class _StatusError(Exception):
    def __init__(self, message: str, *, status_code: int, body: dict):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
