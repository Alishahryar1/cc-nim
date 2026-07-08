"""Tests for the OpenRouter OpenAI-chat provider."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.stream_contracts import parse_sse_text, text_content
from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.open_router import OpenRouterProvider
from providers.transports.openai_chat import OpenAIChatTransport


class AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "moonshotai/kimi-k2.6:free"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.metadata = None
        self.extra_body = {}
        self.thinking = {"type": "enabled"}
        for key, value in kwargs.items():
            setattr(self, key, value)


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
def open_router_provider():
    return OpenRouterProvider(
        ProviderConfig(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
    )


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict] | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    if reasoning_details is not None:
        delta.reasoning_details = reasoning_details
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_init_uses_openai_chat_transport(open_router_provider):
    assert isinstance(open_router_provider, OpenAIChatTransport)
    assert open_router_provider._api_key == "test_openrouter_key"
    assert open_router_provider._base_url == "https://openrouter.ai/api/v1"


def test_build_request_body_uses_openai_chat_shape(open_router_provider):
    body = open_router_provider._build_request_body(MockRequest())

    assert body["model"] == "moonshotai/kimi-k2.6:free"
    assert body["temperature"] == 0.5
    assert body["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]
    assert body["max_tokens"] == 100
    assert body["extra_body"]["reasoning"] == {"enabled": True}


def test_build_request_body_default_max_tokens(open_router_provider):
    body = open_router_provider._build_request_body(MockRequest(max_tokens=None))

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_openrouter_extra_body_rejects_overriding_reserved_fields(
    open_router_provider,
):
    with pytest.raises(InvalidRequestError, match="model"):
        open_router_provider._build_request_body(
            MockRequest(extra_body={"model": "hijack"})
        )


def test_openrouter_extra_body_allows_provider_keys(open_router_provider):
    body = open_router_provider._build_request_body(
        MockRequest(extra_body={"transforms": ["no-web"], "plugins": []}),
        thinking_enabled=False,
    )

    assert body["extra_body"] == {"transforms": ["no-web"], "plugins": []}


def test_build_request_body_omits_reasoning_when_thinking_disabled(
    open_router_provider,
):
    body = open_router_provider._build_request_body(
        MockRequest(thinking={"type": "disabled"})
    )

    assert "extra_body" not in body


def test_build_request_body_maps_thinking_budget_to_reasoning_max_tokens(
    open_router_provider,
):
    body = open_router_provider._build_request_body(
        MockRequest(thinking={"type": "enabled", "budget_tokens": 4096})
    )

    assert body["extra_body"]["reasoning"] == {"enabled": True, "max_tokens": 4096}


def test_build_request_body_replays_openrouter_reasoning_details(
    open_router_provider,
):
    detail = {"type": "reasoning.encrypted", "data": "opaque"}
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": '{"type":"reasoning.encrypted","data":"opaque"}',
                        },
                        {"type": "text", "text": "Need a tool."},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )

    body = open_router_provider._build_request_body(request)

    assistant = next(msg for msg in body["messages"] if msg["role"] == "assistant")
    assert assistant["reasoning_details"] == [detail]


@pytest.mark.asyncio
async def test_stream_maps_reasoning_content_and_details(open_router_provider):
    redacted = {"type": "reasoning.encrypted", "data": "opaque"}
    stream = AsyncStream(
        [
            _chunk(reasoning_content="plan "),
            _chunk(reasoning_details=[redacted]),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [
            event async for event in open_router_provider.stream_response(MockRequest())
        ]

    event_text = "".join(events)
    assert "thinking_delta" in event_text
    assert "plan " in event_text
    assert "redacted_thinking" in event_text
    assert "opaque" in event_text
    assert "done" in text_content(parse_sse_text(event_text))
    assert stream.closed


@pytest.mark.asyncio
async def test_model_infos_filter_tool_models_and_thinking_metadata(
    open_router_provider,
):
    open_router_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="tool-model",
                    supported_parameters=["tools", "reasoning"],
                ),
                SimpleNamespace(id="plain-model", supported_parameters=[]),
            ]
        )
    )

    infos = await open_router_provider.list_model_infos()

    assert {(info.model_id, info.supports_thinking) for info in infos} == {
        ("tool-model", True)
    }


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(open_router_provider):
    open_router_provider._client = MagicMock()
    open_router_provider._client.close = AsyncMock()

    await open_router_provider.cleanup()

    open_router_provider._client.close.assert_awaited_once()
