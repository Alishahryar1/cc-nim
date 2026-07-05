"""Tests for MiniMax native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest, Tool
from core.anthropic.stream_contracts import parse_sse_text, thinking_content
from providers.base import ProviderConfig
from providers.minimax import MINIMAX_DEFAULT_BASE, MiniMaxProvider


class FakeResponse:
    def __init__(self, *, lines=None):
        self.status_code = 200
        self._lines = lines or []
        self.is_closed = False
        self.headers = httpx.Headers()
        self.request = httpx.Request(
            "POST", "https://api.minimax.io/anthropic/v1/messages"
        )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def minimax_config():
    return ProviderConfig(
        api_key="test-minimax-key",
        base_url=MINIMAX_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def minimax_provider(minimax_config):
    return MiniMaxProvider(minimax_config)


def test_default_base_url():
    assert MINIMAX_DEFAULT_BASE == "https://api.minimax.io/anthropic/v1"


def test_init_uses_default_base_url_and_strips_trailing_slash(minimax_config):
    config = minimax_config.model_copy(update={"base_url": f"{MINIMAX_DEFAULT_BASE}/"})

    with patch("httpx.AsyncClient"):
        provider = MiniMaxProvider(config)

    assert provider._api_key == "test-minimax-key"
    assert provider._base_url == MINIMAX_DEFAULT_BASE
    assert provider._provider_name == "MINIMAX"


def test_headers_use_x_api_key(minimax_provider):
    headers = minimax_provider._request_headers()

    assert headers["x-api-key"] == "test-minimax-key"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert minimax_provider._model_list_headers() == {
        "x-api-key": "test-minimax-key",
        "anthropic-version": "2023-06-01",
    }


def test_build_request_body_uses_adaptive_thinking_and_preserves_tools(
    minimax_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [Message(role="user", content="Hello")],
            "tools": [
                Tool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = minimax_provider._build_request_body(request)

    assert body["model"] == "MiniMax-M3"
    assert body["tools"][0]["name"] == "echo"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["stream"] is True


def test_build_request_body_enables_adaptive_thinking_by_default(minimax_provider):
    request = MessagesRequest(
        model="MiniMax-M3",
        messages=[Message(role="user", content="Hello")],
    )

    body = minimax_provider._build_request_body(request)

    assert body["thinking"] == {"type": "adaptive"}


def test_build_request_body_honors_no_thinking(minimax_provider):
    request = MessagesRequest(
        model="MiniMax-M3",
        messages=[Message(role="user", content="Hello")],
    )

    body = minimax_provider._build_request_body(request, thinking_enabled=False)

    assert body["thinking"] == {"type": "disabled"}


def test_build_request_body_preserves_request_disabled_thinking(minimax_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = minimax_provider._build_request_body(request, thinking_enabled=True)

    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_lists_models_from_anthropic_compatible_models_endpoint(
    minimax_provider,
):
    with patch.object(
        minimax_provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "MiniMax-M3", "type": "model"},
                    {"id": "MiniMax-M2.7", "type": "model"},
                ]
            },
            request=httpx.Request("GET", "https://api.minimax.io/anthropic/v1/models"),
        ),
    ) as mock_get:
        assert await minimax_provider.list_model_ids() == frozenset(
            {"MiniMax-M3", "MiniMax-M2.7"}
        )

    mock_get.assert_awaited_once_with(
        "/models",
        headers={
            "x-api-key": "test-minimax-key",
            "anthropic-version": "2023-06-01",
        },
    )


@pytest.mark.asyncio
async def test_stream_uses_messages_path_and_preserves_native_thinking(
    minimax_provider,
):
    request = MessagesRequest(
        model="MiniMax-M3",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    with (
        patch.object(
            minimax_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            minimax_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [event async for event in minimax_provider.stream_response(request)]

    parsed = parse_sse_text("".join(events))
    assert [event.event for event in parsed] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    ]
    assert thinking_content(parsed) == "plan"
    assert response.is_closed
    assert mock_build.call_args.args[:2] == ("POST", "/messages")
    assert mock_build.call_args.kwargs["headers"]["x-api-key"] == "test-minimax-key"
