"""Tests for Wafer native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest, Tool
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.wafer import WAFER_DEFAULT_BASE, WaferProvider
from tests.stream_contract import assert_canonical_stream_error_envelope


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False
        self.headers = httpx.Headers()
        self.request = httpx.Request("POST", "https://pass.wafer.ai/v1/messages")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.is_closed = True

    async def aiter_bytes(self, chunk_size: int = 65_536):
        data = self._text.encode("utf-8")
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    def raise_for_status(self):
        response = httpx.Response(
            self.status_code,
            request=self.request,
            text=self._text,
        )
        response.raise_for_status()


@pytest.fixture
def wafer_config():
    return ProviderConfig(
        api_key="test-wafer-key",
        base_url=WAFER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
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
def wafer_provider(wafer_config):
    return WaferProvider(wafer_config)


def test_default_base_url():
    assert WAFER_DEFAULT_BASE == "https://pass.wafer.ai/v1"


def test_init_uses_default_base_url_and_strips_trailing_slash():
    config = ProviderConfig(api_key="test-wafer-key", base_url=f"{WAFER_DEFAULT_BASE}/")
    with patch("httpx.AsyncClient"):
        provider = WaferProvider(config)

    assert provider._api_key == "test-wafer-key"
    assert provider._base_url == WAFER_DEFAULT_BASE
    assert provider._provider_name == "WAFER"


def test_request_headers_use_bearer_auth_not_x_api_key(wafer_provider):
    headers = wafer_provider._request_headers()

    assert headers["Authorization"] == "Bearer test-wafer-key"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in headers
    assert wafer_provider._model_list_headers() == {
        "Authorization": "Bearer test-wafer-key"
    }


def test_build_request_body_native_shape_and_defaults(wafer_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
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

    body = wafer_provider._build_request_body(request)

    assert body["model"] == "DeepSeek-V4-Pro"
    assert body["messages"][0]["role"] == "user"
    assert body["tools"][0]["name"] == "echo"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["stream"] is True


def test_build_request_body_drops_reasoning_effort_none(wafer_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
            "reasoning_effort": "none",
        }
    )

    body = wafer_provider._build_request_body(request)

    assert "reasoning_effort" not in body
    assert body["thinking"] == {"type": "enabled"}


def test_build_request_body_keeps_upstream_thinking_enabled_when_client_disables_it(
    wafer_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
            "thinking": {"type": "disabled"},
        }
    )

    body = wafer_provider._build_request_body(request, thinking_enabled=False)

    assert body["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_lists_models_from_openai_compatible_models_endpoint(wafer_provider):
    with patch.object(
        wafer_provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "DeepSeek-V4-Pro", "object": "model"},
                    {"id": "MiniMax-M2.7", "object": "model"},
                ],
            },
            request=httpx.Request("GET", "https://pass.wafer.ai/v1/models"),
        ),
    ) as mock_get:
        assert await wafer_provider.list_model_ids() == frozenset(
            {"DeepSeek-V4-Pro", "MiniMax-M2.7"}
        )

    mock_get.assert_awaited_once_with(
        "/models", headers={"Authorization": "Bearer test-wafer-key"}
    )


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(wafer_provider):
    request = MessagesRequest(
        model="MiniMax-M2.7",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    with (
        patch.object(
            wafer_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            wafer_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [event async for event in wafer_provider.stream_response(request)]

    assert events == [
        "event: message_start\n",
        'data: {"type":"message_start"}\n',
        "\n",
        "event: message_stop\n",
        'data: {"type":"message_stop"}\n',
        "\n",
    ]
    assert response.is_closed
    assert mock_build.call_args.args[:2] == ("POST", "/messages")
    assert mock_build.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer test-wafer-key"
    )


@pytest.mark.asyncio
async def test_stream_non_200_maps_to_anthropic_error_event(wafer_provider):
    request = MessagesRequest(
        model="GLM-5.1",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(status_code=500, text="Internal Server Error")

    with (
        patch.object(wafer_provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            wafer_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [
            event
            async for event in wafer_provider.stream_response(
                request, request_id="REQ_WAFER"
            )
        ]

    assert response.is_closed
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Provider API request failed"
    )
    assert "REQ_WAFER" in "".join(events)
