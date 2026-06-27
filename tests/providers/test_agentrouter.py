"""Tests for AgentRouter native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest, Tool
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.agentrouter import AGENTROUTER_DEFAULT_BASE, AgentRouterProvider
from providers.base import ProviderConfig
from tests.stream_contract import assert_canonical_stream_error_envelope


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False
        self.headers = httpx.Headers()
        self.request = httpx.Request("POST", "https://agentrouter.org/v1/messages")

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
def agentrouter_config():
    return ProviderConfig(
        api_key="test-agentrouter-key",
        base_url=AGENTROUTER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def agentrouter_provider(agentrouter_config):
    return AgentRouterProvider(agentrouter_config)


def test_default_base_url():
    assert AGENTROUTER_DEFAULT_BASE == "https://agentrouter.org/v1"


def test_init_uses_default_base_url_and_strips_trailing_slash():
    config = ProviderConfig(
        api_key="test-agentrouter-key",
        base_url=f"{AGENTROUTER_DEFAULT_BASE}/",
    )
    with patch("httpx.AsyncClient"):
        provider = AgentRouterProvider(config)

    assert provider._api_key == "test-agentrouter-key"
    assert provider._base_url == AGENTROUTER_DEFAULT_BASE
    assert provider._provider_name == "AGENTROUTER"


def test_request_headers_contain_claude_cli_fingerprint(agentrouter_provider):
    headers = agentrouter_provider._request_headers()

    assert headers["Authorization"] == "Bearer test-agentrouter-key"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["User-Agent"] == "claude-cli/1.0.108 (external, cli)"
    assert headers["x-app"] == "cli"
    assert headers["x-stainless-lang"] == "js"
    assert headers["x-stainless-package-version"] == "0.55.1"
    assert headers["x-stainless-os"] == "Windows"
    assert headers["x-stainless-arch"] == "x64"
    assert headers["x-stainless-runtime"] == "node"
    assert headers["x-stainless-runtime-version"] == "v22.0.0"
    assert "x-api-key" not in headers


def test_model_list_headers_contain_auth_and_fingerprint(agentrouter_provider):
    headers = agentrouter_provider._model_list_headers()

    assert headers["Authorization"] == "Bearer test-agentrouter-key"
    assert headers["User-Agent"] == "claude-cli/1.0.108 (external, cli)"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
    assert headers["x-app"] == "cli"


def test_build_request_body_native_shape_and_defaults(agentrouter_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-6",
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

    body = agentrouter_provider._build_request_body(request)

    assert body["model"] == "claude-opus-4-6"
    assert body["messages"][0]["role"] == "user"
    assert body["tools"][0]["name"] == "echo"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["stream"] is True


def test_build_request_body_injects_default_budget_tokens(agentrouter_provider):
    """AgentRouter requires budget_tokens; inject default when client omits it."""
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-6",
            "messages": [Message(role="user", content="Hello")],
            "thinking": {"type": "enabled"},
        }
    )

    body = agentrouter_provider._build_request_body(request)

    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 10000


def test_build_request_body_preserves_explicit_budget_tokens(agentrouter_provider):
    """When budget_tokens is explicitly provided, do not override it."""
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-6",
            "messages": [Message(role="user", content="Hello")],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
    )

    body = agentrouter_provider._build_request_body(request)

    assert body["thinking"]["budget_tokens"] == 5000


@pytest.mark.asyncio
async def test_lists_models_from_openai_compatible_models_endpoint(
    agentrouter_provider,
):
    with patch.object(
        agentrouter_provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "claude-opus-4-6", "object": "model"},
                    {"id": "deepseek-v4-flash", "object": "model"},
                ],
            },
            request=httpx.Request("GET", "https://agentrouter.org/v1/models"),
        ),
    ) as mock_get:
        assert await agentrouter_provider.list_model_ids() == frozenset(
            {"claude-opus-4-6", "deepseek-v4-flash"}
        )

    mock_get.assert_awaited_once()
    call_headers = mock_get.call_args.kwargs.get(
        "headers",
        mock_get.call_args.args[1] if len(mock_get.call_args.args) > 1 else {},
    )
    assert call_headers["Authorization"] == "Bearer test-agentrouter-key"


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(agentrouter_provider):
    request = MessagesRequest(
        model="claude-opus-4-6",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
        ]
    )

    with (
        patch.object(
            agentrouter_provider._client,
            "build_request",
            return_value=MagicMock(),
        ) as mock_build,
        patch.object(
            agentrouter_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [
            event async for event in agentrouter_provider.stream_response(request)
        ]

    assert events == [
        "event: message_start\n",
        'data: {"type":"message_start"}\n',
        "\n",
    ]
    assert response.is_closed
    assert mock_build.call_args.args[:2] == ("POST", "/messages")
    assert mock_build.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer test-agentrouter-key"
    )


@pytest.mark.asyncio
async def test_stream_filters_data_null_lines(agentrouter_provider):
    """AgentRouter may emit ``data: null`` lines that must be stripped."""
    request = MessagesRequest(
        model="deepseek-v4-flash",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "data: null",
            "data:null",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    with (
        patch.object(
            agentrouter_provider._client,
            "build_request",
            return_value=MagicMock(),
        ),
        patch.object(
            agentrouter_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [
            event async for event in agentrouter_provider.stream_response(request)
        ]

    joined = "".join(events)
    assert "data: null" not in joined
    assert "data:null" not in joined
    assert "message_start" in joined
    assert "message_stop" in joined


@pytest.mark.asyncio
async def test_stream_non_200_maps_to_anthropic_error_event(agentrouter_provider):
    request = MessagesRequest(
        model="claude-opus-4-6",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(status_code=500, text="Internal Server Error")

    with (
        patch.object(
            agentrouter_provider._client,
            "build_request",
            return_value=MagicMock(),
        ),
        patch.object(
            agentrouter_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [
            event
            async for event in agentrouter_provider.stream_response(
                request, request_id="REQ_AR"
            )
        ]

    assert response.is_closed
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Provider API request failed"
    )
    assert "REQ_AR" in "".join(events)
