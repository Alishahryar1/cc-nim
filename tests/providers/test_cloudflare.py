"""Tests for Cloudflare AI REST native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest
from core.anthropic.stream_contracts import parse_sse_text
from providers.base import ProviderConfig
from providers.cloudflare import (
    CLOUDFLARE_AI_REST_ROOT,
    CloudflareProvider,
    cloudflare_ai_base_url,
)
from providers.exceptions import AuthenticationError, InvalidRequestError


class FakeResponse:
    def __init__(self, *, status_code: int = 200, lines: list[str] | None = None):
        self.status_code = status_code
        self._lines = lines or []
        self.is_closed = False
        self.headers = httpx.Headers()
        self.request = httpx.Request("POST", f"{_BASE_URL}/messages")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_bytes(self, chunk_size: int = 65_536):
        if False:
            yield b""

    async def aclose(self):
        self.is_closed = True

    def raise_for_status(self):
        response = httpx.Response(self.status_code, request=self.request)
        response.raise_for_status()


_ACCOUNT_ID = "account-123"
_BASE_URL = f"{CLOUDFLARE_AI_REST_ROOT}/accounts/{_ACCOUNT_ID}/ai/v1"
_MODEL_SEARCH_URL = f"{CLOUDFLARE_AI_REST_ROOT}/accounts/{_ACCOUNT_ID}/ai/models/search"


@pytest.fixture
def cloudflare_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-cloudflare-token",
        base_url=CLOUDFLARE_AI_REST_ROOT,
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
def cloudflare_provider(cloudflare_config: ProviderConfig) -> CloudflareProvider:
    return CloudflareProvider(cloudflare_config, account_id=_ACCOUNT_ID)


def test_cloudflare_ai_base_url_uses_account_scoped_messages_root() -> None:
    assert cloudflare_ai_base_url(CLOUDFLARE_AI_REST_ROOT, "account/with slash") == (
        f"{CLOUDFLARE_AI_REST_ROOT}/accounts/account%2Fwith%20slash/ai/v1"
    )


def test_missing_account_id_raises_authentication_error(
    cloudflare_config: ProviderConfig,
) -> None:
    with pytest.raises(AuthenticationError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareProvider(cloudflare_config, account_id=" ")


def test_init_composes_account_scoped_base_url(
    cloudflare_config: ProviderConfig,
) -> None:
    with patch("httpx.AsyncClient") as mock_client:
        provider = CloudflareProvider(cloudflare_config, account_id=_ACCOUNT_ID)

    assert provider._api_key == "test-cloudflare-token"
    assert provider._base_url == _BASE_URL
    assert provider._model_search_url == _MODEL_SEARCH_URL
    assert provider._provider_name == "CLOUDFLARE"
    assert mock_client.called


def test_request_headers_use_bearer_auth(cloudflare_provider: CloudflareProvider):
    headers = cloudflare_provider._request_headers()

    assert headers["Authorization"] == "Bearer test-cloudflare-token"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in headers
    assert cloudflare_provider._model_list_headers() == {
        "Authorization": "Bearer test-cloudflare-token"
    }


def test_build_request_body_preserves_slash_model_id_and_forwards_thinking(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [Message(role="user", content="Hello")],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = cloudflare_provider._build_request_body(request, thinking_enabled=True)

    assert body["model"] == "anthropic/claude-sonnet-4-5"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_build_request_body_strips_prior_thinking_blocks_when_disabled(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "visible"},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ],
        }
    )

    body = cloudflare_provider._build_request_body(request, thinking_enabled=False)

    assert body["messages"][0]["content"] == [{"type": "text", "text": "visible"}]


def test_build_request_body_request_disabled_thinking_suppresses_thinking(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = cloudflare_provider._build_request_body(request, thinking_enabled=True)

    assert "thinking" not in body


def test_build_request_body_rejects_extra_body(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"custom": True},
        }
    )

    with pytest.raises(InvalidRequestError, match="does not support extra_body"):
        cloudflare_provider._build_request_body(request)


@pytest.mark.asyncio
async def test_lists_models_from_cloudflare_model_search_endpoint(
    cloudflare_provider: CloudflareProvider,
) -> None:
    with patch.object(
        cloudflare_provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "anthropic/claude-sonnet-4-5", "object": "model"},
                    {"id": "anthropic/claude-opus-4-5", "object": "model"},
                ],
            },
            request=httpx.Request("GET", _MODEL_SEARCH_URL),
        ),
    ) as mock_get:
        assert await cloudflare_provider.list_model_ids() == frozenset(
            {"anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5"}
        )

    mock_get.assert_awaited_once_with(
        _MODEL_SEARCH_URL,
        params={"format": "openrouter"},
        headers={"Authorization": "Bearer test-cloudflare-token"},
    )


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest(
        model="anthropic/claude-sonnet-4-5",
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
            cloudflare_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            cloudflare_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [event async for event in cloudflare_provider.stream_response(request)]

    assert [event.event for event in parse_sse_text("".join(events))] == [
        "message_start",
        "message_stop",
    ]
    assert response.is_closed
    assert mock_build.call_args.args[:2] == ("POST", "/messages")
    assert mock_build.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer test-cloudflare-token"
    )
