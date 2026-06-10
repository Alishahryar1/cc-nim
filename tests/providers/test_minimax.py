"""Tests for MiniMax native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import MINIMAX_DEFAULT_BASE
from providers.minimax import MinimaxProvider


@pytest.fixture
def minimax_config():
    return ProviderConfig(
        api_key="test_minimax_key",
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

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def minimax_provider(minimax_config):
    return MinimaxProvider(minimax_config)


def test_default_base_url():
    assert MINIMAX_DEFAULT_BASE == "https://api.minimaxi.com/anthropic/v1"


def test_init(minimax_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = MinimaxProvider(minimax_config)
    assert provider._api_key == "test_minimax_key"
    assert provider._base_url == "https://api.minimaxi.com/anthropic/v1"
    assert mock_client.called


def test_request_headers(minimax_provider):
    h = minimax_provider._request_headers()
    assert h["Authorization"] == "Bearer test_minimax_key"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["Content-Type"] == "application/json"
    assert h["Accept"] == "text/event-stream"


def test_build_request_body_native(minimax_provider):
    request = MessagesRequest(
        model="MiniMax-M3",
        max_tokens=50,
        messages=[Message(role="user", content="hi")],
    )
    body = minimax_provider._build_request_body(request)
    assert body["model"] == "MiniMax-M3"
    assert body["stream"] is True
    assert body["messages"][0]["role"] == "user"


def test_build_request_body_default_max_tokens(minimax_provider):
    request = MessagesRequest(
        model="MiniMax-M2",
        messages=[Message(role="user", content="x")],
    )
    body = minimax_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(minimax_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = minimax_provider._build_request_body(request)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}


def test_build_request_body_tool_use_passthrough(minimax_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )
    body = minimax_provider._build_request_body(request)
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_respects_global_thinking_disable():
    provider = MinimaxProvider(
        ProviderConfig(
            api_key="k",
            base_url=MINIMAX_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_model_list_uses_openai_root_url(minimax_provider):
    called: dict[str, str] = {}

    async def fake_get(url: str, **_k):
        called["url"] = str(url)
        raise RuntimeError("stop")

    minimax_provider._client.get = fake_get
    with pytest.raises(RuntimeError, match="stop"):
        await minimax_provider._send_model_list_request()

    assert called["url"] == "https://api.minimaxi.com/v1/models"
