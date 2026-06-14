"""Tests for Xiaomi MiMo native Anthropic Messages provider.

MiMo uses the Anthropic-compatible Messages API at:
  POST  https://api.xiaomimimo.com/anthropic/v1/messages
  GET   https://api.xiaomimimo.com/v1/models   (OpenAI-format root, NOT /anthropic/v1/models)

Key differences from DeepSeek (also a native Anthropic provider):
  - Auth header is ``Authorization: Bearer <key>`` + ``anthropic-version``, not ``x-api-key``.
  - Base URL includes the ``/v1`` segment (``/anthropic/v1``), so ``/messages`` resolves correctly.
  - ``extra_body`` is passed through (like Fireworks), not stripped (like DeepSeek).
  - ``_send_model_list_request`` overrides the path to ``/v1/models`` via ``copy_with``.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import XIAOMIMIMO_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.xiaomimimo import XiaomiMiMoProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mimo_config():
    return ProviderConfig(
        api_key="test_mimo_key",
        base_url=XIAOMIMIMO_DEFAULT_BASE,
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
def mimo_provider(mimo_config):
    return XiaomiMiMoProvider(mimo_config)


# ---------------------------------------------------------------------------
# Constant / init tests
# ---------------------------------------------------------------------------


def test_default_base_url_constant():
    """Base URL must include /anthropic/v1 so POST /messages resolves correctly."""
    assert XIAOMIMIMO_DEFAULT_BASE == "https://api.xiaomimimo.com/anthropic/v1"


def test_init(mimo_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = XiaomiMiMoProvider(mimo_config)
    assert provider._api_key == "test_mimo_key"
    assert provider._base_url == "https://api.xiaomimimo.com/anthropic/v1"
    assert mock_client.called


# ---------------------------------------------------------------------------
# Header tests
# ---------------------------------------------------------------------------


def test_request_headers_bearer_auth(mimo_provider):
    """MiMo uses Authorization: Bearer, not x-api-key."""
    h = mimo_provider._request_headers()
    assert h["Authorization"] == "Bearer test_mimo_key"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["Content-Type"] == "application/json"
    assert h["Accept"] == "text/event-stream"
    assert "x-api-key" not in h


def test_model_list_headers(mimo_provider):
    h = mimo_provider._model_list_headers()
    assert h["Authorization"] == "Bearer test_mimo_key"


# ---------------------------------------------------------------------------
# Request body tests
# ---------------------------------------------------------------------------


def test_build_request_body_native_shape(mimo_provider):
    request = MessagesRequest(
        model="mimo-v2.5-pro",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="You are a coding assistant.",
    )
    body = mimo_provider._build_request_body(request)
    assert body["model"] == "mimo-v2.5-pro"
    assert body["stream"] is True
    assert body["max_tokens"] == 100
    assert body["system"] == "You are a coding assistant."
    assert body["messages"][0]["role"] == "user"


def test_build_request_body_default_max_tokens(mimo_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = mimo_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(mimo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "mimo-v2.5-pro",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = mimo_provider._build_request_body(request)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert "extra_body" not in body


def test_build_request_body_global_disable_blocks_thinking():
    provider = XiaomiMiMoProvider(
        ProviderConfig(
            api_key="k",
            base_url=XIAOMIMIMO_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "thinking" not in body


def test_build_request_body_merges_safe_extra_body(mimo_provider):
    """MiMo passes extra_body through (like Fireworks), not strips it."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"custom_param": "value"},
        }
    )
    body = mimo_provider._build_request_body(request)
    assert body["custom_param"] == "value"


def test_build_request_body_rejects_reserved_extra_body_keys(mimo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"temperature": 0.5},
        }
    )
    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        mimo_provider._build_request_body(request)


# ---------------------------------------------------------------------------
# Streaming path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(mimo_provider):
    """The messages POST must resolve to /anthropic/v1/messages, not /anthropic/messages."""
    request = MessagesRequest(
        model="mimo-v2.5-pro",
        messages=[Message(role="user", content="hi")],
    )
    called: dict[str, str] = {}

    async def fake_send(req, *args, **kwargs):
        called["path"] = req.url.path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_closed = False
        mock_resp.raise_for_status = lambda: None

        async def aiter():
            if False:  # pragma: no cover
                yield ""

        mock_resp.aiter_lines = aiter
        mock_resp.aclose = AsyncMock()
        return mock_resp

    mimo_provider._client.send = fake_send
    _ = [x async for x in mimo_provider.stream_response(request, request_id="r1")]

    assert called["path"] == "/anthropic/v1/messages"


# ---------------------------------------------------------------------------
# Model list URL override test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_list_uses_v1_models_root_url(mimo_provider):
    """Model list must call /v1/models at the API root, NOT /anthropic/v1/models."""
    called: dict[str, str] = {}

    async def fake_get(url: str, **_k):
        called["url"] = url
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"data": [{"id": "mimo-v2.5-pro"}]}
        mock_resp.aclose = AsyncMock()
        return mock_resp

    mimo_provider._client.get = fake_get

    await mimo_provider.list_model_infos()

    assert called["url"] == "https://api.xiaomimimo.com/v1/models"


# ---------------------------------------------------------------------------
# Cleanup test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_aclose(mimo_provider):
    mimo_provider._client = AsyncMock()
    await mimo_provider.cleanup()
    mimo_provider._client.aclose.assert_awaited_once()
