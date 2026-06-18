"""Tests for LM Studio native Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.lmstudio import LMStudioProvider
from tests.stream_contract import assert_canonical_stream_error_envelope


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "lmstudio-community/qwen2.5-7b-instruct"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self, exclude_none=True):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_body": self.extra_body,
            "thinking": {"enabled": self.thinking.enabled} if self.thinking else None,
        }


@pytest.fixture
def lmstudio_config():
    return ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    return LMStudioProvider(lmstudio_config)


def test_init(lmstudio_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient"):
        provider = LMStudioProvider(lmstudio_config)
        assert provider._base_url == "http://localhost:1234/v1"
        assert provider._provider_name == "LMSTUDIO"


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("httpx.AsyncClient") as mock_client:
        LMStudioProvider(config)
        call_kwargs = mock_client.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("httpx.AsyncClient"):
        provider = LMStudioProvider(config)
        assert provider._base_url == "http://localhost:1234/v1"


@pytest.mark.asyncio
async def test_stream_response_omits_thinking_when_globally_disabled(lmstudio_config):
    provider = LMStudioProvider(
        lmstudio_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def empty_aiter():
        if False:
            yield ""

    mock_response.aiter_lines = empty_aiter

    with (
        patch.object(provider._client, "build_request") as mock_build,
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        [e async for e in provider.stream_response(req)]

    _, kwargs = mock_build.call_args
    assert "thinking" not in kwargs["json"]


@pytest.mark.asyncio
async def test_stream_response_omits_thinking_when_request_disables_it(
    lmstudio_provider,
):
    req = MockRequest()
    req.thinking.enabled = False

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def empty_aiter():
        if False:
            yield ""

    mock_response.aiter_lines = empty_aiter

    with (
        patch.object(lmstudio_provider._client, "build_request") as mock_build,
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        [e async for e in lmstudio_provider.stream_response(req)]

    _, kwargs = mock_build.call_args
    assert "thinking" not in kwargs["json"]


@pytest.mark.asyncio
async def test_stream_response(lmstudio_provider):
    """Test streaming native Anthropic response."""
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_lines():
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{}}'
        yield ""
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello World"}}'
        yield ""
        yield "event: message_stop"
        yield 'data: {"type":"message_stop"}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    with (
        patch.object(
            lmstudio_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        events = [e async for e in lmstudio_provider.stream_response(req)]

        # Verify request construction
        mock_build.assert_called_once()
        args, kwargs = mock_build.call_args
        assert args[0] == "POST"
        assert args[1] == "/messages"
        assert kwargs["json"]["model"] == "lmstudio-community/qwen2.5-7b-instruct"
        # Verify internal fields are popped
        assert "extra_body" not in kwargs["json"]
        assert kwargs["json"]["max_tokens"] == 100

        # Verify internal ThinkingConfig is mapped to Anthropic API format
        assert kwargs["json"]["thinking"] == {"type": "enabled"}

        # Verify events yielded correctly
        assert len(events) == 9
        assert events[0] == "event: message_start\n"
        assert events[1] == 'data: {"type":"message_start","message":{}}\n'


@pytest.mark.asyncio
async def test_stream_response_adds_max_tokens_if_missing(lmstudio_provider):
    """Fallback max_tokens to 81920 if not present."""
    req = MockRequest()
    mock_response = MagicMock()
    mock_response.status_code = 200

    async def empty_aiter():
        if False:
            yield ""

    mock_response.aiter_lines = empty_aiter

    with (
        patch.object(req, "model_dump", return_value={"model": "test"}),
        patch.object(lmstudio_provider._client, "build_request") as mock_build,
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        # Just run the generator to completion
        [e async for e in lmstudio_provider.stream_response(req)]

        _, kwargs = mock_build.call_args
        assert kwargs["json"]["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_stream_error_status_code(lmstudio_provider):
    """Non-200 status code raises an error that gets caught and yielded as an SSE API error."""
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response
        )
    )

    with (
        patch.object(
            lmstudio_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        events = [
            e
            async for e in lmstudio_provider.stream_response(req, request_id="TEST_ID")
        ]

        assert_canonical_stream_error_envelope(
            events, user_message_substr="Provider API request failed"
        )
        assert "TEST_ID" in "".join(events)


@pytest.mark.asyncio
async def test_stream_network_error(lmstudio_provider):
    """Network errors are caught and yielded as SSE API error events."""
    req = MockRequest()

    with (
        patch.object(
            lmstudio_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ),
    ):
        events = [
            e
            async for e in lmstudio_provider.stream_response(req, request_id="TEST_ID2")
        ]

        blob = "".join(events)
        assert_canonical_stream_error_envelope(
            events, user_message_substr="Connection refused"
        )
        assert "TEST_ID2" in blob


@pytest.mark.asyncio
async def test_stream_error_405_mentions_upstream_provider(lmstudio_provider):
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 405
    mock_response.aread = AsyncMock(return_value=b"Method Not Allowed")
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Method Not Allowed", request=MagicMock(), response=mock_response
        )
    )

    with (
        patch.object(
            lmstudio_provider._client, "build_request", return_value=MagicMock()
        ),
        patch.object(
            lmstudio_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        events = [
            e async for e in lmstudio_provider.stream_response(req, request_id="REQ405")
        ]

    blob = "".join(events)
    assert (
        "Upstream provider LMSTUDIO rejected the request method or endpoint (HTTP 405)."
        in blob
    )
    assert "REQ405" in blob


def test_build_request_body_disabled_thinking_strips_native_thinking_history(
    lmstudio_config,
):
    """Disabled thinking must omit prior assistant thinking/redacted blocks in JSON."""
    provider = LMStudioProvider(
        lmstudio_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "hi"),
            MockMessage(
                "assistant",
                [
                    {"type": "thinking", "thinking": "t"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            ),
        ],
    )
    body = provider._build_request_body(req, thinking_enabled=False)
    assert body["messages"][1]["content"] == ""
    assert "redacted_thinking" not in str(body)


def test_build_request_body_preserves_signed_thinking_native_history(lmstudio_config):
    """When thinking is enabled, signed thinking blocks are kept; unsigned stripped."""
    provider = LMStudioProvider(lmstudio_config)
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "hi"),
            MockMessage(
                "assistant",
                [
                    {
                        "type": "thinking",
                        "thinking": "signed",
                        "signature": "sig",
                    },
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            ),
        ],
    )
    body = provider._build_request_body(req, thinking_enabled=True)
    c = body["messages"][1]["content"]
    assert isinstance(c, list)
    assert any(isinstance(b, dict) and b.get("type") == "thinking" for b in c)
    assert any(isinstance(b, dict) and b.get("type") == "redacted_thinking" for b in c)
