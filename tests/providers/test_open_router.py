"""Tests for OpenRouter providers."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
    thinking_content,
)
from providers.base import ProviderConfig
from providers.open_router import OpenRouterProvider
from providers.open_router.request import OPENROUTER_DEFAULT_MAX_TOKENS


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
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._text.encode()

    def raise_for_status(self):
        import httpx

        response = httpx.Response(
            self.status_code,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
            text=self._text,
        )
        response.raise_for_status()

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def open_router_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
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
def open_router_provider(open_router_config):
    return OpenRouterProvider(open_router_config)


def test_init(open_router_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient") as mock_client:
        provider = OpenRouterProvider(open_router_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_client.assert_called_once()


def test_init_uses_configurable_timeouts():
    """Provider passes configurable read/write/connect timeouts to httpx."""
    config = ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("httpx.AsyncClient") as mock_client:
        OpenRouterProvider(config)
        timeout = mock_client.call_args.kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_is_native_anthropic(open_router_provider):
    req = MockRequest()
    body = open_router_provider._build_request_body(req)

    assert body["model"] == "moonshotai/kimi-k2.6:free"
    assert body["temperature"] == 0.5
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["system"] == "System prompt"
    assert body["reasoning"] == {"enabled": True}
    assert "extra_body" not in body


def test_openrouter_extra_body_rejects_overriding_reserved_fields() -> None:
    from providers.exceptions import InvalidRequestError
    from providers.open_router.request import build_request_body

    r = MockRequest()
    r.extra_body = {"model": "hijack"}
    with pytest.raises(InvalidRequestError, match="model"):
        build_request_body(r, thinking_enabled=True)


def test_openrouter_extra_body_allows_openrouter_only_keys() -> None:
    from providers.open_router.request import build_request_body

    r = MockRequest()
    r.extra_body = {"transforms": ["no-web"], "plugins": []}
    body = build_request_body(r, thinking_enabled=False)
    assert body["transforms"] == ["no-web"]
    assert body["plugins"] == []


def test_build_request_body_omits_reasoning_when_globally_disabled(
    open_router_config,
):
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )

    body = provider._build_request_body(MockRequest())

    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    open_router_provider,
):
    req = MockRequest()
    req.thinking.enabled = False

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_native_thinking_disabled(
    open_router_provider,
):
    req = MockRequest(thinking={"type": "disabled"})

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def test_build_request_body_maps_thinking_budget_to_reasoning_max_tokens(
    open_router_provider,
):
    req = MockRequest(thinking={"type": "enabled", "budget_tokens": 4096})

    body = open_router_provider._build_request_body(req)

    assert body["reasoning"] == {"enabled": True, "max_tokens": 4096}


def test_build_request_body_default_max_tokens(open_router_provider):
    req = MockRequest(max_tokens=None)

    body = open_router_provider._build_request_body(req)

    assert body["max_tokens"] == OPENROUTER_DEFAULT_MAX_TOKENS


def test_build_request_body_strips_unsigned_thinking_history(open_router_provider):
    req = MockRequest(
        messages=[
            MockMessage("user", "hello"),
            MockMessage(
                "assistant",
                [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Hello"},
                ],
            ),
            MockMessage("user", "can you think hard about 2+2"),
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["messages"][1]["content"] == [
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Hello"},
    ]


def test_build_request_body_strips_redacted_when_thinking_disabled(
    open_router_config,
):
    """Disabled thinking must remove all assistant thinking history including redacted."""
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest(
        messages=[
            MockMessage(
                "assistant",
                [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Hi"},
                ],
            )
        ]
    )
    body = provider._build_request_body(req)
    assert body["messages"][0]["content"] == [{"type": "text", "text": "Hi"}]


def test_build_request_body_preserves_signed_thinking_history(open_router_provider):
    req = MockRequest(
        messages=[
            MockMessage(
                "assistant",
                [
                    {
                        "type": "thinking",
                        "thinking": "signed",
                        "signature": "sig_123",
                    }
                ],
            )
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "signed", "signature": "sig_123"}
    ]


def test_build_request_body_flattens_system_blocks(open_router_provider):
    req = MockRequest(
        system=[
            {"type": "text", "text": "First system block."},
            {"type": "text", "text": "Second system block."},
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["system"] == "First system block.\n\nSecond system block."


@pytest.mark.asyncio
async def test_stream_response_passes_native_sse_events(open_router_provider):
    req = MockRequest()
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
            "event: data",
            "data: [DONE]",
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request") as mock_build,
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    _, kwargs = mock_build.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test_openrouter_key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert events[0].startswith("event: message_start")
    assert events[-1].startswith("event: message_stop")
    assert any("Hello" in event for event in events)
    assert "[DONE]" not in "".join(events)
    assert response.is_closed


@pytest.mark.asyncio
async def test_stream_response_suppresses_native_thinking_when_disabled(
    open_router_config,
):
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"secret"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
        ]
    )

    with (
        patch.object(provider._client, "build_request"),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in provider.stream_response(MockRequest())]

    event_text = "".join(events)
    assert "thinking_delta" not in event_text
    assert "secret" not in event_text
    assert "Answer" in event_text

    text_start = next(event for event in events if "content_block_start" in event)
    payload = parse_sse_text(text_start)[0].data
    assert payload["index"] == 0


@pytest.mark.asyncio
async def test_stream_response_preserves_redacted_thinking_when_enabled(
    open_router_provider,
):
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
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
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    event_text = "".join(events)
    assert "redacted_thinking" in event_text
    assert "opaque" in event_text
    assert "Answer" in event_text

    parsed = parse_sse_text(event_text)
    first_start = next(
        p
        for p in parsed
        if p.event == "content_block_start"
        and p.data.get("content_block", {}).get("type") == "redacted_thinking"
    )
    assert first_start.data["index"] == 0


@pytest.mark.asyncio
async def test_stream_response_drops_redacted_thinking_when_disabled(
    open_router_config,
):
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
        ]
    )

    with (
        patch.object(provider._client, "build_request"),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in provider.stream_response(MockRequest())]

    event_text = "".join(events)
    assert "redacted_thinking" not in event_text
    assert "opaque" not in event_text
    assert "Answer" in event_text

    start_event = next(event for event in events if "content_block_start" in event)
    payload = parse_sse_text(start_event)[0].data
    assert payload["index"] == 0
    assert payload["content_block"]["type"] == "text"


@pytest.mark.asyncio
async def test_stream_response_reopens_interleaved_thinking_after_text(
    open_router_provider,
):
    """Overthinking+text+more thinking: downstream indices must not reuse closed blocks."""
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"thinking","thinking":"","signature":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"thinking_delta","thinking":"first"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,'
            '"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"thinking_delta","thinking":" second"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,'
            '"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    parsed = parse_sse_text("".join(events))
    assert_anthropic_stream_contract(parsed)
    assert thinking_content(parsed) == "first second"
    assert "Answer" in text_content(parsed)
    stop_payloads = [
        p.data
        for p in parsed
        if p.event == "content_block_stop"
        and p.data.get("type") == "content_block_stop"
    ]
    seen_stop_indices: set[int] = set()
    for s in stop_payloads:
        idx = s.get("index")
        assert isinstance(idx, int)
        assert idx not in seen_stop_indices, "stop reused or duplicated index"
        seen_stop_indices.add(idx)
    # Two distinct thinking block indices: initial + reopened segment
    think_starts = [
        p
        for p in parsed
        if p.event == "content_block_start"
        and p.data.get("content_block", {}).get("type") == "thinking"
    ]
    assert len(think_starts) == 2, (
        "reopened thinking must have its own `content_block_start`"
    )


@pytest.mark.asyncio
async def test_stream_response_reopened_tool_use_preserves_tool_identity(
    open_router_provider,
):
    """After overlapping close, resumed input_json_delta must keep original tool id/name."""
    lines: list[str] = []
    for payload in (
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_real_1",
                "name": "Read",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path'},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '":"/tmp"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_stop", "index": 0},
    ):
        event_name = payload["type"]
        lines.extend((f"event: {event_name}", f"data: {json.dumps(payload)}", ""))

    response = FakeResponse(lines=lines)

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    parsed = parse_sse_text("".join(events))
    tool_starts = [
        p
        for p in parsed
        if p.event == "content_block_start"
        and p.data.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 2
    for start in tool_starts:
        block = start.data["content_block"]
        assert block["id"] == "toolu_real_1"
        assert block["name"] == "Read"


@pytest.mark.asyncio
async def test_stream_response_closes_overlapping_thinking_before_text(
    open_router_provider,
):
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    event_text = "".join(events)
    thinking_stop = event_text.index('"type": "content_block_stop", "index": 0')
    text_start = event_text.index('"content_block": {"type": "text"')
    assert thinking_stop < text_start
    assert event_text.count('"index": 0') == 3
    assert "Answer" in event_text


@pytest.mark.asyncio
async def test_stream_response_error_path(open_router_provider):
    req = MockRequest()

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API failed"),
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    assert "message_start" in event_text
    assert "API failed" in event_text
    assert "message_stop" in event_text
