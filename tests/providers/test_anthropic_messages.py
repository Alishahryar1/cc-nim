"""Tests for the shared native Anthropic Messages transport."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.sse import format_sse_event
from core.anthropic.stream_contracts import event_index, parse_sse_text
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from tests.stream_contract import assert_canonical_stream_error_envelope


class NativeProvider(AnthropicMessagesTransport):
    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="TEST_NATIVE",
            default_base_url="https://example.test/v1",
        )

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "X-Test": "1"}


class MockRequest:
    model = "test-model"

    def __init__(self, *, thinking_enabled: bool = True, body: dict | None = None):
        self.thinking = MagicMock()
        self.thinking.enabled = thinking_enabled
        self._body = body or {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"ignored": True},
            "thinking": {"enabled": thinking_enabled},
        }

    def model_dump(self, exclude_none=True):
        return dict(self._body)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        lines=None,
        text="",
        raise_after_line_index: int | None = None,
        raise_error: Exception | None = None,
    ):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self._raise_after_line_index = raise_after_line_index
        self._raise_error = raise_error or RuntimeError("mid-stream failure")
        self.is_closed = False
        self.request = httpx.Request("POST", "https://example.test/v1/messages")
        self.headers = httpx.Headers()

    async def aiter_lines(self):
        for i, line in enumerate(self._lines):
            yield line
            if (
                self._raise_after_line_index is not None
                and i >= self._raise_after_line_index
            ):
                raise self._raise_error

    async def aread(self):
        return self._text.encode()

    def raise_for_status(self):
        response = httpx.Response(
            self.status_code,
            request=self.request,
            text=self._text,
        )
        response.raise_for_status()

    async def aclose(self):
        self.is_closed = True

    async def aiter_bytes(self, chunk_size: int = 65_536):
        data = self._text.encode("utf-8")
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]


def _lines_from_events(*events: str) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines.extend(event.splitlines())
    return lines


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test-key",
        base_url="https://custom.test/v1/",
        proxy="socks5://127.0.0.1:9999",
        rate_limit=10,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
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


def test_init_configures_httpx_client(provider_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = NativeProvider(provider_config)

    assert provider._provider_name == "TEST_NATIVE"
    assert provider._api_key == "test-key"
    assert provider._base_url == "https://custom.test/v1"
    kwargs = mock_client.call_args.kwargs
    timeout = kwargs["timeout"]
    assert kwargs["base_url"] == "https://custom.test/v1"
    assert kwargs["proxy"] == "socks5://127.0.0.1:9999"
    assert timeout.read == 600.0
    assert timeout.write == 15.0
    assert timeout.connect == 5.0


def test_default_request_body_strips_internal_fields(provider_config):
    provider = NativeProvider(provider_config)

    body = provider._build_request_body(MockRequest())

    assert body["model"] == "test-model"
    assert body["thinking"] == {"type": "enabled"}
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert "extra_body" not in body


def test_default_request_body_preserves_thinking_budget(provider_config):
    provider = NativeProvider(provider_config)
    req = MockRequest(
        body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )

    body = provider._build_request_body(req)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}


@pytest.mark.asyncio
async def test_stream_uses_retry_builds_request_and_closes_response(
    provider_config,
    mock_rate_limiter,
):
    provider = NativeProvider(provider_config)
    req = MockRequest()
    request_obj = httpx.Request("POST", "https://custom.test/v1/messages")
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
            provider._client, "build_request", return_value=request_obj
        ) as mock_build,
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_send,
    ):
        events = [event async for event in provider.stream_response(req)]

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
    assert mock_build.call_args.kwargs["headers"] == {
        "Content-Type": "application/json",
        "X-Test": "1",
    }
    assert mock_build.call_args.kwargs["json"]["thinking"] == {"type": "enabled"}
    mock_send.assert_awaited_once_with(request_obj, stream=True)
    mock_rate_limiter.execute_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_maps_non_200_to_error_event_and_closes_response(
    provider_config,
):
    provider = NativeProvider(provider_config)
    req = MockRequest()
    response = FakeResponse(status_code=500, text="Internal Server Error")

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [
            event async for event in provider.stream_response(req, request_id="REQ_123")
        ]

    assert response.is_closed
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Upstream provider TEST_NATIVE returned HTTP 500."
    )
    blob = "".join(events)
    assert "Internal Server Error" in blob
    assert "REQ_123" in blob


@pytest.mark.asyncio
async def test_midstream_error_closes_open_block_and_uses_fresh_content_index(
    provider_config,
):
    """After upstream message_start + content_block_start, synthetic errors must not reuse index 0."""
    provider = NativeProvider(provider_config)
    req = MockRequest()
    mid = "msg_midstream_err"
    msg_start = format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    block_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    lines: list[str] = []
    for blob in (msg_start, block_start):
        lines.extend(blob.splitlines())
    response = FakeResponse(lines=lines, raise_after_line_index=len(lines) - 1)

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in provider.stream_response(req)]

    assert_canonical_stream_error_envelope(
        events, user_message_substr="mid-stream failure"
    )
    parsed = parse_sse_text("".join(events))
    starts = [e for e in parsed if e.event == "content_block_start"]
    assert event_index(starts[0]) == 0
    assert event_index(starts[-1]) == 1
    assert {event_index(e) for e in parsed if e.event == "content_block_stop"} == {0, 1}


@pytest.mark.asyncio
async def test_clean_eof_after_complete_native_tool_call_salvages_tool_use(
    provider_config,
):
    """Native stream EOF after complete tool args gets a deterministic tool_use tail."""
    provider = NativeProvider(provider_config)
    req = MockRequest()
    msg_start = format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool_eof",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    block_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_eof",
                "name": "echo_smoke",
                "input": {},
            },
        },
    )
    args = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
    )
    lines: list[str] = []
    for blob in (msg_start, block_start, args):
        lines.extend(blob.splitlines())
    response = FakeResponse(lines=lines)

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in provider.stream_response(req)]

    parsed = parse_sse_text("".join(events))
    assert parsed[-1].event == "message_stop"
    assert any(
        event.event == "message_delta"
        and event.data.get("delta", {}).get("stop_reason") == "tool_use"
        for event in parsed
    )
    assert not any(event.event == "error" for event in parsed)


@pytest.mark.asyncio
async def test_clean_eof_after_native_text_continues_with_overlap_trim(
    provider_config,
):
    """Native text truncation is continued and overlap-trimmed."""
    provider = NativeProvider(provider_config)
    req = MockRequest()
    msg_start = format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_text_eof",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    block_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    text_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello wor"},
        },
    )
    lines: list[str] = []
    for blob in (msg_start, block_start, text_delta):
        lines.extend(blob.splitlines())
    response = FakeResponse(lines=lines)

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        patch.object(
            provider,
            "_collect_native_recovery_text",
            new_callable=AsyncMock,
            return_value=("world", ""),
        ),
    ):
        events = [e async for e in provider.stream_response(req)]

    parsed = parse_sse_text("".join(events))
    text_deltas = [
        event.data.get("delta", {}).get("text", "")
        for event in parsed
        if event.event == "content_block_delta"
    ]
    assert text_deltas == ["hello wor", "ld"]
    assert "".join(text_deltas) == "hello world"
    assert any(
        event.event == "message_delta"
        and event.data.get("delta", {}).get("stop_reason") == "end_turn"
        for event in parsed
    )
    assert not any(event.event == "error" for event in parsed)


@pytest.mark.asyncio
async def test_precommit_native_holdback_retries_without_leaking_partial(
    provider_config,
):
    """A retryable early cutoff before holdback commit is retried invisibly."""
    provider = NativeProvider(provider_config)
    req = MockRequest()

    msg_start = format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_holdback",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    block_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    hidden_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hidden"},
        },
    )
    visible_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "visible"},
        },
    )
    block_stop = format_sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    )
    message_delta = format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    message_stop = format_sse_event("message_stop", {"type": "message_stop"})
    first_lines = _lines_from_events(msg_start, block_start, hidden_delta)
    first = FakeResponse(
        lines=first_lines,
        raise_after_line_index=len(first_lines) - 1,
        raise_error=httpx.ReadError("early cutoff"),
    )
    second = FakeResponse(
        lines=_lines_from_events(
            msg_start,
            block_start,
            visible_delta,
            block_stop,
            message_delta,
            message_stop,
        ),
    )

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=[first, second],
        ) as mock_send,
    ):
        events = [e async for e in provider.stream_response(req)]

    event_text = "".join(events)
    assert mock_send.await_count == 2
    assert "hidden" not in event_text
    assert "visible" in event_text
    assert parse_sse_text(event_text)[-1].event == "message_stop"
