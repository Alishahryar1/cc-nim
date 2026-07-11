"""Tests for the shared native Anthropic Messages transport."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    format_sse_event,
)
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.stream_recovery import (
    EARLY_TRANSPARENT_TOTAL_ATTEMPTS,
    MIDSTREAM_RECOVERY_ATTEMPTS,
    TruncatedProviderStreamError,
)
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.transports.anthropic_messages.transport import (
    _AnthropicMessagesStreamRunner,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter


class NativeProvider(AnthropicMessagesTransport):
    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="TEST_NATIVE",
            default_base_url="https://example.test/v1",
            rate_limiter=rate_limiter,
        )

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "X-Test": "1"}


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        lines=None,
        text="",
        raise_after_line_index: int | None = None,
        raise_error: Exception | None = None,
        close_error: Exception | None = None,
    ):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self._raise_after_line_index = raise_after_line_index
        self._raise_error = raise_error or RuntimeError("mid-stream failure")
        self._close_error = close_error
        self.close_calls = 0
        self.line_iterator_close_calls = 0
        self.is_closed = False
        self.request = httpx.Request("POST", "https://example.test/v1/messages")
        self.headers = httpx.Headers()

    async def aiter_lines(self):
        try:
            for i, line in enumerate(self._lines):
                yield line
                if (
                    self._raise_after_line_index is not None
                    and i >= self._raise_after_line_index
                ):
                    raise self._raise_error
        finally:
            self.line_iterator_close_calls += 1

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
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error
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


@pytest.fixture
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    instance = MagicMock(spec=ProviderRateLimiter)

    async def _passthrough(fn, *args, **kwargs):
        return await fn(*args, **kwargs)

    instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
    instance.concurrency_slot.side_effect = _slot
    yield instance


def test_init_configures_httpx_client(provider_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = NativeProvider(
            provider_config,
            rate_limiter=passthrough_rate_limiter(),
        )

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
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )

    body = provider._build_request_body(make_messages_request(max_tokens=None))

    assert body["model"] == "test-model"
    assert body["thinking"] == {"type": "enabled"}
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert "extra_body" not in body


def test_default_request_body_preserves_thinking_budget(provider_config):
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request(
        max_tokens=None,
        thinking={"type": "enabled", "budget_tokens": 4096},
    )

    body = provider._build_request_body(req)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}


@pytest.mark.asyncio
async def test_send_stream_request_forces_upstream_streaming(provider_config):
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    request_obj = httpx.Request("POST", "https://custom.test/v1/messages")
    response = FakeResponse()
    body = {"model": "test-model", "stream": False}

    with (
        patch.object(
            provider._client, "build_request", return_value=request_obj
        ) as mock_build,
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        await provider._send_stream_request(body)

    assert body["stream"] is False
    assert mock_build.call_args.kwargs["json"]["stream"] is True


@pytest.mark.asyncio
async def test_stream_uses_retry_builds_request_and_closes_response(
    provider_config,
    mock_rate_limiter,
):
    provider = NativeProvider(provider_config, rate_limiter=mock_rate_limiter)
    req = make_messages_request()
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

    assert [event.event for event in parse_sse_text("".join(events))] == [
        "message_start",
        "message_stop",
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
async def test_closing_public_native_stream_closes_response_once(
    provider_config,
    mock_rate_limiter,
):
    provider = NativeProvider(provider_config, rate_limiter=mock_rate_limiter)
    request = make_messages_request()
    response = FakeResponse(
        lines=_lines_from_events(
            format_sse_event(
                "message_start",
                {"type": "message_start", "message": {}},
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "x" * 65_536},
                },
            ),
        )
    )

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        stream = provider.stream_response(request, request_id="req_native_early_close")
        await anext(stream)
        assert isinstance(stream, AsyncCloseable)
        await stream.aclose()

    assert response.close_calls == 1
    assert response.line_iterator_close_calls == 1


@pytest.mark.asyncio
async def test_late_error_after_native_message_stop_keeps_successful_lifecycle(
    provider_config,
    mock_rate_limiter,
):
    provider = NativeProvider(provider_config, rate_limiter=mock_rate_limiter)
    req = make_messages_request()
    lines = [
        "event: message_start",
        'data: {"type":"message_start"}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    response = FakeResponse(
        lines=lines,
        raise_after_line_index=len(lines) - 1,
        raise_error=httpx.ReadError(
            "late cleanup failure",
            request=httpx.Request("POST", "https://example.test/v1/messages"),
        ),
    )

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [event async for event in provider.stream_response(req)]

    assert [event.event for event in parse_sse_text("".join(events))] == [
        "message_start",
        "message_stop",
    ]
    assert response.is_closed
    mock_rate_limiter.execute_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_maps_pre_start_non_200_to_provider_error_and_closes_response(
    provider_config,
):
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
    response = FakeResponse(status_code=500, text="Internal Server Error")

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_response(req, request_id="REQ_123")]

    assert response.is_closed
    assert "Upstream provider TEST_NATIVE returned HTTP 500." in exc_info.value.message
    assert "Internal Server Error" in exc_info.value.message
    assert "REQ_123" in exc_info.value.message


@pytest.mark.asyncio
async def test_native_http_close_failure_cannot_mask_mapped_status(provider_config):
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
    response = FakeResponse(
        status_code=400,
        text="original native HTTP body",
        close_error=RuntimeError("cleanup api_key=SECRET"),
    )

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_response(
                req,
                request_id="req_native_http_close_failure",
            )
        ]

    assert response.close_calls == 1
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400
    assert "original native HTTP body" in exc_info.value.message
    assert "cleanup" not in exc_info.value.message
    assert "SECRET" not in exc_info.value.message


@pytest.mark.asyncio
async def test_precommit_native_error_raises_without_leaking_open_block(
    provider_config,
):
    """A native error before holdback commit raises instead of sending HTTP 200 SSE."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
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
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [e async for e in provider.stream_response(req)]

    assert "mid-stream failure" in exc_info.value.message


@pytest.mark.asyncio
async def test_native_upstream_error_event_retries_then_raises_canonical_failure(
    provider_config,
):
    """Native wire errors become one semantic failure instead of leaking or masking."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
    upstream_error = format_sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "native quota exhausted api_key=SECRET useful detail",
            },
        },
    )
    responses = [
        FakeResponse(lines=_lines_from_events(upstream_error))
        for _ in range(EARLY_TRANSPARENT_TOTAL_ATTEMPTS)
    ]
    events: list[str] = []

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=responses,
        ) as mock_send,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        async for event in provider.stream_response(
            req,
            request_id="req_native_error",
        ):
            events.extend((event,))

    failure = exc_info.value
    assert mock_send.await_count == EARLY_TRANSPARENT_TOTAL_ATTEMPTS
    assert all(response.is_closed for response in responses)
    assert events == []
    assert failure.kind is FailureKind.RATE_LIMIT
    assert failure.status_code == 429
    assert failure.retryable
    assert "native quota exhausted" in failure.message
    assert "useful detail" in failure.message
    assert "api_key=<redacted>" in failure.message
    assert "SECRET" not in failure.message
    assert "Request ID: req_native_error" in failure.message


@pytest.mark.asyncio
async def test_native_stream_close_failure_cannot_mask_execution_failure(
    provider_config,
):
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
    upstream_error = format_sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "original native provider failure",
            },
        },
    )
    response = FakeResponse(
        lines=_lines_from_events(upstream_error),
        close_error=RuntimeError("cleanup api_key=SECRET"),
    )

    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_response(
                req,
                request_id="req_native_close_failure",
            )
        ]

    assert response.close_calls == 1
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400
    assert "original native provider failure" in exc_info.value.message
    assert "cleanup" not in exc_info.value.message
    assert "SECRET" not in exc_info.value.message


@pytest.mark.asyncio
async def test_midstream_error_after_native_message_delta_raises_without_wire_terminal(
    provider_config,
):
    """Providers preserve the committed prefix and leave terminal wire errors to API."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
    msg_start = format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_terminal_cutoff",
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
            "delta": {"type": "text_delta", "text": "hello" + ("x" * 70_000)},
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
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )
    response = FakeResponse(
        lines=_lines_from_events(
            msg_start, block_start, text_delta, block_stop, message_delta
        )
    )

    events: list[str] = []
    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
        patch.object(
            _AnthropicMessagesStreamRunner,
            "_collect_recovery_text",
            new_callable=AsyncMock,
            return_value=("hello recovered", ""),
        ) as mock_collect,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        async for event in provider.stream_response(req):
            events.extend((event,))

    parsed = parse_sse_text("".join(events))
    assert mock_collect.await_count == 0
    assert "Provider stream ended without message_stop." in exc_info.value.message
    assert sum(event.event == "message_delta" for event in parsed) == 1
    assert sum(event.event == "message_stop" for event in parsed) == 0
    assert sum(event.event == "error" for event in parsed) == 0
    message_delta_index = next(
        index for index, event in enumerate(parsed) if event.event == "message_delta"
    )
    assert all(
        event.event
        not in {"content_block_start", "content_block_delta", "content_block_stop"}
        for event in parsed[message_delta_index + 1 :]
    )


@pytest.mark.asyncio
async def test_native_text_recovery_closes_thinking_before_text_suffix(
    provider_config,
):
    """Recovery suffixes preserve Anthropic block ordering when switching types."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    runner = _AnthropicMessagesStreamRunner(
        provider,
        request=make_messages_request(),
        input_tokens=0,
        request_id="req_native_recovery",
        thinking_enabled=True,
    )
    ledger = AnthropicStreamLedger("msg_recovery", "test-model")
    ledger.start_thinking_block()
    ledger.emit_thinking_delta("thinking")

    with patch.object(
        runner,
        "_collect_recovery_text",
        new_callable=AsyncMock,
        return_value=("answer", "thinking more"),
    ) as mock_collect:
        events = await runner._recovery_events(
            body={"messages": []},
            ledger=ledger,
            error=TimeoutError("cutoff"),
            req_tag="",
            thinking_enabled=True,
        )

    assert events is not None
    assert mock_collect.await_args is not None
    recovery_body = mock_collect.await_args.args[0]
    assert "thinking" in recovery_body["messages"][-1]["content"]
    parsed = parse_sse_text("".join(events))
    assert [event.event for event in parsed] == [
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert parsed[0].data["index"] == 0
    assert parsed[1].data["index"] == 0
    assert parsed[2].data["index"] == 1
    assert parsed[2].data["content_block"]["type"] == "text"
    assert parsed[3].data["index"] == 1


@pytest.mark.asyncio
async def test_clean_eof_after_complete_native_tool_call_salvages_tool_use(
    provider_config,
):
    """Native stream EOF after complete tool args gets a deterministic tool_use tail."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
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
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
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
            _AnthropicMessagesStreamRunner,
            "_collect_recovery_text",
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
async def test_native_recovery_collect_text_requires_message_stop(provider_config):
    """Native recovery collectors reject truncated continuation streams."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    text_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
    )

    async def _iter_chunks(_response, *, state, thinking_enabled):
        yield text_delta

    runner = _AnthropicMessagesStreamRunner(
        provider,
        request=make_messages_request(),
        input_tokens=0,
        request_id=None,
        thinking_enabled=True,
    )

    with (
        patch.object(runner, "_iter_stream_chunks", _iter_chunks),
        patch.object(
            provider,
            "_validated_stream_send",
            new_callable=AsyncMock,
            return_value=FakeResponse(),
        ) as mock_send,
        pytest.raises(TruncatedProviderStreamError),
    ):
        await runner._collect_recovery_text(
            {"messages": []},
            req_tag="",
            thinking_enabled=True,
        )

    assert mock_send.await_count == MIDSTREAM_RECOVERY_ATTEMPTS


@pytest.mark.asyncio
async def test_native_recovery_collect_text_accepts_message_stop(provider_config):
    """Native recovery collectors return text only after message_stop."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    text_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
    )
    message_stop = format_sse_event("message_stop", {"type": "message_stop"})

    async def _iter_chunks(_response, *, state, thinking_enabled):
        yield text_delta
        yield message_stop

    runner = _AnthropicMessagesStreamRunner(
        provider,
        request=make_messages_request(),
        input_tokens=0,
        request_id=None,
        thinking_enabled=True,
    )

    with (
        patch.object(runner, "_iter_stream_chunks", _iter_chunks),
        patch.object(
            provider,
            "_validated_stream_send",
            new_callable=AsyncMock,
            return_value=FakeResponse(),
        ),
    ):
        result = await runner._collect_recovery_text(
            {"messages": []}, req_tag="", thinking_enabled=True
        )

    assert result == ("world", "")


@pytest.mark.asyncio
async def test_native_recovery_collect_text_reads_eager_start_content(provider_config):
    """Native recovery reads text/thinking carried on content_block_start."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    text_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "hello"},
        },
    )
    thinking_start = format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": "step"},
        },
    )
    text_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
    )
    thinking_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": " two"},
        },
    )
    message_stop = format_sse_event("message_stop", {"type": "message_stop"})

    async def _iter_chunks(_response, *, state, thinking_enabled):
        yield text_start
        yield thinking_start
        yield text_delta
        yield thinking_delta
        yield message_stop

    runner = _AnthropicMessagesStreamRunner(
        provider,
        request=make_messages_request(),
        input_tokens=0,
        request_id=None,
        thinking_enabled=True,
    )

    with (
        patch.object(runner, "_iter_stream_chunks", _iter_chunks),
        patch.object(
            provider,
            "_validated_stream_send",
            new_callable=AsyncMock,
            return_value=FakeResponse(),
        ),
    ):
        result = await runner._collect_recovery_text(
            {"messages": []}, req_tag="", thinking_enabled=True
        )

    assert result == ("hello world", "step two")


@pytest.mark.asyncio
async def test_truncated_native_recovery_stream_raises_after_closing_block(
    provider_config,
):
    """Partial native recovery bytes are not converted into a success tail."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()
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
    original_text = "hello wor" + ("x" * 70_000)
    original_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": original_text},
        },
    )
    recovery_delta = format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
    )
    original = FakeResponse(
        lines=_lines_from_events(msg_start, block_start, original_delta)
    )
    recovery_responses = [
        FakeResponse(lines=_lines_from_events(recovery_delta))
        for _ in range(MIDSTREAM_RECOVERY_ATTEMPTS)
    ]

    events: list[str] = []
    with (
        patch.object(provider._client, "build_request", return_value=MagicMock()),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=[original, *recovery_responses],
        ) as mock_send,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        async for event in provider.stream_response(req):
            events.extend((event,))

    event_text = "".join(events)
    assert mock_send.await_count == 1 + MIDSTREAM_RECOVERY_ATTEMPTS
    assert original_text in event_text
    assert "world" not in event_text
    assert "Provider stream ended without message_stop." in exc_info.value.message
    assert "Provider stream ended without message_stop." not in event_text
    parsed = parse_sse_text(event_text)
    assert parsed[-1].event == "content_block_stop"
    assert not any(event.event == "error" for event in parsed)
    assert not any(event.event == "message_stop" for event in parsed)
    assert not any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("text") == "ld"
        for event in parse_sse_text(event_text)
    )


@pytest.mark.asyncio
async def test_precommit_native_holdback_retries_without_leaking_partial(
    provider_config,
):
    """A retryable early cutoff before holdback commit is retried invisibly."""
    provider = NativeProvider(
        provider_config,
        rate_limiter=passthrough_rate_limiter(),
    )
    req = make_messages_request()

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
