"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
)
from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    make_text_recovery_body,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.openai_chat.provider import (
    _OpenAIChatStreamRunner,
)
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
    has_committed_sse_output,
    iter_heuristic_tool_use_sse,
)
from free_claude_code.providers.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    TruncatedProviderStreamError,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter


class AsyncStreamMock:
    """Async iterable mock that yields chunks then optionally raises."""

    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk
        if self._error:
            raise self._error


class ClosableAsyncStreamMock(AsyncStreamMock):
    """Async stream mock that records cleanup."""

    def __init__(self, chunks, error=None):
        super().__init__(chunks, error=error)
        self.closed = False

    async def aclose(self):
        self.closed = True


def _make_provider():
    """Create a provider instance for testing."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
    )
    return NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        rate_limiter=passthrough_rate_limiter(),
    )


def _make_tool_assembler(provider: NvidiaNimProvider) -> OpenAIToolCallAssembler:
    return OpenAIToolCallAssembler(
        record_extra_content=provider._record_tool_call_extra_content
    )


def _make_provider_with_thinking_enabled(enabled: bool):
    """Create a provider instance with thinking explicitly enabled or disabled."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
        enable_thinking=enabled,
    )
    return NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        rate_limiter=passthrough_rate_limiter(),
    )


def _make_request(model: str = "test-model", stream: bool = True, **overrides: object):
    """Create a concrete request matching the original streaming-test defaults."""
    request_overrides: dict[str, object] = {
        "messages": [],
        "max_tokens": 4096,
        "temperature": None,
        "top_p": None,
        "system": None,
        "tools": None,
        "extra_body": None,
        "thinking": None,
        "stream": stream,
    }
    request_overrides.update(overrides)
    return make_messages_request(model, **request_overrides)


def _make_stream_runner(
    provider: NvidiaNimProvider,
    *,
    request=None,
    request_id: str | None = None,
) -> _OpenAIChatStreamRunner:
    return _OpenAIChatStreamRunner(
        provider,
        request=request or _make_request(),
        input_tokens=0,
        request_id=request_id,
        thinking_enabled=None,
    )


def _make_chunk(
    content=None, finish_reason=None, tool_calls=None, reasoning_content=None
):
    """Create a mock streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.reasoning_content = reasoning_content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_tool_calls_chunk(*, name: str, arguments: str, tool_id: str, index: int = 0):
    """Single OpenAI-style tool_calls delta (starts a native streamed tool block)."""
    tc = MagicMock()
    tc.index = index
    tc.id = tool_id
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc.function = fn
    return _make_chunk(tool_calls=[tc])


async def _collect_stream(provider, request):
    """Collect all SSE events from a stream."""
    return [e async for e in provider.stream_response(request)]


async def _collect_stream_error(provider, request, **kwargs) -> ExecutionFailure:
    with pytest.raises(ExecutionFailure) as exc_info:
        [e async for e in provider.stream_response(request, **kwargs)]
    return exc_info.value


async def _collect_stream_and_error(
    provider, request, **kwargs
) -> tuple[list[str], ExecutionFailure]:
    events: list[str] = []
    with pytest.raises(ExecutionFailure) as exc_info:
        async for event in provider.stream_response(request, **kwargs):
            events.extend((event,))
    return events, exc_info.value


def _assert_no_content_deltas_after_error_text(
    events: list[str], error_substr: str
) -> None:
    """After the error text delta, only block close + message tail events may follow."""
    parsed = parse_sse_text("".join(events))
    first_error_idx = None
    for i, ev in enumerate(parsed):
        if ev.event != "content_block_delta":
            continue
        delta = ev.data.get("delta", {})
        if delta.get("type") == "text_delta" and error_substr in str(
            delta.get("text", "")
        ):
            first_error_idx = i
            break
    assert first_error_idx is not None, (error_substr, "".join(events))
    for ev in parsed[first_error_idx + 1 :]:
        assert ev.event in ("content_block_stop", "message_delta", "message_stop"), (
            ev.event,
            ev.data,
        )


def _assert_error_not_in_text_deltas_after_tool(
    events: list[str], error_substr: str
) -> None:
    """Transport errors after a native tool call must not use assistant text_delta (issue #206)."""
    blob = "".join(events)
    for ev in parse_sse_text(blob):
        if ev.event != "content_block_delta":
            continue
        delta = ev.data.get("delta", {})
        if delta.get("type") == "text_delta" and error_substr in str(
            delta.get("text", "")
        ):
            raise AssertionError(
                f"error leaked as text_delta after tool_use: {ev.data!r} full={blob!r}"
            )


class TestStreamingExceptionHandling:
    """Tests for error paths during stream_response."""

    @pytest.mark.asyncio
    async def test_pre_start_api_error_raises_provider_error(self):
        """Before holdback commit, provider failures raise for API-level non-200."""
        provider = _make_provider()
        request = _make_request()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API failed"),
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "API failed" in error.message

    @pytest.mark.asyncio
    async def test_read_timeout_with_empty_message_raises_fallback(self):
        """ReadTimeout(TimeoutError()) should raise a non-empty timeout message."""
        provider = _make_provider()
        request = _make_request()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout(""),
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            error = await _collect_stream_error(
                provider,
                request,
                request_id="req_timeout123",
            )

        assert "timed out after" in error.message
        assert "Request ID: req_timeout123" in error.message

    @pytest.mark.asyncio
    async def test_error_after_precommit_partial_content_raises(self):
        """Precommit partial text is discarded so the API can return non-200."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Hello ")
        stream_mock = AsyncStreamMock([chunk1], error=RuntimeError("Connection lost"))

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "Connection lost" in error.message

    @pytest.mark.asyncio
    async def test_error_after_native_tool_call_closes_block_then_raises(self):
        """A provider closes tool state, then leaves terminal serialization to API."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments="{}", tool_id="call_206", index=0
        )
        stream_mock = AsyncStreamMock(
            [tool_chunk], error=RuntimeError("Connection lost after tool")
        )
        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events, error = await _collect_stream_and_error(provider, request)
        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert "tool_use" in event_text
        assert parsed[-1].event == "content_block_stop"
        assert "Connection lost after tool" in error.message
        assert "Connection lost after tool" not in event_text
        assert "event: error\n" not in event_text
        assert "message_stop" not in event_text
        _assert_error_not_in_text_deltas_after_tool(
            events, "Connection lost after tool"
        )

    @pytest.mark.asyncio
    async def test_empty_response_gets_space(self):
        """Empty response with no text/tools gets a single space text block."""
        provider = _make_provider()
        request = _make_request()

        empty_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_upstream_completion_tokens_null_emits_int_usage(self):
        """NIM/GLM may send usage.completion_tokens=null; final SSE must not use JSON null."""
        provider = _make_provider()
        request = _make_request()

        delta = SimpleNamespace(
            content="hello",
            tool_calls=None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(delta=delta, finish_reason="stop")
        usage = SimpleNamespace(completion_tokens=None, prompt_tokens=None)
        chunk = SimpleNamespace(choices=[choice], usage=usage)
        stream_mock = AsyncStreamMock([chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        delta_events = [e for e in parsed if e.event == "message_delta"]
        assert len(delta_events) == 1
        usage_out = delta_events[0].data.get("usage", {})
        assert isinstance(usage_out.get("output_tokens"), int)
        assert usage_out["output_tokens"] is not None
        assert '"output_tokens": null' not in "".join(events)

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_emits_placeholder_text(self):
        """When the model streams only ``reasoning_content`` (no ``content``), add text block.

        NIM / some templates may emit no main ``content``; a minimal text block matches
        the empty-body placeholder and helps clients that expect a text segment.
        """
        provider = _make_provider_with_thinking_enabled(True)
        request = _make_request()
        chunk1 = _make_chunk(reasoning_content="reasoning only from provider")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])
        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)
        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_thinking_content(self):
        """Thinking content via think tags is emitted as thinking blocks."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="<think>reasoning</think>answer")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking" in event_text
        assert "reasoning" in event_text
        assert "answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content_field(self):
        """reasoning_content delta field is emitted as thinking block."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="I think...")
        chunk2 = _make_chunk(content="The answer")
        chunk3 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2, chunk3])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert "I think..." in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_empty_reasoning_content_starts_thinking_block_only(self):
        """Empty reasoning_content is stateful but must not emit visible thinking text."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        thinking_starts = [
            event
            for event in parsed
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "thinking"
        ]
        thinking_deltas = [
            event
            for event in parsed
            if event.event == "content_block_delta"
            and event.data["delta"]["type"] == "thinking_delta"
        ]
        assert len(thinking_starts) == 1
        assert thinking_deltas == []
        assert parsed[-1].event == "message_stop"

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content_suppressed_when_disabled(self):
        """reasoning deltas are stripped while normal text still streams."""
        provider = _make_provider_with_thinking_enabled(False)
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="I think...")
        chunk2 = _make_chunk(content="<think>secret</think>The answer")
        chunk3 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2, chunk3])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" not in event_text
        assert "I think..." not in event_text
        assert "secret" not in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_upstream_405_mentions_provider_name(self):
        """HTTP 405s are surfaced as upstream method/endpoint rejections."""
        provider = _make_provider()
        request = _make_request()

        response = httpx.Response(
            status_code=405,
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )
        error = httpx.HTTPStatusError(
            "Method Not Allowed",
            request=response.request,
            response=response,
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            stream_error = await _collect_stream_error(
                provider,
                request,
                request_id="REQ405",
            )

        assert (
            "Upstream provider NIM rejected the request method or endpoint (HTTP 405)."
            in stream_error.message
        )
        assert "Request ID: REQ405" in stream_error.message

    @pytest.mark.asyncio
    async def test_stream_with_openai_bad_request_surfaces_upstream_body(self):
        """OpenAI SDK bodies should be raised so users can copy exact provider errors."""
        provider = _make_provider()
        request = _make_request()
        response = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )
        body = {
            "error": {
                "type": "BadRequest",
                "message": "Thinking mode does not support this tool_choice",
            }
        }
        error = openai.BadRequestError("Bad Request", response=response, body=body)

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            stream_error = await _collect_stream_error(
                provider,
                request,
                request_id="REQ_BODY",
            )

        assert "Upstream provider NIM returned HTTP 400." in stream_error.message
        assert "Category: BadRequest" in stream_error.message
        assert "Thinking mode does not support this tool_choice" in stream_error.message
        assert (
            '{"error":{"type":"BadRequest","message":"Thinking mode does not support this tool_choice"}}'
            in stream_error.message
        )
        assert "Request ID: REQ_BODY" in stream_error.message

    @pytest.mark.asyncio
    async def test_error_after_native_tool_call_failure_includes_body(self):
        """Detailed failure data survives after the provider closes tool state."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments="{}", tool_id="call_body", index=0
        )
        response = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )
        body = {"error": {"message": "bad after tool"}}
        error = openai.BadRequestError("Bad Request", response=response, body=body)
        stream_mock = AsyncStreamMock([tool_chunk], error=error)

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events, stream_error = await _collect_stream_and_error(
                provider,
                request,
                request_id="REQ_TOOL_BODY",
            )

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert "tool_use" in event_text
        assert parsed[-1].event == "content_block_stop"
        assert "event: error\n" not in event_text
        assert "bad after tool" not in event_text
        assert "Request ID: REQ_TOOL_BODY" not in event_text
        assert "message_stop" not in event_text
        assert "bad after tool" in stream_error.message
        assert "Request ID: REQ_TOOL_BODY" in stream_error.message
        _assert_error_not_in_text_deltas_after_tool(events, "bad after tool")

    @pytest.mark.asyncio
    async def test_clean_eof_after_complete_tool_call_salvages_tool_use(self):
        """A complete tool JSON payload missing finish_reason is committed as tool_use."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments='{"message":"ok"}', tool_id="call_eof"
        )
        stream_mock = AsyncStreamMock([tool_chunk])

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        assert parsed[-1].event == "message_stop"
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )
        assert not any(event.event == "error" for event in parsed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
    async def test_heuristic_only_tool_stream_does_not_emit_fallback_text(
        self, finish_reason
    ):
        """Text-parsed tool calls count as emitted tool output when finalizing."""
        provider = _make_provider()
        request = _make_request()
        heuristic_tool = (
            "● <function=Read><parameter=path>test.py</parameter>"
            "<parameter=limit>10</parameter>"
        )
        stream_mock = AsyncStreamMock(
            [
                _make_chunk(content=heuristic_tool),
                _make_chunk(finish_reason=finish_reason),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        assert any(
            event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
            for event in parsed
        )
        assert not any(
            event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
            and event.data.get("delta", {}).get("text") == " "
            for event in parsed
        )
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )

    @pytest.mark.asyncio
    async def test_precommit_openai_holdback_retries_without_leaking_partial(self):
        """A retryable early cutoff before holdback commit is retried invisibly."""
        provider = _make_provider()
        request = _make_request()
        first_stream = AsyncStreamMock(
            [_make_chunk(content="hidden")],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [
                _make_chunk(content="visible"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as mock_create:
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert mock_create.await_count == 2
        assert "hidden" not in event_text
        assert "visible" in event_text
        parsed = parse_sse_text(event_text)
        assert parsed[0].event == "message_start"
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert parsed[-1].event == "message_stop"

    @pytest.mark.asyncio
    async def test_clean_eof_after_text_continues_with_overlap_trim(self):
        """A truncated text stream is continued and duplicate overlap is trimmed."""
        provider = _make_provider()
        request = _make_request()
        stream_mock = AsyncStreamMock([_make_chunk(content="hello wor")])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_text",
                new_callable=AsyncMock,
                return_value=("world", ""),
            ),
        ):
            events = await _collect_stream(provider, request)

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
    async def test_disabled_thinking_recovery_discards_reasoning(self):
        provider = _make_provider_with_thinking_enabled(False)
        request = _make_request()
        initial_stream = AsyncStreamMock([_make_chunk(content="hello")])
        recovery_stream = AsyncStreamMock(
            [
                _make_chunk(reasoning_content="hidden reasoning"),
                _make_chunk(content="hello world"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[initial_stream, recovery_stream],
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
        )
        assert text == "hello world"
        assert "hidden reasoning" not in "".join(events)
        assert not any(
            event.data.get("delta", {}).get("type") == "thinking_delta"
            for event in parsed
        )

    @pytest.mark.asyncio
    async def test_recovery_collect_text_requires_finish_reason(self):
        """Recovery collectors reject truncated OpenAI-chat continuation streams."""
        streams = [
            ClosableAsyncStreamMock([_make_chunk(content=f"world {index}")])
            for index in range(MIDSTREAM_RECOVERY_ATTEMPTS)
        ]
        create_stream = AsyncMock(side_effect=[(stream, {}) for stream in streams])
        provider = _make_provider()
        runner = _make_stream_runner(provider)

        with (
            patch.object(provider, "_create_stream", create_stream),
            pytest.raises(TruncatedProviderStreamError),
        ):
            await runner._collect_recovery_text(
                {"messages": []}, include_reasoning=True
            )

        assert create_stream.await_count == MIDSTREAM_RECOVERY_ATTEMPTS
        assert all(stream.closed for stream in streams)

    @pytest.mark.asyncio
    async def test_recovery_collect_text_closes_retryable_failed_streams(self):
        """Recovery collectors close failed stream attempts before retrying."""
        streams = [
            ClosableAsyncStreamMock(
                [_make_chunk(content=f"partial {index}")],
                error=TimeoutError("recovery cutoff"),
            )
            for index in range(MIDSTREAM_RECOVERY_ATTEMPTS)
        ]
        create_stream = AsyncMock(side_effect=[(stream, {}) for stream in streams])
        provider = _make_provider()
        runner = _make_stream_runner(provider)

        with (
            patch.object(provider, "_create_stream", create_stream),
            pytest.raises(TimeoutError),
        ):
            await runner._collect_recovery_text(
                {"messages": []}, include_reasoning=True
            )

        assert create_stream.await_count == MIDSTREAM_RECOVERY_ATTEMPTS
        assert all(stream.closed for stream in streams)

    @pytest.mark.asyncio
    async def test_recovery_collect_text_accepts_finish_reason(self):
        """Recovery collectors return text only after the upstream terminal marker."""
        stream = ClosableAsyncStreamMock(
            [
                _make_chunk(content="world"),
                _make_chunk(finish_reason="stop"),
            ]
        )
        create_stream = AsyncMock(
            return_value=(
                stream,
                {},
            )
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider)

        with patch.object(provider, "_create_stream", create_stream):
            result = await runner._collect_recovery_text(
                {"messages": []}, include_reasoning=True
            )

        assert result == ("world", "")
        assert stream.closed is True

    def test_text_recovery_body_preserves_thinking_context(self):
        """Continuation prompts include emitted thinking without provider-specific fields."""
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Read"}],
            "tool_choice": {"type": "auto"},
        }

        recovery_body = make_text_recovery_body(
            body,
            partial_text="visible answer",
            partial_thinking="hidden reasoning",
        )

        assert "tools" not in recovery_body
        assert "tool_choice" not in recovery_body
        assert recovery_body["messages"][-2] == {
            "role": "assistant",
            "content": "visible answer",
        }
        recovery_prompt = recovery_body["messages"][-1]
        assert recovery_prompt["role"] == "user"
        assert "hidden reasoning" in recovery_prompt["content"]
        assert "reasoning_content" not in recovery_prompt

    @pytest.mark.asyncio
    async def test_openai_text_recovery_passes_thinking_context(self):
        """OpenAI-chat recovery call sites seed emitted thinking in the prompt."""
        runner = _make_stream_runner(
            _make_provider(), request=_make_request(), request_id="req_recovery"
        )
        ledger = AnthropicStreamLedger("msg_recovery", "model")
        ledger.start_thinking_block()
        ledger.emit_thinking_delta("hidden reasoning")
        list(ledger.ensure_text_block())
        ledger.emit_text_delta("visible answer")

        with patch.object(
            runner,
            "_collect_recovery_text",
            new_callable=AsyncMock,
            return_value=("visible answer done", "hidden reasoning more"),
        ) as mock_collect:
            events = await runner._recovery_events(
                body={"messages": [{"role": "user", "content": "hello"}]},
                ledger=ledger,
                error=TimeoutError("cutoff"),
                tool_argument_alias_buffers={},
                thinking_enabled=True,
            )

        assert events is not None
        assert mock_collect.await_args is not None
        recovery_body = mock_collect.await_args.args[0]
        assert "hidden reasoning" in recovery_body["messages"][-1]["content"]
        assert mock_collect.await_args.kwargs["include_reasoning"] is True

    @pytest.mark.asyncio
    async def test_primary_stream_closes_when_iteration_fails(self):
        """OpenAI-chat main streams close after iterator failures."""
        provider = _make_provider()
        request = _make_request()
        stream = ClosableAsyncStreamMock(
            [_make_chunk(content="partial")],
            error=ValueError("provider stream failed"),
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            error = await _collect_stream_error(provider, request)

        assert stream.closed is True
        assert "provider stream failed" in error.message.lower()

    @pytest.mark.asyncio
    async def test_truncated_recovery_stream_closes_block_then_raises(self):
        """Partial recovery bytes never become success or provider-owned wire errors."""
        provider = _make_provider()
        request = _make_request()
        original_text = "hello wor" + ("x" * 70_000)
        original_stream = AsyncStreamMock([_make_chunk(content=original_text)])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=original_stream,
            ) as mock_create,
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_text",
                new_callable=AsyncMock,
                side_effect=TruncatedProviderStreamError(
                    "Recovery stream ended without finish_reason."
                ),
            ) as mock_collect,
        ):
            events, error = await _collect_stream_and_error(provider, request)

        event_text = "".join(events)
        assert mock_create.await_count == 1
        assert mock_collect.await_count == 1
        assert original_text in event_text
        assert "world" not in event_text
        assert "Provider stream ended without finish_reason." in error.message
        assert "Provider stream ended without finish_reason." not in event_text
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
    async def test_incomplete_tool_call_repair_appends_schema_valid_suffix(self):
        """A truncated tool JSON prefix is repaired append-only before tool_use tail."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "echo_smoke",
                    "description": "Echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ]
        )
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments='{"message":', tool_id="call_repair"
        )
        stream_mock = AsyncStreamMock([tool_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_text",
                new_callable=AsyncMock,
                return_value=('"ok"}', ""),
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert '"partial_json": "\\"ok\\"}"' in event_text
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )
        assert not any(event.event == "error" for event in parsed)

    @pytest.mark.asyncio
    async def test_stream_rate_limited_retries_via_execute_with_retry(self):
        """When rate limited, execute_with_retry handles retries transparently."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Response")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            # Mock execute_with_retry to pass through to the actual function
            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)

            with patch.object(
                provider._rate_limiter,
                "execute_with_retry",
                new_callable=AsyncMock,
                side_effect=_passthrough,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "Response" in event_text


class TestProcessToolCall:
    """Tests for OpenAI tool-call assembly."""

    def test_heuristic_tool_use_sse_marks_committed_tool_output(self):
        """Heuristic tool blocks are emitted content, even without OpenAI tool state."""
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        ledger = AnthropicStreamLedger("msg_test", "test-model")
        events = list(
            iter_heuristic_tool_use_sse(
                ledger,
                {
                    "id": "toolu_heuristic",
                    "name": "Read",
                    "input": {"path": "test.py"},
                },
            )
        )

        event_text = "".join(events)
        assert "tool_use" in event_text
        assert ledger.has_emitted_tool_block()
        assert has_committed_sse_output(ledger)

    def test_tool_call_with_id(self):
        """Tool call with id starts a tool block."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_use" in event_text
        assert "search" in event_text
        assert "call_123" in event_text

    def test_tool_call_id_arrives_before_name_still_emits_id_and_name(self):
        """Split-stream tool: id (no name) then name then args; id preserved on start."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        t1 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": None, "arguments": ""},
        }
        t2 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": "Grep", "arguments": ""},
        }
        t3 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": None, "arguments": "{}"},
        }
        b1 = "".join(_make_tool_assembler(provider).process_tool_call(t1, sse))
        b2 = "".join(_make_tool_assembler(provider).process_tool_call(t2, sse))
        b3 = "".join(_make_tool_assembler(provider).process_tool_call(t3, sse))
        combined = b1 + b2 + b3
        assert "call_split" in combined
        assert "Grep" in combined
        assert b1 == ""

    def test_tool_call_arguments_buffered_until_name(self):
        """Argument deltas before tool name are emitted after the block starts."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        t1 = {
            "index": 0,
            "id": "call_buf",
            "function": {"name": None, "arguments": '{"x":'},
        }
        t2 = {
            "index": 0,
            "id": "call_buf",
            "function": {"name": "Read", "arguments": "1}"},
        }
        b1 = "".join(_make_tool_assembler(provider).process_tool_call(t1, sse))
        b2 = "".join(_make_tool_assembler(provider).process_tool_call(t2, sse))
        assert b1 == ""
        combined = b2
        assert "Read" in combined
        assert "call_buf" in combined
        assert '{"x":' in combined or "partial_json" in combined

    def test_tool_call_without_id_generates_uuid(self):
        """Tool call without id generates a uuid-based id."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": None,
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_" in event_text

    def test_task_tool_forces_background_false(self):
        """Task tool with run_in_background=true is forced to false."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        args = json.dumps({"run_in_background": True, "prompt": "test"})
        tc = {
            "index": 0,
            "id": "call_task",
            "function": {"name": "Task", "arguments": args},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        # The intercepted args should have run_in_background=false
        assert "false" in event_text.lower()

    def test_task_tool_chunked_args_forces_background_false(self):
        """Chunked Task args are buffered until valid JSON, then forced to false."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc1 = {
            "index": 0,
            "id": "call_task_chunked",
            "function": {"name": "Task", "arguments": '{"run_in_background": true,'},
        }
        tc2 = {
            "index": 0,
            "id": "call_task_chunked",
            "function": {"name": None, "arguments": ' "prompt": "test"}'},
        }

        events1 = list(_make_tool_assembler(provider).process_tool_call(tc1, sse))
        assert len(events1) > 0
        assert "false" not in "".join(events1).lower()

        events2 = list(_make_tool_assembler(provider).process_tool_call(tc2, sse))
        event_text = "".join(events1 + events2)
        assert "false" in event_text.lower()

    def test_task_tool_invalid_json_logs_warning_on_flush(self, caplog):
        """Invalid JSON args for Task tool emits {} on flush and logs a warning."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        assert len(events) > 0

        with caplog.at_level("WARNING"):
            flushed = list(_make_tool_assembler(provider).flush_task_arg_buffers(sse))
        assert len(flushed) > 0
        assert "{}" in "".join(flushed)
        assert any("Task args invalid JSON" in r.message for r in caplog.records)

    def test_negative_tool_index_fallback(self):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": -1,
            "id": "call_neg",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        # Should not crash, should still emit events
        assert len(events) > 0

    def test_none_tool_index_defaults_to_zero(self):
        """Gemini may stream tool_call deltas with a null index."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": None,
            "id": "call_none",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)

        assert "tool_use" in event_text
        assert "call_none" in event_text

    def test_tool_args_emitted_as_delta(self):
        """Arguments are emitted as input_json_delta events."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "input_json_delta" in event_text


class TestStreamChunkEdgeCases:
    """Tests for edge cases in stream chunk handling."""

    @pytest.mark.asyncio
    async def test_stream_chunk_with_empty_choices_skipped(self):
        """Chunk with choices=[] is skipped without crashing."""
        provider = _make_provider()
        request = _make_request()

        empty_choices_chunk = MagicMock()
        empty_choices_chunk.choices = []
        empty_choices_chunk.usage = None

        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_choices_chunk, finish_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_chunk_with_none_delta_handled(self):
        """Chunk with choice.delta=None is handled defensively."""
        provider = _make_provider()
        request = _make_request()

        none_delta_chunk = MagicMock()
        none_delta_chunk.usage = None
        choice = MagicMock()
        choice.delta = None
        choice.finish_reason = None
        none_delta_chunk.choices = [choice]

        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([none_delta_chunk, finish_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_generator_cleanup_on_exception(self):
        """When stream raises mid-iteration, message_stop still emitted."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Partial")
        stream_mock = AsyncStreamMock(
            [chunk1], error=ConnectionResetError("Connection reset")
        )

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                provider._rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "Connection reset" in error.message

    def test_stream_malformed_tool_args_chunked(self):
        """Chunked tool args that never form valid JSON are flushed with {}."""
        provider = _make_provider()
        from free_claude_code.core.anthropic import AnthropicStreamLedger

        sse = AnthropicStreamLedger("msg_test", "test-model")
        tc1 = {
            "index": 0,
            "id": "call_malformed",
            "function": {"name": "Task", "arguments": '{"broken":'},
        }
        tc2 = {
            "index": 0,
            "id": "call_malformed",
            "function": {"name": None, "arguments": " never valid }"},
        }

        events1 = list(_make_tool_assembler(provider).process_tool_call(tc1, sse))
        events2 = list(_make_tool_assembler(provider).process_tool_call(tc2, sse))
        flushed = list(_make_tool_assembler(provider).flush_task_arg_buffers(sse))

        event_text = "".join(events1 + events2 + flushed)
        assert "tool_use" in event_text
        assert "{}" in event_text


@pytest.mark.asyncio
async def test_openai_compat_stream_ends_with_contract_when_tool_name_never_arrives() -> (
    None
):
    """Nameless / incomplete tool-call buffer must not break Anthropic stream contract."""
    provider = _make_provider()
    request = _make_request()
    tc0 = SimpleNamespace(
        index=0,
        id="call_inc",
        function=SimpleNamespace(name=None, arguments="{}"),
    )
    stream_mock = AsyncStreamMock([_make_chunk(tool_calls=[tc0])])
    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ),
        patch.object(
            provider._rate_limiter,
            "wait_if_blocked",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        error = await _collect_stream_error(provider, request)

    assert "Provider stream ended without finish_reason." in error.message
