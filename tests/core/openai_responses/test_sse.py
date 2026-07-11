from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesRequest,
)
from free_claude_code.core.openai_responses.anthropic_sse import (
    AnthropicSseEvent,
    iter_sse_events,
)

_ADAPTER = OpenAIResponsesAdapter()


class _CloseTrackingAsyncIterator:
    def __init__(
        self,
        values: list[Any],
        *,
        iteration_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._values = iter(values)
        self._iteration_error = iteration_error
        self._close_error = close_error
        self.close_calls = 0

    def __aiter__(self) -> _CloseTrackingAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._values)
        except StopIteration:
            if self._iteration_error is not None:
                error = self._iteration_error
                self._iteration_error = None
                raise error from None
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _responses_sse(
    chunks: AsyncIterator[str], request: dict[str, Any]
) -> AsyncIterator[str]:
    return _ADAPTER.iter_sse_from_anthropic(
        chunks,
        OpenAIResponsesRequest.model_validate(request),
    )


@pytest.mark.asyncio
async def test_anthropic_sse_parser_closes_source_after_normal_completion() -> None:
    source = _CloseTrackingAsyncIterator(
        [format_sse_event("message_start", {"type": "message_start"})]
    )

    events = [event async for event in iter_sse_events(source)]

    assert [event.event for event in events] == ["message_start"]
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_anthropic_sse_parser_closes_source_on_early_close() -> None:
    source = _CloseTrackingAsyncIterator(
        [
            format_sse_event("message_start", {"type": "message_start"}),
            format_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    events = iter_sse_events(source)

    assert (await anext(events)).event == "message_start"
    assert isinstance(events, AsyncCloseable)
    await events.aclose()

    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_anthropic_sse_parser_preserves_source_failure() -> None:
    source_failure = RuntimeError("source failed")
    source = _CloseTrackingAsyncIterator(
        [],
        iteration_error=source_failure,
        close_error=RuntimeError("close failed"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        [event async for event in iter_sse_events(source)]

    assert exc_info.value is source_failure
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_responses_transform_closes_direct_event_source_on_early_close() -> None:
    events = _CloseTrackingAsyncIterator(
        [
            AnthropicSseEvent(
                event="message_start",
                data={"type": "message_start", "message": {}},
            ),
            AnthropicSseEvent(
                event="message_stop",
                data={"type": "message_stop"},
            ),
        ]
    )
    with patch(
        "free_claude_code.core.openai_responses.stream.iter_sse_events",
        return_value=events,
    ):
        stream = _responses_sse(
            _aiter([]),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
        assert parse_sse_text(await anext(stream))[0].event == "response.created"
        assert isinstance(stream, AsyncCloseable)
        await stream.aclose()

    assert events.close_calls == 1


@pytest.mark.asyncio
async def test_responses_transform_preserves_direct_source_failure() -> None:
    source_failure = RuntimeError("event source failed")
    events = _CloseTrackingAsyncIterator(
        [],
        iteration_error=source_failure,
        close_error=RuntimeError("event close failed"),
    )
    with (
        patch(
            "free_claude_code.core.openai_responses.stream.iter_sse_events",
            return_value=events,
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        [
            chunk
            async for chunk in _responses_sse(
                _aiter([]),
                {"model": "nvidia_nim/test-model", "stream": True},
            )
        ]

    assert exc_info.value is source_failure
    assert events.close_calls == 1


@pytest.mark.asyncio
async def test_anthropic_text_stream_converts_to_responses_sse() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(_anthropic_text_stream("Hello Codex")),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    event_names = [event.event for event in events]
    assert event_names[:3] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert "response.output_text.delta" in event_names
    assert events[-1].event == "response.completed"
    completed = events[-1].data["response"]
    assert completed["output"][0]["content"][0]["text"] == "Hello Codex"
    assert completed["parallel_tool_calls"] is True
    assert completed["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_response_payload_preserves_explicit_request_options() -> None:
    response = await _completed_response_from_sse(
        _aiter(_anthropic_text_stream("done")),
        {
            "model": "nvidia_nim/test-model",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_output_tokens": 0,
        },
    )

    assert response["parallel_tool_calls"] is False
    assert response["tool_choice"] == "none"
    assert response["temperature"] == 0.0
    assert response["top_p"] == 0.0
    assert response["max_output_tokens"] == 0


@pytest.mark.asyncio
async def test_response_payload_maps_explicit_null_options_to_wire_defaults() -> None:
    response = await _completed_response_from_sse(
        _aiter(_anthropic_text_stream("done")),
        {
            "model": "nvidia_nim/test-model",
            "parallel_tool_calls": None,
            "tool_choice": None,
        },
    )

    assert response["parallel_tool_calls"] is True
    assert response["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_anthropic_tool_stream_converts_to_function_call_item() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(_anthropic_tool_stream()),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    names = [event.event for event in events]
    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    completed = events[-1].data["response"]
    function_call = completed["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "toolu_1"
    assert function_call["name"] == "echo"
    assert function_call["arguments"] == '{"value":"FCC"}'


@pytest.mark.asyncio
async def test_anthropic_function_tool_arguments_are_normalized() -> None:
    response = await _completed_response_from_sse(
        _aiter(_anthropic_tool_stream(partial_json='{ "value" : "FCC" }')),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert response["output"][0]["arguments"] == '{"value":"FCC"}'


@pytest.mark.asyncio
async def test_anthropic_malformed_function_tool_arguments_fail_response() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(_anthropic_tool_stream(partial_json='{"value":"FCC" "bad"}')),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    assert events[-1].event == "response.failed"
    assert "response.function_call_arguments.done" not in [
        event.event for event in events
    ]
    assert "response.output_item.done" not in [event.event for event in events]
    failed = events[-1].data["response"]
    assert failed["status"] == "failed"
    assert failed["output"] == []
    assert failed["error"]["type"] == "api_error"
    assert "replay-unsafe Responses output" in failed["error"]["message"]


@pytest.mark.asyncio
async def test_anthropic_malformed_function_tool_arguments_fail_on_eof() -> None:
    stream = _anthropic_tool_stream(
        partial_json='{"value":"FCC" "bad"}',
        include_block_stop=False,
    )
    text = await _collect_sse(
        _responses_sse(
            _aiter(stream[:-1]),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    assert events[-1].event == "response.failed"
    assert events[-1].data["response"]["output"] == []


@pytest.mark.asyncio
async def test_provider_failure_outranks_provisional_incomplete_function_call() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="provider is busy\n\nRequest ID: req_failure_precedence",
        retryable=True,
    )
    incomplete_tool = _anthropic_tool_stream(partial_json='{"value":')[:4]

    text = await _collect_sse(
        _responses_sse(
            _aiter_then_raise(incomplete_tool, failure),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    assert [event.event for event in events].count("response.failed") == 1
    assert events[-1].event == "response.failed"
    failed = events[-1].data["response"]
    assert failed["id"] == events[0].data["response"]["id"]
    assert failed["error"] == {
        "message": "provider is busy\n\nRequest ID: req_failure_precedence",
        "type": "rate_limit_error",
        "param": None,
        "code": None,
    }


@pytest.mark.asyncio
async def test_post_start_failure_observer_runs_before_terminal_failure_event() -> None:
    failure = RuntimeError("socket closed")
    timeline: list[tuple[str, object]] = []

    def observe_terminal_failure(exc: BaseException) -> None:
        timeline.append(("observed", exc))

    stream = _ADAPTER.iter_sse_from_anthropic(
        _aiter_then_raise(_anthropic_text_stream("partial")[:3], failure),
        OpenAIResponsesRequest.model_validate(
            {"model": "nvidia_nim/test-model", "stream": True}
        ),
        on_post_start_terminal_failure=observe_terminal_failure,
    )
    parts: list[str] = []
    async for chunk in stream:
        parts.append(chunk)
        event = parse_sse_text(chunk)
        assert len(event) == 1
        timeline.append(("emitted", event[0].event))

    events = parse_sse_text("".join(parts))
    assert timeline.count(("observed", failure)) == 1
    assert timeline.index(("observed", failure)) < timeline.index(
        ("emitted", "response.failed")
    )
    assert events[-1].data["response"]["error"]["message"] == "socket closed"


@pytest.mark.asyncio
async def test_unrelated_post_start_exception_group_remains_unexpected() -> None:
    grouped = ExceptionGroup("stream failed", [RuntimeError("socket closed")])
    observed: list[BaseException] = []
    stream = _ADAPTER.iter_sse_from_anthropic(
        _aiter_then_raise(_anthropic_text_stream("partial")[:3], grouped),
        OpenAIResponsesRequest.model_validate(
            {"model": "nvidia_nim/test-model", "stream": True}
        ),
        on_post_start_terminal_failure=observed.append,
    )

    events = parse_sse_text(await _collect_sse(stream))

    assert observed == [grouped]
    assert events[-1].event == "response.failed"
    assert events[-1].data["response"]["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_namespaced_anthropic_tool_stream_restores_responses_namespace() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(_anthropic_tool_stream(tool_name="mcp__node_repl__js")),
            {
                "model": "nvidia_nim/test-model",
                "stream": True,
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__node_repl",
                        "tools": [
                            {
                                "type": "function",
                                "name": "js",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ],
            },
        )
    )

    events = parse_sse_text(text)
    completed = events[-1].data["response"]
    function_call = completed["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["namespace"] == "mcp__node_repl"
    assert function_call["name"] == "js"


@pytest.mark.asyncio
async def test_anthropic_custom_tool_stream_converts_to_custom_tool_call() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(
                _anthropic_tool_stream(
                    tool_name="apply_patch",
                    partial_json='{"input":"*** Begin Patch"}',
                )
            ),
            {
                "model": "nvidia_nim/test-model",
                "stream": True,
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {"type": "text"},
                    }
                ],
            },
        )
    )

    events = parse_sse_text(text)
    names = [event.event for event in events]
    assert "response.custom_tool_call_input.delta" in names
    assert "response.custom_tool_call_input.done" in names
    assert "response.function_call_arguments.delta" not in names
    completed = events[-1].data["response"]
    custom_call = completed["output"][0]
    assert custom_call["type"] == "custom_tool_call"
    assert custom_call["call_id"] == "toolu_1"
    assert custom_call["name"] == "apply_patch"
    assert custom_call["input"] == "*** Begin Patch"


@pytest.mark.asyncio
async def test_custom_tool_input_remains_free_form_when_not_json() -> None:
    response = await _completed_response_from_sse(
        _aiter(
            _anthropic_tool_stream(
                tool_name="apply_patch",
                partial_json="*** Begin Patch",
            )
        ),
        {
            "model": "nvidia_nim/test-model",
            "stream": True,
            "tools": [{"type": "custom", "name": "apply_patch"}],
        },
    )

    custom_call = response["output"][0]
    assert custom_call["type"] == "custom_tool_call"
    assert custom_call["input"] == "*** Begin Patch"


@pytest.mark.asyncio
async def test_anthropic_error_stream_converts_to_response_failed_event() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(
                [
                    format_sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "upstream failed",
                            },
                        },
                    )
                ]
            ),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    assert events[0].event == "response.created"
    assert events[1].event == "response.failed"
    failed = events[1].data["response"]
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "upstream failed"


@pytest.mark.asyncio
async def test_split_usage_deltas_are_accumulated() -> None:
    response = await _completed_response_from_sse(
        _aiter(
            [
                *_anthropic_text_stream("usage")[:-2],
                format_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"input_tokens": 11},
                    },
                ),
                format_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {},
                        "usage": {"output_tokens": 7},
                    },
                ),
                format_sse_event("message_stop", {"type": "message_stop"}),
            ]
        ),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert response["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert "stop_reason" not in response


@pytest.mark.asyncio
async def test_reasoning_stream_reports_reasoning_usage_detail() -> None:
    response = await _completed_response_from_sse(
        _aiter(_anthropic_reasoning_stream("inspect the code before answering")),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    usage = response["usage"]
    assert usage["input_tokens"] == 3
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 23
    assert usage["output_tokens_details"]["reasoning_tokens"] > 0


@pytest.mark.asyncio
async def test_reasoning_usage_detail_is_capped_at_output_tokens() -> None:
    response = await _completed_response_from_sse(
        _aiter(
            _anthropic_reasoning_stream(
                "this reasoning text is intentionally long enough to exceed one token",
                output_tokens=1,
            )
        ),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert response["usage"]["output_tokens"] == 1
    assert response["usage"]["output_tokens_details"]["reasoning_tokens"] == 1


@pytest.mark.asyncio
async def test_reasoning_usage_detail_omits_zero_capped_count() -> None:
    response = await _completed_response_from_sse(
        _aiter(
            _anthropic_reasoning_stream(
                "reasoning text exists without reported output tokens",
                output_tokens=None,
            )
        ),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert response["usage"] == {
        "input_tokens": 3,
        "output_tokens": 0,
        "total_tokens": 3,
    }


@pytest.mark.asyncio
async def test_text_only_usage_omits_reasoning_usage_detail() -> None:
    response = await _completed_response_from_sse(
        _aiter(_anthropic_text_stream("plain text only")),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert response["usage"] == {
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }


@pytest.mark.parametrize(
    ("request_payload", "tool_name", "partial_json", "expected_type", "expected_field"),
    [
        (
            {"model": "nvidia_nim/test-model", "stream": True},
            "echo",
            '{"value":"FCC"}',
            "function_call",
            ("arguments", '{"value":"FCC"}'),
        ),
        (
            {
                "model": "nvidia_nim/test-model",
                "stream": True,
                "tools": [{"type": "custom", "name": "apply_patch"}],
            },
            "apply_patch",
            '{"input":"*** Begin Patch"}',
            "custom_tool_call",
            ("input", "*** Begin Patch"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_pending_tool_blocks_flush_on_message_stop_and_eof(
    request_payload: dict[str, object],
    tool_name: str,
    partial_json: str,
    expected_type: str,
    expected_field: tuple[str, str],
) -> None:
    stream = _anthropic_tool_stream(
        tool_name=tool_name, partial_json=partial_json, include_block_stop=False
    )
    message_stop_response = await _completed_response_from_sse(
        _aiter(stream), request_payload
    )
    eof_response = await _completed_response_from_sse(
        _aiter(stream[:-1]), request_payload
    )

    for response in (message_stop_response, eof_response):
        call = response["output"][0]
        assert call["type"] == expected_type
        assert call[expected_field[0]] == expected_field[1]


@pytest.mark.asyncio
async def test_overlapping_text_and_tool_blocks_keep_reserved_output_indexes() -> None:
    text = await _collect_sse(
        _responses_sse(
            _aiter(_overlapping_text_tool_stream()),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    item_events = [
        (event.event, event.data["output_index"])
        for event in events
        if event.event in {"response.output_item.added", "response.output_item.done"}
    ]
    assert item_events == [
        ("response.output_item.added", 0),
        ("response.output_item.added", 1),
        ("response.output_item.done", 1),
        ("response.output_item.done", 0),
    ]
    completed = events[-1].data["response"]
    assert [item["type"] for item in completed["output"]] == [
        "message",
        "function_call",
    ]
    assert completed["output"][0]["content"][0]["text"] == "text"
    assert completed["output"][1]["arguments"] == '{"value":"FCC"}'


@pytest.mark.asyncio
async def test_overlapping_text_blocks_do_not_merge_content_by_index() -> None:
    response = await _completed_response_from_sse(
        _aiter(_overlapping_text_stream()),
        {"model": "nvidia_nim/test-model", "stream": True},
    )

    assert [item["content"][0]["text"] for item in response["output"]] == [
        "A1-A2",
        "B1-B2",
    ]


async def _collect_sse(chunks: AsyncIterator[str]) -> str:
    parts = [chunk async for chunk in chunks]
    return "".join(parts)


async def _completed_response_from_sse(
    chunks: AsyncIterator[str],
    request: dict[str, object],
) -> dict[str, Any]:
    text = await _collect_sse(_responses_sse(chunks, request))
    events = parse_sse_text(text)
    assert events[-1].event == "response.completed"
    return events[-1].data["response"]


async def _aiter(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def _aiter_then_raise(
    chunks: list[str], failure: BaseException
) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk
    raise failure


def _anthropic_text_stream(text: str) -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
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
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_tool_stream(
    tool_name: str = "echo",
    partial_json: str = '{"value":"FCC"}',
    *,
    include_block_stop: bool = True,
) -> list[str]:
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": tool_name,
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            },
        ),
    ]
    if include_block_stop:
        chunks.append(
            format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
        )
    chunks.extend(
        [
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                },
            ),
            format_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    return chunks


def _anthropic_reasoning_stream(
    reasoning: str,
    *,
    output_tokens: int | None = 20,
) -> list[str]:
    usage = {"input_tokens": 3}
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens

    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": usage,
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _overlapping_text_tool_stream() -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
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
                "delta": {"type": "text_delta", "text": "text"},
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "echo",
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"value":"FCC"}',
                },
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _overlapping_text_stream() -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
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
                "delta": {"type": "text_delta", "text": "A1-"},
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "B1-"},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "A2"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "B2"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
