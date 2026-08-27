"""Contracts for the standard API-key OpenAI Responses transport."""

import asyncio
import json
from collections.abc import Callable, Mapping

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.support import REASONING_ON, immediate_admission


def _request(**overrides: object) -> MessagesRequest:
    payload: dict[str, object] = {
        "model": "upstream-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 123,
        "metadata": {"source": "test"},
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _completed_response(
    *,
    model: str = "upstream-model",
    input_tokens: int = 8,
    cached_tokens: int = 3,
    output_tokens: int = 2,
) -> dict[str, object]:
    return {
        "id": "resp_test",
        "created_at": 0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": model,
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "background": False,
        "conversation": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "prompt_cache_key": None,
        "reasoning": None,
        "safety_identifier": None,
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "user": None,
        "store": False,
    }


def _text_delta(text: str, *, sequence: int = 0) -> dict[str, object]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence,
        "item_id": "item_text",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
        "logprobs": [],
    }


def _completed_event(*, sequence: int = 1) -> dict[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": sequence,
        "response": _completed_response(),
    }


def _sse(*events: Mapping[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _transport(client: AsyncOpenAI, *, max_attempts: int = 5):
    return OpenAIResponsesTransport(
        client=client,
        admission=immediate_admission(
            provider_name="TEST_RESPONSES",
            max_attempts=max_attempts,
        ),
        provider_name="TEST_RESPONSES",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
    )


async def _collect(
    transport: OpenAIResponsesTransport,
    request: MessagesRequest | None = None,
) -> list[str]:
    return [
        chunk
        async for chunk in transport.stream_messages(
            request or _request(),
            input_tokens=11,
            request_id="req_responses",
            response_model="public-model",
            reasoning=REASONING_ON,
        )
    ]


async def _collect_native(
    transport: OpenAIResponsesTransport,
    request: OpenAIResponsesRequest,
    *,
    reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
) -> list[str]:
    return [
        chunk
        async for chunk in transport.stream_responses(
            request,
            input_tokens=11,
            request_id="req_native_responses",
            response_model="public-model",
            reasoning=reasoning,
        )
    ]


@pytest.mark.asyncio
async def test_native_responses_preserves_request_and_upstream_event_identity() -> None:
    captured: list[dict[str, object]] = []
    created_response = {
        **_completed_response(model="upstream-model"),
        "status": "in_progress",
        "output": [],
        "usage": None,
    }
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": created_response,
    }
    delta = _text_delta("hello", sequence=1)
    completed = _completed_event(sequence=2)

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(created, delta, completed),
        )

    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "store": True,
            "previous_response_id": "resp_previous",
            "future_option": {"enabled": True},
        }
    )
    client = _client(handler)
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    assert captured == [
        {
            "model": "upstream-model",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "store": False,
            "future_option": {"enabled": True},
        }
    ]
    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].data["response"]["id"] == "resp_test"
    assert events[0].data["response"]["model"] == "public-model"
    assert events[1].data == delta
    assert events[2].data["response"]["model"] == "public-model"


@pytest.mark.asyncio
async def test_native_responses_applies_resolved_reasoning_override() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed_event()),
        )

    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": "hello",
            "reasoning": {"effort": "high", "summary": "detailed"},
        }
    )
    client = _client(handler)
    try:
        await _collect_native(
            _transport(client),
            request,
            reasoning=ReasoningPolicy.off(),
        )
    finally:
        await client.close()

    assert captured[0]["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_native_response_failed_retries_before_public_commitment() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        created_response = {
            **_completed_response(),
            "id": f"resp_attempt_{attempts}",
            "status": "in_progress",
            "output": [],
            "usage": None,
        }
        created = {
            "type": "response.created",
            "sequence_number": 0,
            "response": created_response,
        }
        if attempts == 1:
            failed = {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    **created_response,
                    "status": "failed",
                    "error": {
                        "message": "temporary",
                        "type": "server_error",
                        "code": "server_error",
                    },
                },
            }
            events = (created, failed)
        else:
            completed = {
                **_completed_event(sequence=1),
                "response": {
                    **_completed_response(),
                    "id": "resp_attempt_2",
                },
            }
            events = (created, completed)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(*events),
        )

    client = _client(handler)
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client, max_attempts=2), request)
    finally:
        await client.close()

    body = "".join(chunks)
    events = parse_sse_text(body)
    assert attempts == 2
    assert [event.event for event in events] == [
        "response.created",
        "response.completed",
    ]
    assert "resp_attempt_1" not in body
    assert events[0].data["response"]["id"] == "resp_attempt_2"


@pytest.mark.asyncio
async def test_native_committed_truncation_emits_one_failed_terminal() -> None:
    attempts = 0
    committed = "x" * 70_000
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            **_completed_response(),
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(created, _text_delta(committed, sequence=1)),
        )

    client = _client(handler)
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client, max_attempts=2), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert attempts == 1
    assert [event.event for event in events][-1] == "response.failed"
    assert sum(event.event == "response.failed" for event in events) == 1
    assert events[-1].data["response"]["id"] == "resp_test"
    assert events[-1].data["response"]["model"] == "public-model"
    assert events[-1].data["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_native_response_incomplete_is_a_normal_terminal() -> None:
    incomplete = {
        "type": "response.incomplete",
        "sequence_number": 1,
        "response": {
            **_completed_response(),
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(incomplete),
        )
    )
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == ["response.incomplete"]


@pytest.mark.asyncio
async def test_native_response_stops_at_terminal_before_trailing_ping() -> None:
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed_event(), {"type": "ping"}),
        )
    )
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == ["response.completed"]


@pytest.mark.asyncio
async def test_standard_responses_preserves_public_fields_and_usage() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("hello"), _completed_event()),
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    assert len(captured) == 1
    assert captured[0]["model"] == "upstream-model"
    assert captured[0]["max_output_tokens"] == 123
    assert captured[0]["metadata"] == {"source": "test"}
    assert captured[0]["store"] is False
    events = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "hello"
    final_usage = next(
        event.data["usage"] for event in events if event.event == "message_delta"
    )
    assert final_usage == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
    }


@pytest.mark.asyncio
async def test_standard_responses_maps_reasoning_and_tool_calls() -> None:
    events: tuple[dict[str, object], ...] = (
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 0,
            "item_id": "reasoning_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "thinking",
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "item_tool",
                "call_id": "call_tool",
                "name": "Read",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "item_tool",
            "output_index": 1,
            "delta": '{"path":"README.md"}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "item_tool",
                "call_id": "call_tool",
                "name": "Read",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            },
        },
        _completed_event(sequence=4),
    )

    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(*events),
        )
    )
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    parsed = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(parsed)
    assert thinking_content(parsed) == "thinking"
    tool_start = next(
        event.data["content_block"]
        for event in parsed
        if event.event == "content_block_start"
        and event.data["content_block"]["type"] == "tool_use"
    )
    assert tool_start == {
        "type": "tool_use",
        "id": "call_tool",
        "name": "Read",
        "input": {},
    }
    tool_delta = next(
        event.data["delta"]
        for event in parsed
        if event.event == "content_block_delta"
        and event.data["delta"]["type"] == "input_json_delta"
    )
    assert tool_delta["partial_json"] == '{"path":"README.md"}'


@pytest.mark.asyncio
async def test_retryable_open_failure_retries_inside_one_transport() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx2.Response(
                500,
                json={"error": {"message": "temporary", "type": "server_error"}},
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("recovered"), _completed_event()),
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    assert attempts == 2
    parsed = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(parsed)
    assert text_content(parsed) == "recovered"


@pytest.mark.asyncio
async def test_early_truncated_retry_has_one_visible_lifecycle() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        body = (
            _sse(_text_delta("discarded"))
            if attempts == 1
            else _sse(_text_delta("kept"), _completed_event())
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    parsed = parse_sse_text("".join(chunks))
    assert attempts == 2
    assert text_content(parsed) == "kept"
    assert "discarded" not in "".join(chunks)
    assert sum(event.event == "message_start" for event in parsed) == 1
    assert sum(event.event == "message_stop" for event in parsed) == 1
    assert_anthropic_stream_contract(parsed)


@pytest.mark.asyncio
async def test_exhausted_5xx_uses_exact_attempt_budget() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            503,
            json={"error": {"message": "busy", "type": "server_error"}},
        )

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            await _collect(_transport(client))
    finally:
        await client.close()

    assert attempts == 5
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_post_commit_truncation_is_not_replayed() -> None:
    attempts = 0
    committed = "x" * 70_000

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta(committed)),
        )

    client = _client(handler)
    chunks: list[str] = []
    try:
        with pytest.raises(ExecutionFailure):
            async for chunk in _transport(client).stream_messages(
                _request(),
                input_tokens=1,
                request_id="req_committed",
                response_model="public-model",
                reasoning=REASONING_ON,
            ):
                chunks.extend((chunk,))
    finally:
        await client.close()

    assert attempts == 1
    assert "".join(chunks).count(committed) == 1
    assert "".join(chunks).count("event: message_start") == 1


class _BlockingBody(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        self.entered.set()
        await asyncio.Event().wait()
        yield b""

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_cancellation_closes_the_sdk_stream() -> None:
    body = _BlockingBody()
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body,
        )
    )
    task = asyncio.create_task(_collect(_transport(client)))
    await asyncio.wait_for(body.entered.wait(), timeout=5)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(body.closed.wait(), timeout=5)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_preflight_rejects_fields_responses_cannot_represent() -> None:
    client = _client(lambda _request: httpx2.Response(500))
    transport = _transport(client)
    request = _request(stop_sequences=["done"])

    try:
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            transport.preflight_messages(request, reasoning=REASONING_ON)
    finally:
        await client.close()
