"""Tests for public SSE response start gating."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from free_claude_code.api.response_streams import (
    anthropic_sse_streaming_response,
    bind_response_lifetime,
    terminal_execution_error_response,
)
from free_claude_code.core.anthropic import (
    anthropic_error_payload,
    anthropic_failure_payload,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind


async def _body_chunks(chunks: list[str]) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk


async def _body_raises(exc: BaseException) -> AsyncGenerator[str]:
    raise exc
    yield "unreachable"


async def _body_then_raises(
    chunks: list[str], exc: BaseException
) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk
    raise exc


def _json_error(exc: BaseException) -> JSONResponse:
    if isinstance(exc, ExecutionFailure):
        return terminal_execution_error_response(
            status_code=exc.status_code,
            content=anthropic_failure_payload(exc),
        )
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {"type": "api_error", "message": "failed"},
        },
    )


async def _drain(response: StreamingResponse) -> str:
    parts = [
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ]
    return "".join(parts)


@pytest.mark.asyncio
async def test_anthropic_response_waits_for_first_chunk_before_returning() -> None:
    ready = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        await ready.wait()
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

    task = asyncio.create_task(
        anthropic_sse_streaming_response(
            body(),
            pre_start_error_response=_json_error,
            request_id="req_test",
        )
    )

    await asyncio.sleep(0)
    assert not task.done()

    ready.set()
    response = await asyncio.wait_for(task, timeout=1)
    assert isinstance(response, StreamingResponse)
    assert "message_start" in await _drain(response)


@pytest.mark.asyncio
async def test_anthropic_pre_start_provider_error_returns_non_200_json() -> None:
    response = await anthropic_sse_streaming_response(
        _body_raises(
            ExecutionFailure(
                kind=FailureKind.RATE_LIMIT,
                status_code=429,
                message="provider says slow down",
                retryable=True,
            )
        ),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "provider says slow down"


@pytest.mark.asyncio
async def test_terminal_execution_error_response_disables_client_retry() -> None:
    response = terminal_execution_error_response(
        status_code=429,
        content=anthropic_error_payload(
            error_type="rate_limit_error",
            message="provider says slow down",
        ),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert body["error"] == {
        "type": "rate_limit_error",
        "message": "provider says slow down",
    }


@pytest.mark.asyncio
async def test_anthropic_post_start_exception_emits_terminal_error_frame() -> None:
    response = await anthropic_sse_streaming_response(
        _body_then_raises(
            ['event: message_start\ndata: {"type":"message_start"}\n\n'],
            RuntimeError("socket cut"),
        ),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )

    assert isinstance(response, StreamingResponse)
    text = await _drain(response)
    events = parse_sse_text(text)
    assert [event.event for event in events] == ["message_start", "error"]
    assert events[-1].data["error"]["message"] == "socket cut"


@pytest.mark.asyncio
async def test_non_streaming_response_releases_resource_before_return() -> None:
    release = AsyncMock()
    response = JSONResponse({"ok": True})

    result = await bind_response_lifetime(response, release)

    assert result is response
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_after_normal_completion() -> None:
    release = AsyncMock()
    response = StreamingResponse(_body_chunks(["one", "two"]))

    result = await bind_response_lifetime(response, release)

    assert result is response
    release.assert_not_awaited()
    assert await _drain(response) == "onetwo"
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_after_body_failure() -> None:
    release = AsyncMock()
    response = StreamingResponse(
        _body_then_raises(["one"], RuntimeError("stream failed"))
    )
    await bind_response_lifetime(response, release)

    with pytest.raises(RuntimeError, match="stream failed"):
        await _drain(response)

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_when_consumer_closes_early() -> None:
    release = AsyncMock()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "one"
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    response = StreamingResponse(body())
    await bind_response_lifetime(response, release)
    iterator = cast(
        AsyncGenerator[str],
        response.body_iterator.__aiter__(),
    )

    assert await anext(iterator) == "one"
    await iterator.aclose()

    assert source_closed.is_set()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_when_consumer_is_cancelled() -> None:
    release = AsyncMock()
    entered = asyncio.Event()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "one"
            entered.set()
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    response = StreamingResponse(body())
    await bind_response_lifetime(response, release)
    drain_task = asyncio.create_task(_drain(response))
    await entered.wait()

    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    assert source_closed.is_set()
    release.assert_awaited_once()
