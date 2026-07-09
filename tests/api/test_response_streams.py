"""Tests for public SSE response start gating."""

import asyncio
import json
from collections.abc import AsyncGenerator

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from api.response_streams import (
    EGRESS_STREAM_INTERRUPTED_MESSAGE,
    anthropic_sse_streaming_response,
)
from core.anthropic.stream_contracts import parse_sse_text
from providers.exceptions import RateLimitError


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
    if isinstance(exc, RateLimitError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
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
        _body_raises(RateLimitError("provider says slow down")),
        pre_start_error_response=_json_error,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "provider says slow down"


@pytest.mark.asyncio
async def test_anthropic_post_start_exception_emits_terminal_error_frame() -> None:
    response = await anthropic_sse_streaming_response(
        _body_then_raises(
            ['event: message_start\ndata: {"type":"message_start"}\n\n'],
            RuntimeError("socket cut"),
        ),
        pre_start_error_response=_json_error,
    )

    assert isinstance(response, StreamingResponse)
    text = await _drain(response)
    events = parse_sse_text(text)
    assert [event.event for event in events] == ["message_start", "error"]
    assert events[-1].data["error"]["message"] == EGRESS_STREAM_INTERRUPTED_MESSAGE
