"""FastAPI streaming response wrappers for public API wire formats."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from free_claude_code.core.anthropic import get_user_facing_error_message
from free_claude_code.core.anthropic.streaming import (
    ANTHROPIC_SSE_RESPONSE_HEADERS,
    anthropic_terminal_error_frame,
)
from free_claude_code.core.trace import trace_event

TERMINAL_EXECUTION_ERROR_HEADERS = {"x-should-retry": "false"}

PreStartErrorResponse = Callable[[BaseException], Response]
TerminalFrameEmitter = Callable[[BaseException], str]


class EmptyStreamError(RuntimeError):
    """Raised when a public stream ends before emitting any protocol chunk."""


def terminal_execution_error_response(
    *, status_code: int, content: dict[str, Any]
) -> JSONResponse:
    """Return a final provider-execution error without enabling client retries."""
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=dict(TERMINAL_EXECUTION_ERROR_HEADERS),
    )


def _trace_egress_failure(exc: BaseException) -> None:
    trace_event(
        stage="egress",
        event="free_claude_code.api.response.egress_error_frame_emitted",
        source="api",
        exc_type=type(exc).__name__,
    )


async def _first_chunk_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    pre_start_error_response: PreStartErrorResponse,
    terminal_frame: TerminalFrameEmitter | None,
) -> Response:
    try:
        first_chunk = await anext(body)
    except StopAsyncIteration:
        return pre_start_error_response(
            EmptyStreamError("Stream ended before emitting a response.")
        )
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        return pre_start_error_response(exc)
    except Exception as exc:
        return pre_start_error_response(exc)

    return StreamingResponse(
        _replay_first_chunk_then_stream(
            first_chunk,
            body,
            terminal_frame=terminal_frame,
        ),
        media_type="text/event-stream",
        headers=dict(headers),
    )


async def _replay_first_chunk_then_stream(
    first_chunk: str,
    body: AsyncIterator[str],
    *,
    terminal_frame: TerminalFrameEmitter | None,
) -> AsyncGenerator[str]:
    yield first_chunk
    try:
        async for chunk in body:
            yield chunk
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        if terminal_frame is None:
            raise
        _trace_egress_failure(exc)
        yield terminal_frame(exc)
    except Exception as exc:
        if terminal_frame is None:
            raise
        _trace_egress_failure(exc)
        yield terminal_frame(exc)


async def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    pre_start_error_response: PreStartErrorResponse,
) -> Response:
    """Return a streaming response for Anthropic-style SSE streams."""
    return await _first_chunk_streaming_response(
        body,
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
        pre_start_error_response=pre_start_error_response,
        terminal_frame=lambda exc: anthropic_terminal_error_frame(
            get_user_facing_error_message(exc)
        ),
    )


async def openai_responses_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    pre_start_error_response: PreStartErrorResponse,
) -> Response:
    """Return a streaming response for OpenAI Responses-style SSE."""
    return await _first_chunk_streaming_response(
        body,
        headers=headers,
        pre_start_error_response=pre_start_error_response,
        terminal_frame=None,
    )
