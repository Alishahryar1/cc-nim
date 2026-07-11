"""FastAPI streaming response wrappers for public API wire formats."""

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from typing import Any, Literal, Protocol, runtime_checkable

from fastapi.responses import JSONResponse, Response, StreamingResponse

from free_claude_code.core.anthropic import anthropic_error_type_for_failure
from free_claude_code.core.anthropic.streaming import (
    ANTHROPIC_SSE_RESPONSE_HEADERS,
    anthropic_terminal_error_frame,
    anthropic_terminal_failure_frame,
)
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.trace import trace_event

TERMINAL_EXECUTION_ERROR_HEADERS = {"x-should-retry": "false"}

PreStartErrorResponse = Callable[[BaseException], Response]
TerminalFrameEmitter = Callable[[BaseException], str]
TerminalFailureObserver = Callable[[BaseException], None]
ReleaseResponseResource = Callable[[], Awaitable[None]]
WireApi = Literal["messages", "responses"]


@runtime_checkable
class _AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class EmptyStreamError(RuntimeError):
    """Raised when a public stream ends before emitting any protocol chunk."""


async def bind_response_lifetime(
    response: object,
    release: ReleaseResponseResource,
) -> object:
    """Retain a runtime resource until a response body is fully consumed."""
    if isinstance(response, StreamingResponse):
        response.body_iterator = _release_after_stream(
            response.body_iterator,
            release,
        )
        return response
    await release()
    return response


async def _release_after_stream(
    body: AsyncIterable[Any],
    release: ReleaseResponseResource,
) -> AsyncGenerator[Any]:
    iterator = body.__aiter__()
    try:
        async for chunk in iterator:
            yield chunk
    finally:
        try:
            if isinstance(iterator, _AsyncCloseable):
                await iterator.aclose()
        finally:
            await release()


def terminal_execution_error_response(
    *, status_code: int, content: dict[str, Any]
) -> JSONResponse:
    """Return a final provider-execution error without enabling client retries."""
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=dict(TERMINAL_EXECUTION_ERROR_HEADERS),
    )


def trace_terminal_execution_error(
    *,
    wire_api: WireApi,
    request_id: str,
    status_code: int,
    error_type: str,
    error: BaseException | None = None,
) -> None:
    """Record one correlated terminal-execution decision at the HTTP boundary."""
    fields: dict[str, object] = {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": wire_api,
        "request_id": request_id,
        "status_code": status_code,
        "error_type": error_type,
        "client_should_retry": False,
    }
    failure = find_execution_failure(error) if error is not None else None
    if error is not None:
        fields["exc_type"] = type(failure or error).__name__
    if failure is not None:
        fields["failure_kind"] = failure.kind.value
        fields["provider_retryable"] = failure.retryable
    trace_event(**fields)


async def _first_chunk_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    pre_start_error_response: PreStartErrorResponse,
    terminal_frame: TerminalFrameEmitter | None,
    terminal_failure_observer: TerminalFailureObserver | None,
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
            terminal_failure_observer=terminal_failure_observer,
        ),
        media_type="text/event-stream",
        headers=dict(headers),
    )


async def _replay_first_chunk_then_stream(
    first_chunk: str,
    body: AsyncIterator[str],
    *,
    terminal_frame: TerminalFrameEmitter | None,
    terminal_failure_observer: TerminalFailureObserver | None,
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
        terminal_error = find_execution_failure(exc) or exc
        if terminal_failure_observer is not None:
            terminal_failure_observer(terminal_error)
        yield terminal_frame(terminal_error)
    except Exception as exc:
        if terminal_frame is None:
            raise
        if terminal_failure_observer is not None:
            terminal_failure_observer(exc)
        yield terminal_frame(exc)


async def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    pre_start_error_response: PreStartErrorResponse,
    request_id: str,
) -> Response:
    """Return a streaming response for Anthropic-style SSE streams."""
    return await _first_chunk_streaming_response(
        body,
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
        pre_start_error_response=pre_start_error_response,
        terminal_frame=_anthropic_terminal_frame,
        terminal_failure_observer=lambda exc: _trace_anthropic_terminal_failure(
            exc,
            request_id=request_id,
        ),
    )


def _anthropic_terminal_frame(exc: BaseException) -> str:
    failure = find_execution_failure(exc)
    if failure is not None:
        return anthropic_terminal_failure_frame(failure)
    return anthropic_terminal_error_frame(safe_exception_message(exc))


def _trace_anthropic_terminal_failure(
    exc: BaseException,
    *,
    request_id: str,
) -> None:
    failure = find_execution_failure(exc)
    trace_terminal_execution_error(
        wire_api="messages",
        request_id=request_id,
        status_code=failure.status_code if failure is not None else 500,
        error_type=(
            anthropic_error_type_for_failure(failure)
            if failure is not None
            else "api_error"
        ),
        error=exc,
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
        terminal_failure_observer=None,
    )
