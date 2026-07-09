"""Translate Anthropic SSE streams into OpenAI Responses SSE streams."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from core.trace import trace_event

from .anthropic_sse import iter_sse_events
from .streaming import ResponsesStreamAssembler

DEFAULT_STREAM_INTERRUPTED_MESSAGE = (
    "The upstream response stream ended unexpectedly; the request could not be "
    "completed."
)


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
    *,
    stream_error_message: str = DEFAULT_STREAM_INTERRUPTED_MESSAGE,
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""

    assembler = ResponsesStreamAssembler(request)
    emitted_any_chunk = False
    try:
        async for event in iter_sse_events(chunks):
            for chunk in assembler.process_anthropic_event(event):
                yield chunk
                emitted_any_chunk = True
            if assembler.terminal:
                return
        for chunk in assembler.finish_if_needed():
            yield chunk
            emitted_any_chunk = True
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        if not emitted_any_chunk:
            raise
        trace_event(
            stage="responses",
            event="responses.stream.terminal_failure_frame",
            source="openai_responses",
            exc_type=type(exc).__name__,
        )
        for chunk in assembler.fail_response(
            {"error": {"type": "api_error", "message": stream_error_message}}
        ):
            yield chunk
        return
    except Exception as exc:
        if not emitted_any_chunk:
            raise
        trace_event(
            stage="responses",
            event="responses.stream.terminal_failure_frame",
            source="openai_responses",
            exc_type=type(exc).__name__,
        )
        for chunk in assembler.fail_response(
            {"error": {"type": "api_error", "message": stream_error_message}}
        ):
            yield chunk
        return
