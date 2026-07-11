"""Translate Anthropic SSE streams into OpenAI Responses SSE streams."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from free_claude_code.core.anthropic import get_user_facing_error_message
from free_claude_code.core.trace import trace_event

from .anthropic_sse import iter_sse_events
from .models import OpenAIResponsesRequest
from .streaming import ResponsesStreamAssembler


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: OpenAIResponsesRequest,
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
            {
                "error": {
                    "type": "api_error",
                    "message": get_user_facing_error_message(exc),
                }
            }
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
            {
                "error": {
                    "type": "api_error",
                    "message": get_user_facing_error_message(exc),
                }
            }
        ):
            yield chunk
        return
