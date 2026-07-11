"""Translate Anthropic SSE streams into OpenAI Responses SSE streams."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure

from .anthropic_sse import iter_sse_events
from .models import OpenAIResponsesRequest
from .streaming import ResponsesStreamAssembler

PostStartTerminalFailureObserver = Callable[[BaseException], None]


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: OpenAIResponsesRequest,
    *,
    on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
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
    except ExecutionFailure as exc:
        if not emitted_any_chunk:
            raise
        _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
        for chunk in assembler.fail_execution(exc):
            yield chunk
    except BaseExceptionGroup as exc:
        if not emitted_any_chunk:
            raise
        failure = find_execution_failure(exc)
        if failure is not None:
            _observe_post_start_terminal_failure(
                on_post_start_terminal_failure, failure
            )
            for chunk in assembler.fail_execution(failure):
                yield chunk
        else:
            _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
            for chunk in assembler.fail_response(_unexpected_error_data(exc)):
                yield chunk
    except Exception as exc:
        if not emitted_any_chunk:
            raise
        _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
        for chunk in assembler.fail_response(_unexpected_error_data(exc)):
            yield chunk


def _observe_post_start_terminal_failure(
    observer: PostStartTerminalFailureObserver | None,
    exc: BaseException,
) -> None:
    if observer is not None:
        observer(exc)


def _unexpected_error_data(exc: BaseException) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "type": "api_error",
            "message": safe_exception_message(exc),
        }
    }
