"""Translate Anthropic SSE streams into OpenAI Responses SSE streams."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from .anthropic_sse import iter_sse_events
from .stream_state import ResponsesStreamAssembler


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""

    assembler = ResponsesStreamAssembler(request)
    async for event in iter_sse_events(chunks):
        for chunk in assembler.process_anthropic_event(event):
            yield chunk
        if assembler.terminal:
            return
    for chunk in assembler.finish_if_needed():
        yield chunk
