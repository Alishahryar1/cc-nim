"""FastAPI streaming response wrappers for public API wire formats."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

from fastapi.responses import StreamingResponse

from core.anthropic.streaming import ANTHROPIC_SSE_RESPONSE_HEADERS


def anthropic_sse_streaming_response(body: AsyncIterator[str]) -> StreamingResponse:
    """Return a streaming response for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def openai_responses_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
) -> StreamingResponse:
    """Return a streaming response for OpenAI Responses-style SSE."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=dict(headers),
    )
