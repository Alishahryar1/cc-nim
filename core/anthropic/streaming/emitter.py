"""Anthropic SSE serialization helpers."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

ANTHROPIC_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    """Map OpenAI ``finish_reason`` values to Anthropic ``stop_reason`` values."""
    return (
        STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"
    )


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class AnthropicSseEmitter:
    """Serialize Anthropic SSE events and optionally log raw event bodies."""

    def __init__(self, *, log_raw_events: bool = False) -> None:
        self._log_raw_events = log_raw_events

    def event(self, event_type: str, data: dict[str, Any]) -> str:
        event = format_sse_event(event_type, data)
        if self._log_raw_events:
            logger.debug("SSE_EVENT: {} - {}", event_type, event.strip())
        return event
