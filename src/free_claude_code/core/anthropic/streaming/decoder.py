"""Incremental framing for Anthropic-compatible SSE streams."""

import re

from ..stream_contracts import SSEEvent, parse_sse_text

_EVENT_BOUNDARY = re.compile(r"\r?\n\r?\n")


class AnthropicSSEDecoder:
    """Decode arbitrarily split SSE text without losing frame order."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> tuple[SSEEvent, ...]:
        """Consume one wire chunk and return every complete event."""

        self._buffer += chunk
        events: list[SSEEvent] = []
        while match := _EVENT_BOUNDARY.search(self._buffer):
            boundary_end = match.end()
            raw = self._buffer[:boundary_end]
            self._buffer = self._buffer[boundary_end:]
            events.extend(parse_sse_text(raw))
        return tuple(events)

    def finish(self) -> tuple[SSEEvent, ...]:
        """Return a final unterminated event, if one is present."""

        remainder = self._buffer
        self._buffer = ""
        if not remainder.strip():
            return ()
        return tuple(parse_sse_text(remainder))
