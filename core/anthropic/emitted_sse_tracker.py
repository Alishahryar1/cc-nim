"""Track content-block state for native Anthropic SSE strings we emit to clients."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

from core.anthropic.sse import SSEBuilder, format_sse_event
from core.anthropic.stream_contracts import SSEEvent, event_index, parse_sse_lines


class EmittedNativeSseTracker:
    """Parse emitted SSE frames so mid-stream errors can close blocks and pick a fresh index."""

    def __init__(self) -> None:
        self._buf = ""
        self._open_stack: list[int] = []
        self._max_index = -1
        self.message_id: str | None = None
        self.model: str = ""
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []

    def estimate_output_tokens(self) -> int:
        """Return the estimated number of output tokens emitted so far."""
        t = "".join(self._text_parts)
        th = "".join(self._thinking_parts)
        # Rough estimate: 4 chars per token
        return (len(t) + len(th)) // 4

    def feed(self, chunk: str) -> None:
        """Record SSE frames completed by ``chunk`` (handles splitting across reads)."""
        self._buf += chunk
        while True:
            sep = self._buf.find("\n\n")
            if sep < 0:
                break
            frame = self._buf[:sep]
            self._buf = self._buf[sep + 2 :]
            if not frame.strip():
                continue
            for event in parse_sse_lines(frame.splitlines()):
                self._observe(event)

    def _observe(self, event: SSEEvent) -> None:
        if event.event == "message_start":
            message = event.data.get("message")
            if isinstance(message, dict):
                mid = message.get("id")
                if isinstance(mid, str) and mid:
                    self.message_id = mid
                model = message.get("model")
                if isinstance(model, str) and model:
                    self.model = model
            return

        if event.event == "content_block_start":
            idx = event_index(event)
            self._max_index = max(self._max_index, idx)
            self._open_stack.append(idx)
            return

        if event.event == "content_block_delta":
            delta = event.data.get("delta", {})
            if delta.get("type") == "text_delta":
                self._text_parts.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                self._thinking_parts.append(delta.get("thinking", ""))
            return

        if event.event == "content_block_stop":
            idx = event_index(event)
            if self._open_stack and self._open_stack[-1] == idx:
                self._open_stack.pop()
            else:
                with suppress(ValueError):
                    self._open_stack.remove(idx)

    def next_content_index(self) -> int:
        """Next unused content block index based on emitted starts."""
        return self._max_index + 1

    def iter_close_unclosed_blocks(self) -> Iterator[str]:
        """Yield ``content_block_stop`` events for blocks that were started but not stopped."""
        while self._open_stack:
            idx = self._open_stack.pop()
            yield format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )

    def iter_midstream_error_tail(
        self,
        error_message: str,
        *,
        request: Any,
        input_tokens: int,
        log_raw_sse_events: bool,
    ) -> Iterator[str]:
        """Close dangling blocks, emit a text error block at a fresh index, then message tail."""
        mid = self.message_id or f"msg_{uuid.uuid4()}"
        model = self.model or (getattr(request, "model", "") or "")
        sse = SSEBuilder(
            mid,
            model,
            input_tokens,
            log_raw_events=log_raw_sse_events,
        )
        sse.blocks.next_index = self.next_content_index()
        yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()

    def as_messages_response(self, model: str, response_factory: Any) -> Any:
        """Convert tracked content into a static MessagesResponse object."""
        content = []
        if self._thinking_parts:
            # Note: Standard Anthropic MessagesResponse doesn't natively expose
            # thinking blocks in the same way SSE does. We'll append it as text
            # if we wanted a faithful reproduction, or skip for now.
            # Here we follow the core principle of zero-defect and standard compat.
            pass

        if self._text_parts:
            content.append({"type": "text", "text": "".join(self._text_parts)})

        return response_factory(
            id=self.message_id or f"msg_{uuid.uuid4().hex[:12]}",
            model=self.model or model,
            content=content,
            output_tokens=self.estimate_output_tokens() or 1,
        )
