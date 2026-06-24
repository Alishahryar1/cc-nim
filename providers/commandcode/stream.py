"""Native Command Code SSE stream parser for /alpha/generate."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from loguru import logger


def _dump_sse(event_type: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


class CommandCodeStreamRunner:
    """Parses CCStreamEvents from /alpha/generate and yields Anthropic SSE blocks."""

    def __init__(
        self,
        response_iterator: AsyncIterator[str],
        request_id: str | None,
        model: str,
    ):
        self._iterator = response_iterator
        self._request_id = request_id or "msg_cc_default"
        self._model = model

        self._block_index = 0
        self._in_text_block = False
        self._current_tool_id: str | None = None

        self._tool_call_indexes: dict[str, int] = {}

    async def run(self) -> AsyncIterator[str]:
        """Consume the response stream and yield native Anthropic SSE chunks."""
        # Always start with message_start
        yield _dump_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._request_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self._model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        async def _json_stream_parser() -> AsyncIterator[dict]:
            buffer = ""
            depth = 0
            in_string = False
            escape = False

            async for chunk in self._iterator:
                for char in chunk:
                    buffer += char

                    if not in_string:
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                import contextlib

                                with contextlib.suppress(json.JSONDecodeError):
                                    yield json.loads(buffer)
                                buffer = ""
                        elif char == '"':
                            in_string = True
                    else:
                        if escape:
                            escape = False
                        elif char == "\\":
                            escape = True
                        elif char == '"':
                            in_string = False

        async for event in _json_stream_parser():
            event_type = event.get("type")

            if not event_type and "error" in event:
                # Handle direct JSON error responses from API
                err = event.get("error", {})
                msg = err.get("message", "Unknown CC error")
                logger.error("CC API returned error in JSON: {}", msg)
                yield _dump_sse(
                    "error",
                    {"type": "error", "error": {"type": "api_error", "message": msg}},
                )
                break

            if event_type == "text-delta":
                if not self._in_text_block:
                    if self._in_text_block or self._current_tool_id:
                        yield _dump_sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": self._block_index},
                        )
                        self._block_index += 1
                        self._in_text_block = False
                        self._current_tool_id = None

                    yield _dump_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self._block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    self._in_text_block = True

                text = event.get("text", "")
                if text:
                    yield _dump_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._block_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )

            elif event_type in ("tool-use", "tool-input-start", "tool-call"):
                tool_id = event.get("toolCallId") or event.get("id")
                tool_name = event.get("toolName")

                if tool_id and tool_id not in self._tool_call_indexes:
                    if self._in_text_block or self._current_tool_id:
                        yield _dump_sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": self._block_index},
                        )
                        self._block_index += 1
                        self._in_text_block = False
                        self._current_tool_id = None

                    self._current_tool_id = tool_id
                    self._tool_call_indexes[tool_id] = self._block_index

                    yield _dump_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self._block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": {},
                            },
                        },
                    )

                    # If this is a tool-call with complete input, we stream it as a delta immediately
                    input_obj = event.get("input")
                    if input_obj is not None:
                        input_str = json.dumps(input_obj)
                        yield _dump_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": input_str,
                                },
                            },
                        )

            elif event_type in ("tool-delta", "tool-input-delta"):
                tool_id = (
                    event.get("id") or event.get("toolCallId") or self._current_tool_id
                )
                if tool_id in self._tool_call_indexes:
                    idx = self._tool_call_indexes[tool_id]
                    delta = event.get("delta") or event.get("text", "")
                    if delta:
                        yield _dump_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": delta,
                                },
                            },
                        )

            elif event_type == "finish":
                if self._in_text_block or self._current_tool_id:
                    yield _dump_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": self._block_index},
                    )
                    self._block_index += 1
                    self._in_text_block = False
                    self._current_tool_id = None

                reason = event.get("finishReason", "stop")
                if reason in ("tool_calls", "tool-calls"):
                    reason = "tool_use"
                elif reason in ("length", "max_tokens"):
                    reason = "max_tokens"
                else:
                    reason = "end_turn"

                usage = event.get("totalUsage", {})
                yield _dump_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": None},
                        "usage": {"output_tokens": usage.get("outputTokens", 0)},
                    },
                )
                yield _dump_sse("message_stop", {"type": "message_stop"})
                break

            elif event_type == "error":
                err_data = event.get("error", {})
                msg = err_data.get("message", "Unknown CommandCode error")
                logger.error("CommandCode stream error: {}", msg)
                yield _dump_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": msg},
                    },
                )
                break
