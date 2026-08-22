"""Convert Anthropic SSE stream to OpenAI Chat Completions SSE format."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any


def _new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


async def chat_completions_sse_from_anthropic(
    anthropic_chunks: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    """Convert Anthropic SSE events to OpenAI Chat Completions SSE events."""
    chunk_id = _new_id()
    finish_reason: str | None = None
    tool_call_index = 0
    current_tool_call_id: str | None = None
    current_tool_call_name: str | None = None
    current_tool_call_args = ""
    buffer = ""

    async for raw_line in anthropic_chunks:
        buffer += raw_line
        lines = buffer.split("\n")
        buffer = lines.pop()

        for line in lines:
            line = line.rstrip("\r")
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError, TypeError:
                continue

            evt = data.get("type", "")

            if evt == "message_start":
                msg = data.get("message", {})
                if msg.get("model"):
                    model = msg["model"]
                continue

            if evt == "content_block_start":
                cb = data.get("content_block", {})
                bt = cb.get("type", "")
                if bt == "text":
                    yield _sse(chunk_id, model, 0, {"role": "assistant", "content": ""})
                elif bt == "tool_use":
                    current_tool_call_id = cb.get("id", "")
                    current_tool_call_name = cb.get("name", "")
                    current_tool_call_args = ""
                    tc = {
                        "index": tool_call_index,
                        "id": current_tool_call_id,
                        "type": "function",
                        "function": {"name": current_tool_call_name, "arguments": ""},
                    }
                    yield _sse(chunk_id, model, 0, {"tool_calls": [tc]})
                continue

            if evt == "content_block_delta":
                delta = data.get("delta", {})
                dt = delta.get("type", "")
                if dt == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield _sse(chunk_id, model, 0, {"content": text})
                elif dt == "input_json_delta":
                    pj = delta.get("partial_json", "")
                    if pj and current_tool_call_id:
                        current_tool_call_args += pj
                        tc = {"index": tool_call_index, "function": {"arguments": pj}}
                        yield _sse(chunk_id, model, 0, {"tool_calls": [tc]})
                continue

            if evt == "content_block_stop":
                if current_tool_call_id:
                    tool_call_index += 1
                    current_tool_call_id = None
                    current_tool_call_name = None
                    current_tool_call_args = ""
                continue

            if evt == "message_delta":
                sr = data.get("delta", {}).get("stop_reason")
                if sr:
                    finish_reason = {
                        "end_turn": "stop",
                        "stop_sequence": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                    }.get(sr, "stop")
                continue

            if evt == "message_stop":
                yield _sse_final(chunk_id, model, finish_reason or "stop")
                yield "data: [DONE]\n\n"
                return

            if evt == "error":
                yield _sse_final(chunk_id, model, "error")
                yield "data: [DONE]\n\n"
                return

    yield _sse_final(chunk_id, model, finish_reason or "stop")
    yield "data: [DONE]\n\n"


def _sse(chunk_id: str, model: str, index: int, delta: dict[str, Any]) -> str:
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _sse_final(chunk_id: str, model: str, finish_reason: str) -> str:
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"
