"""Composite Anthropic SSE serialization for local web server tools."""

import json
from collections.abc import AsyncIterator

from free_claude_code.core.anthropic.streaming import format_sse_event


async def stream_composite_web_response(
    *,
    message_id: str,
    model: str,
    content: list[dict[str, object]],
    usage: dict[str, object],
    stop_reason: str,
) -> AsyncIterator[str]:
    """Emit one public Messages lifecycle from several internal model rounds."""
    yield message_start_frame(
        message_id=message_id,
        model=model,
        usage=usage,
    )
    async for frame in stream_content_blocks(content, start_index=0):
        yield frame
    for frame in message_end_frames(usage=usage, stop_reason=stop_reason):
        yield frame


def message_start_frame(
    *,
    message_id: str,
    model: str,
    usage: dict[str, object],
) -> str:
    """Serialize the single public lifecycle start."""
    start_usage = {key: value for key, value in usage.items() if key != "output_tokens"}
    start_usage["output_tokens"] = 1
    return format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": start_usage,
            },
        },
    )


async def stream_content_blocks(
    content: list[dict[str, object]],
    *,
    start_index: int,
) -> AsyncIterator[str]:
    """Serialize public blocks with monotonically increasing indexes."""
    for offset, block in enumerate(content):
        async for frame in _stream_content_block(start_index + offset, block):
            yield frame


def message_end_frames(
    *,
    usage: dict[str, object],
    stop_reason: str,
) -> tuple[str, str]:
    """Serialize the single public lifecycle completion."""
    return (
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    )


async def _stream_content_block(
    index: int,
    block: dict[str, object],
) -> AsyncIterator[str]:
    block_type = block.get("type")
    if block_type == "text":
        yield _block_start(index, {"type": "text", "text": ""})
        text = block.get("text")
        if isinstance(text, str) and text:
            yield _block_delta(index, {"type": "text_delta", "text": text})
    elif block_type == "thinking":
        yield _block_start(index, {"type": "thinking", "thinking": ""})
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking:
            yield _block_delta(index, {"type": "thinking_delta", "thinking": thinking})
        signature = block.get("signature")
        if isinstance(signature, str):
            yield _block_delta(
                index, {"type": "signature_delta", "signature": signature}
            )
    elif block_type == "server_tool_use":
        initial = dict(block)
        tool_input = initial.pop("input", {})
        initial["input"] = {}
        yield _block_start(index, initial)
        yield _block_delta(
            index,
            {
                "type": "input_json_delta",
                "partial_json": json.dumps(tool_input, separators=(",", ":")),
            },
        )
    else:
        yield _block_start(index, block)
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def _block_start(index: int, block: dict[str, object]) -> str:
    return format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        },
    )


def _block_delta(index: int, delta: dict[str, object]) -> str:
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )
