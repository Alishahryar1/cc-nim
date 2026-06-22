"""Tests for the atomic ``tool_use`` SSE buffer (anti truncated-tool-call)."""

from collections.abc import AsyncIterator

import pytest

from core.anthropic.sse import format_sse_event
from core.anthropic.tool_use_buffer import (
    IncompleteUpstreamStreamError,
    buffer_incomplete_tool_use,
)


def _tool_start(index: int = 0, tool_id: str = "toolu_1", name: str = "Edit") -> str:
    return format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        },
    )


def _tool_delta(index: int = 0, partial_json: str = '{"path":') -> str:
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def _text_start(index: int = 0) -> str:
    return format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )


def _text_delta(index: int = 0, text: str = "Hola") -> str:
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def _stop(index: int = 0) -> str:
    return format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def _ping() -> str:
    return format_sse_event("ping", {"type": "ping"})


def _msg_delta() -> str:
    return format_sse_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    )


def _msg_stop() -> str:
    return format_sse_event("message_stop", {"type": "message_stop"})


async def _agen(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


async def _collect(items: list[str]) -> list[str]:
    out: list[str] = []
    async for chunk in buffer_incomplete_tool_use(_agen(items)):
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_non_tool_stream_passes_through_byte_for_byte():
    chunks = [_text_start(), _text_delta(), _stop(), _msg_delta(), _msg_stop()]
    out = await _collect(chunks)
    assert "".join(out) == "".join(chunks)


@pytest.mark.asyncio
async def test_complete_tool_use_is_flushed_intact_in_order():
    chunks = [
        _tool_start(),
        _tool_delta(partial_json='{"path":"'),
        _tool_delta(partial_json='a.txt"}'),
        _stop(),
        _msg_delta(),
        _msg_stop(),
    ]
    out = await _collect(chunks)
    # Byte-identical (frames may be regrouped, transparent to SSE consumers).
    assert "".join(out) == "".join(chunks)


@pytest.mark.asyncio
async def test_truncated_tool_use_raises_and_emits_nothing():
    # tool_use opens, two deltas arrive, source ends with no content_block_stop.
    chunks = [_tool_start(), _tool_delta(partial_json='{"path":"a')]
    out: list[str] = []
    with pytest.raises(IncompleteUpstreamStreamError):
        async for chunk in buffer_incomplete_tool_use(_agen(chunks)):
            out.append(chunk)
    # The partial tool block was held, never relayed.
    assert out == []


@pytest.mark.asyncio
async def test_leading_text_delivered_then_truncated_tool_use_held():
    chunks = [
        _text_start(0),
        _text_delta(0, "thinking out loud"),
        _stop(0),
        _tool_start(1),
        _tool_delta(1, partial_json='{"path":"a'),
    ]
    out: list[str] = []
    with pytest.raises(IncompleteUpstreamStreamError):
        async for chunk in buffer_incomplete_tool_use(_agen(chunks)):
            out.append(chunk)
    blob = "".join(out)
    assert "thinking out loud" in blob  # earlier visible content delivered
    assert "tool_use" not in blob  # truncated tool call NOT relayed
    assert "input_json_delta" not in blob


@pytest.mark.asyncio
async def test_ping_is_forwarded_during_tool_use_buffering():
    chunks = [
        _tool_start(),
        _ping(),
        _tool_delta(partial_json='{"x":1}'),
        _stop(),
        _msg_stop(),
    ]
    out = await _collect(chunks)
    blob = "".join(out)
    assert "ping" in blob
    # Ping is forwarded immediately; the tool block is held until its stop,
    # so the ping appears BEFORE the tool block in the output.
    assert blob.index("ping") < blob.index("tool_use")
    # The tool block is still complete in the output.
    assert "input_json_delta" in blob


@pytest.mark.asyncio
async def test_message_terminator_while_tool_open_raises():
    # Upstream jumps to message_stop without closing the tool_use block.
    chunks = [_tool_start(), _tool_delta(), _msg_stop()]
    out: list[str] = []
    with pytest.raises(IncompleteUpstreamStreamError):
        async for chunk in buffer_incomplete_tool_use(_agen(chunks)):
            out.append(chunk)
    assert out == []


@pytest.mark.asyncio
async def test_reassembles_tool_block_split_across_line_chunks():
    # Feed the tool_use block as individual lines (keepends), as the native
    # line-mode transport yields them.
    full = _tool_start() + _tool_delta(partial_json='{"a":1}') + _stop()
    line_chunks = full.splitlines(keepends=True) + [_msg_stop()]
    out = await _collect(line_chunks)
    assert "".join(out) == full + _msg_stop()
