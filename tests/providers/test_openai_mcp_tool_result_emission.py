"""Unit tests for the Anthropic ``tool_result`` block emitter on the ledger.

Required because the M3 fix replaces a plain-text fallback with proper
Anthropic SSE ``tool_result`` blocks. These tests pin the wire shape.

The stream ledger module loads cleanly without the missing
``free_claude_code.providers.admission`` module, so we import it directly.
"""

import json

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming.ledger import (
    AnthropicStreamLedger,
)


def _new_ledger() -> AnthropicStreamLedger:
    return AnthropicStreamLedger(message_id="msg_test", model="claude-test")


def _frame(event: str) -> dict:
    """Parse one SSE event line into a dict preserving event/data."""
    lines = [ln for ln in event.splitlines() if ln.startswith(("event:", "data:"))]
    assert len(lines) == 2, f"unexpected SSE shape: {event!r}"
    etype = lines[0].split(":", 1)[1].strip()
    data = json.loads(lines[1].split(":", 1)[1].strip())
    return {"event": etype, "data": data}


def test_emit_tool_result_block_basic_shape() -> None:
    ledger = _new_ledger()
    events = list(
        ledger.emit_tool_result_block(
            tool_use_id="toolu_v1",
            content="permission_denied",
            is_error=True,
        )
    )
    frames = [_frame(e) for e in events]
    kinds = [f["event"] for f in frames]
    assert kinds == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    start = frames[0]["data"]["content_block"]
    assert start["type"] == "tool_result"
    assert start["tool_use_id"] == "toolu_v1"
    assert start["content"] == "permission_denied"
    assert start["is_error"] is True
    assert frames[0]["data"]["index"] == frames[2]["data"]["index"]


def test_emit_tool_result_block_default_is_error_false() -> None:
    ledger = _new_ledger()
    events = list(
        ledger.emit_tool_result_block(tool_use_id="toolu_v2", content='{"ok": true}')
    )
    start = _frame(events[0])["data"]["content_block"]
    assert start["is_error"] is False


def test_tool_result_block_emits_via_parse_sse_text() -> None:
    """Round-trip the emitter output through ``parse_sse_text``.

    Ensures the block can be consumed by the same downstream parser used in
    ``test_streaming_errors.py``.
    """
    ledger = _new_ledger()
    raw = "".join(
        ledger.emit_tool_result_block(
            tool_use_id="call_xyz", content="denied by policy", is_error=True
        )
    )
    parsed = parse_sse_text(raw)
    assert parsed, "parse_sse_text accepted no events from the emitter"
    kinds = [p.event for p in parsed]
    assert kinds == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    block = parsed[0].data["content_block"]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_xyz"


def test_tool_result_blocks_are_unique_per_emit() -> None:
    """Each emission must allocate a fresh block index — never reuse."""
    ledger = _new_ledger()
    first = list(ledger.emit_tool_result_block(tool_use_id="a", content="x"))
    second = list(ledger.emit_tool_result_block(tool_use_id="b", content="y"))
    idx_0 = _frame(first[0])["data"]["index"]
    idx_1 = _frame(second[0])["data"]["index"]
    assert idx_0 != idx_1
