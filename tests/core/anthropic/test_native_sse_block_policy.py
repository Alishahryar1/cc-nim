"""Unit tests for shared native Anthropic SSE thinking policy / block remapping."""

import json

from free_claude_code.core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    format_native_sse_event,
    sanitize_tool_name,
    transform_native_sse_block_event,
)


def test_thinking_start_dropped_when_disabled() -> None:
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    }
    ev = format_native_sse_event(
        "content_block_start",
        json.dumps(payload),
    )
    assert transform_native_sse_block_event(ev, st, thinking_enabled=False) is None


def test_thinking_delta_dropped_when_disabled() -> None:
    st = NativeSseBlockPolicyState()
    # No prior start in stream (OpenRouter-style: returns None when thinking off)
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "secret"},
    }
    ev = format_native_sse_event("content_block_delta", json.dumps(payload))
    assert transform_native_sse_block_event(ev, st, thinking_enabled=False) is None


def test_text_block_passthrough_when_thinking_disabled() -> None:
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    ev = format_native_sse_event("content_block_start", json.dumps(payload))
    out = transform_native_sse_block_event(ev, st, thinking_enabled=False)
    assert out is not None
    assert '"index": 0' in (out or "")


def test_interleaved_thinking_signature_delta_remaps_to_reopened_block_index() -> None:
    """After text interrupts thinking, signature_delta must follow the reopened segment index."""
    st = NativeSseBlockPolicyState()

    def run(ev: str) -> str | None:
        return transform_native_sse_block_event(ev, st, thinking_enabled=True)

    out1 = run(
        format_native_sse_event(
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            ),
        )
    )
    assert out1 is not None and '"index": 0' in out1

    out2 = run(
        format_native_sse_event(
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
        )
    )
    assert out2 is not None

    out3 = run(
        format_native_sse_event(
            "content_block_delta",
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "plan"},
                }
            ),
        )
    )
    assert out3 is not None
    assert "content_block_start" in out3

    out4 = run(
        format_native_sse_event(
            "content_block_delta",
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                }
            ),
        )
    )
    assert out4 is not None
    assert '"index": 2' in out4
    assert "signature_delta" in out4


def test_startless_text_delta_synthesizes_start_when_thinking_disabled() -> None:
    """Startless text deltas must not be dropped when thinking is disabled (OpenRouter)."""
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    ev = format_native_sse_event("content_block_delta", json.dumps(payload))
    out = transform_native_sse_block_event(ev, st, thinking_enabled=False)
    assert out is not None
    assert "content_block_start" in (out or "")
    assert "Hello" in (out or "")
    assert "text_delta" in (out or "")


def test_sanitize_tool_name_strips_leaked_control_token() -> None:
    """Mistral-family chat templates occasionally leak [TOOL_CALLS] into the name."""
    assert sanitize_tool_name("[TOOL_CALLS]Read") == "Read"
    assert sanitize_tool_name("[TOOL_CALLS][/TOOL_CALLS]Write") == "Write"
    assert sanitize_tool_name("Read") == "Read"
    assert sanitize_tool_name("") == ""


def test_tool_use_block_start_gets_sanitized_name() -> None:
    """A tool_use content_block_start with a corrupted name is cleaned in place."""
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "[TOOL_CALLS]Read",
        },
    }
    ev = format_native_sse_event("content_block_start", json.dumps(payload))
    out = transform_native_sse_block_event(ev, st, thinking_enabled=True)
    assert out is not None
    assert '"name": "Read"' in out
    assert "TOOL_CALLS" not in out
