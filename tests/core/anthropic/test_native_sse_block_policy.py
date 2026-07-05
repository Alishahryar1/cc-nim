"""Unit tests for shared native Anthropic SSE thinking policy / block remapping."""

import json

from core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    format_native_sse_event,
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
