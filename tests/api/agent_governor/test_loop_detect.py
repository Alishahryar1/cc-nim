"""Tests for the same-tool same-args loop detector."""

from __future__ import annotations

from api.agent_governor.conversation_stats import ConversationStats, ToolCallRef
from api.agent_governor.loop_detect import detect_repeated_signature


def _stats(*refs: ToolCallRef) -> ConversationStats:
    return ConversationStats(
        total_messages=len(refs),
        total_tool_calls=len(refs),
        consecutive_tool_calls=len(refs),
        recent_tool_calls=list(refs),
    )


def test_no_repeats():
    s = _stats(
        ToolCallRef("read", "a1"),
        ToolCallRef("write", "b2"),
        ToolCallRef("bash", "c3"),
    )
    assert detect_repeated_signature(s, threshold=3) is None


def test_three_repeats_of_same_tool_same_args():
    ref = ToolCallRef("read", "abc")
    s = _stats(ref, ref, ref)
    out = detect_repeated_signature(s, threshold=3)
    assert out is not None
    assert out.name == "read"


def test_same_tool_different_args_does_not_trigger():
    s = _stats(
        ToolCallRef("read", "x1"),
        ToolCallRef("read", "x2"),
        ToolCallRef("read", "x3"),
    )
    assert detect_repeated_signature(s, threshold=3) is None


def test_threshold_below_two_disables():
    ref = ToolCallRef("read", "abc")
    s = _stats(ref, ref, ref)
    assert detect_repeated_signature(s, threshold=1) is None
    assert detect_repeated_signature(s, threshold=0) is None
