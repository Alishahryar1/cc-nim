"""Tests for ConversationStats — tool-call counters and recent signatures."""

from __future__ import annotations

from api.models.anthropic import (
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
)
from api.agent_governor.conversation_stats import compute_stats


def _user_text(text: str) -> Message:
    return Message(role="user", content=[ContentBlockText(type="text", text=text)])


def _assistant_text(text: str) -> Message:
    return Message(
        role="assistant", content=[ContentBlockText(type="text", text=text)]
    )


def _assistant_tool_use(name: str, args: dict, tool_id: str = "t") -> Message:
    return Message(
        role="assistant",
        content=[
            ContentBlockToolUse(
                type="tool_use", id=tool_id, name=name, input=args
            )
        ],
    )


def _user_tool_result(tool_use_id: str, content: str) -> Message:
    return Message(
        role="user",
        content=[
            ContentBlockToolResult(
                type="tool_result", tool_use_id=tool_use_id, content=content
            )
        ],
    )


def test_empty_history():
    stats = compute_stats([])
    assert stats.total_messages == 0
    assert stats.total_tool_calls == 0
    assert stats.consecutive_tool_calls == 0
    assert stats.recent_tool_calls == []


def test_single_user_message_no_tools():
    stats = compute_stats([_user_text("hi")])
    assert stats.total_tool_calls == 0
    assert stats.consecutive_tool_calls == 0


def test_text_assistant_resets_streak():
    msgs = [
        _user_text("do thing"),
        _assistant_tool_use("read", {"path": "a"}, "t1"),
        _user_tool_result("t1", "ok"),
        _assistant_text("done"),
    ]
    stats = compute_stats(msgs)
    assert stats.total_tool_calls == 1
    assert stats.consecutive_tool_calls == 0
    assert stats.last_assistant_text_len == 4


def test_consecutive_tool_calls_with_tool_results():
    msgs = [
        _user_text("do thing"),
        _assistant_tool_use("read", {"path": "a"}, "t1"),
        _user_tool_result("t1", "ok"),
        _assistant_tool_use("read", {"path": "b"}, "t2"),
        _user_tool_result("t2", "ok"),
        _assistant_tool_use("read", {"path": "c"}, "t3"),
        _user_tool_result("t3", "ok"),
    ]
    stats = compute_stats(msgs)
    assert stats.total_tool_calls == 3
    assert stats.consecutive_tool_calls == 3


def test_user_text_resets_streak():
    msgs = [
        _assistant_tool_use("read", {"path": "a"}, "t1"),
        _user_tool_result("t1", "ok"),
        _user_text("new question"),
        _assistant_tool_use("read", {"path": "b"}, "t2"),
    ]
    stats = compute_stats(msgs)
    assert stats.total_tool_calls == 2
    assert stats.consecutive_tool_calls == 1


def test_recent_tool_calls_window():
    msgs = []
    for i in range(12):
        msgs.append(_assistant_tool_use("read", {"path": f"f{i}"}, f"t{i}"))
        msgs.append(_user_tool_result(f"t{i}", "ok"))
    stats = compute_stats(msgs, recent_window=5)
    assert len(stats.recent_tool_calls) == 5
    # The window should contain the LAST 5 calls (f7..f11).
    names = [ref.name for ref in stats.recent_tool_calls]
    assert names == ["read", "read", "read", "read", "read"]
