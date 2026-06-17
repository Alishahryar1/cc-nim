"""Detect repeated tool calls (same tool + same args) in a conversation tail."""

from __future__ import annotations

from .conversation_stats import ConversationStats, ToolCallRef


def detect_repeated_signature(
    stats: ConversationStats, *, threshold: int
) -> ToolCallRef | None:
    """Return the ToolCallRef seen ``threshold``+ times in recent_tool_calls."""
    if threshold <= 1 or not stats.recent_tool_calls:
        return None
    counts: dict[tuple[str, str], int] = {}
    for ref in stats.recent_tool_calls:
        key = (ref.name, ref.args_hash)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= threshold:
            return ref
    return None
