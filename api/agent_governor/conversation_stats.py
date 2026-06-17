"""Pure helpers for inspecting conversation history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class ToolCallRef:
    """Lightweight reference to a single tool_use block."""

    name: str
    args_hash: str


@dataclass(slots=True)
class ConversationStats:
    """Summary of tool-call patterns in a conversation."""

    total_messages: int = 0
    total_tool_calls: int = 0
    consecutive_tool_calls: int = 0
    recent_tool_calls: list[ToolCallRef] = field(default_factory=list)
    last_assistant_text_len: int = 0


def _iter_content_blocks(message: Any) -> Iterable[Any]:
    """Yield content blocks from a Message; tolerate str-content."""
    content = getattr(message, "content", None)
    if isinstance(content, list):
        yield from content


def _hash_args(args: Any) -> str:
    """Stable short hash of tool arguments for loop detection."""
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        canonical = repr(args)
    return hashlib.sha1(canonical.encode("utf-8", errors="replace")).hexdigest()[:12]


def compute_stats(messages: list[Any], *, recent_window: int = 8) -> ConversationStats:
    """Walk messages, count tool_use blocks, capture recent tool signatures.

    Counts:
    * ``total_tool_calls``: every tool_use block across the full history.
    * ``consecutive_tool_calls``: tool_use blocks at the tail without an
      intervening assistant text-only turn or fresh user message.
    * ``recent_tool_calls``: last ``recent_window`` (name, args_hash) pairs,
      newest last — used by the loop detector.
    * ``last_assistant_text_len``: chars of text in the most recent assistant
      message (zero if none, or if it was tool-only). Useful proxy for whether
      the model has produced a real answer recently.
    """
    stats = ConversationStats(total_messages=len(messages))
    consecutive_streak = 0
    recent: list[ToolCallRef] = []
    last_text_len = 0
    last_assistant_seen = False

    for message in messages:
        role = getattr(message, "role", None)
        tool_uses_in_message: list[ToolCallRef] = []
        tool_results_in_message = 0
        text_chars = 0
        for block in _iter_content_blocks(message):
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                ref = ToolCallRef(
                    name=getattr(block, "name", "?") or "?",
                    args_hash=_hash_args(getattr(block, "input", {})),
                )
                tool_uses_in_message.append(ref)
            elif block_type == "tool_result":
                tool_results_in_message += 1
            elif block_type == "text":
                text_value = getattr(block, "text", "") or ""
                text_chars += len(text_value)

        stats.total_tool_calls += len(tool_uses_in_message)
        recent.extend(tool_uses_in_message)

        if role == "assistant":
            last_assistant_seen = True
            last_text_len = text_chars
            if tool_uses_in_message:
                # Continue or start a streak. Whether it was broken before is
                # irrelevant; we count the trailing run.
                consecutive_streak += len(tool_uses_in_message)
            else:
                # assistant text-only turn ends the trailing run.
                consecutive_streak = 0
        elif role == "user":
            # Pure tool_result turns are part of the agent loop.
            # Anything else (real user text, mixed content, image, etc.)
            # is a fresh user turn that resets the streak.
            has_only_tool_results = (
                tool_results_in_message > 0 and text_chars == 0
            )
            if not has_only_tool_results:
                consecutive_streak = 0

    stats.consecutive_tool_calls = consecutive_streak
    stats.recent_tool_calls = recent[-recent_window:]
    stats.last_assistant_text_len = last_text_len if last_assistant_seen else 0
    return stats
