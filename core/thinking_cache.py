"""Server-side LRU cache for DeepSeek thinking content.

DeepSeek-v4-pro always generates thinking blocks, but Claude Code 2.1+ strips
them from conversation history before sending to the API.  This cache lets the
proxy recover the real thinking text when injecting placeholders for prior turns,
replacing the generic "(prior reasoning not available)" with actual content.

Cache key  : frozenset of tool_use IDs produced during that assistant turn.
Cache value: the complete thinking text accumulated during that turn's stream.

IDs are unique per turn (DeepSeek assigns them), so the lookup is collision-free
as long as the history spans fewer than MAX_SIZE completed turns.
"""
from __future__ import annotations

from collections import OrderedDict

_MAX_SIZE = 300


class ThinkingCache:
    """LRU cache mapping tool_use ID sets → thinking text."""

    _lru: OrderedDict[frozenset, str] = OrderedDict()

    @classmethod
    def store(cls, tool_ids: frozenset[str], thinking: str) -> None:
        """Store thinking text for a turn identified by its tool_use IDs."""
        if not tool_ids or not thinking:
            return
        if tool_ids in cls._lru:
            cls._lru.move_to_end(tool_ids)
        else:
            if len(cls._lru) >= _MAX_SIZE:
                cls._lru.popitem(last=False)
        cls._lru[tool_ids] = thinking

    @classmethod
    def lookup(cls, tool_ids: frozenset[str]) -> str | None:
        """Return cached thinking for a given set of tool_use IDs, or None."""
        if not tool_ids:
            return None
        val = cls._lru.get(tool_ids)
        if val is not None:
            cls._lru.move_to_end(tool_ids)
        return val
