"""FIFO queue state for one messaging conversation tree."""

from __future__ import annotations

import asyncio
from collections import deque


class MessageNodeQueue:
    """Queue with snapshot/remove helpers, backed by a deque and a set index."""

    def __init__(self, items: list[str] | None = None) -> None:
        self._deque: deque[str] = deque()
        self._set: set[str] = set()
        for item in items or []:
            self.put_nowait(item)

    def put_nowait(self, item: str) -> None:
        self._deque.append(item)
        self._set.add(item)

    def get_nowait(self) -> str:
        if not self._deque:
            raise asyncio.QueueEmpty()
        item = self._deque.popleft()
        self._set.discard(item)
        return item

    def qsize(self) -> int:
        return len(self._deque)

    def snapshot(self) -> list[str]:
        return list(self._deque)

    def remove_if_present(self, item: str) -> bool:
        if item not in self._set:
            return False
        self._set.discard(item)
        self._deque = deque(x for x in self._deque if x != item)
        return True

    def drain(self) -> list[str]:
        items = list(self._deque)
        self._deque.clear()
        self._set.clear()
        return items
