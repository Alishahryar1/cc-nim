"""Platform-neutral voice note helpers."""

import asyncio
from pathlib import Path
from typing import Protocol

from .models import MessageScope


class Transcriber(Protocol):
    """Consumer-owned voice transcription boundary."""

    async def transcribe(self, file_path: Path) -> str: ...

    async def close(self) -> None: ...


class PendingVoiceRegistry:
    """Track voice notes that are still waiting on transcription."""

    def __init__(self) -> None:
        self._pending: dict[tuple[MessageScope, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, scope: MessageScope, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            entry = (voice_msg_id, status_msg_id)
            self._pending[(scope, voice_msg_id)] = entry
            self._pending[(scope, status_msg_id)] = entry

    async def cancel(
        self, scope: MessageScope, reply_id: str
    ) -> tuple[str, str] | None:
        async with self._lock:
            entry = self._pending.pop((scope, reply_id), None)
            if entry is None:
                return None
            voice_msg_id, status_msg_id = entry
            self._pending.pop((scope, voice_msg_id), None)
            self._pending.pop((scope, status_msg_id), None)
            return entry

    async def is_pending(self, scope: MessageScope, voice_msg_id: str) -> bool:
        async with self._lock:
            return (scope, voice_msg_id) in self._pending

    async def complete(
        self, scope: MessageScope, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            self._pending.pop((scope, voice_msg_id), None)
            self._pending.pop((scope, status_msg_id), None)
