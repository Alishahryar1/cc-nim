"""Platform-neutral voice note helpers."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import MessageScope


class Transcriber(Protocol):
    """Consumer-owned voice transcription boundary."""

    async def transcribe(self, file_path: Path) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PendingVoiceClaim:
    """Opaque ownership token for one pending voice-note generation."""

    scope: MessageScope
    voice_message_id: str
    claim_id: str


@dataclass(frozen=True, slots=True)
class VoiceCancellationResult:
    """Message IDs released by one successful voice cancellation."""

    voice_message_id: str
    status_message_id: str | None


@dataclass(slots=True)
class _PendingVoice:
    claim: PendingVoiceClaim
    status_message_id: str | None = None


class PendingVoiceRegistry:
    """Own atomic reservation, cancellation, and handoff of voice notes."""

    def __init__(self) -> None:
        self._pending: dict[tuple[MessageScope, str], _PendingVoice] = {}
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        scope: MessageScope,
        voice_message_id: str,
    ) -> PendingVoiceClaim | None:
        async with self._lock:
            key = (scope, voice_message_id)
            if key in self._pending:
                return None
            claim = PendingVoiceClaim(
                scope=scope,
                voice_message_id=voice_message_id,
                claim_id=uuid4().hex,
            )
            self._pending[key] = _PendingVoice(claim=claim)
            return claim

    async def bind_status(
        self,
        claim: PendingVoiceClaim,
        status_message_id: str,
    ) -> bool:
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if entry is None:
                return False
            if entry.status_message_id is not None:
                return entry.status_message_id == status_message_id
            status_key = (claim.scope, status_message_id)
            existing = self._pending.get(status_key)
            if existing is not None and existing is not entry:
                return False
            entry.status_message_id = status_message_id
            self._pending[status_key] = entry
            return True

    async def claim_for_handoff(self, claim: PendingVoiceClaim) -> bool:
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if entry is None or entry.status_message_id is None:
                return False
            self._remove(entry)
            return True

    async def discard(self, claim: PendingVoiceClaim) -> bool:
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if entry is None:
                return False
            self._remove(entry)
            return True

    async def cancel(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        async with self._lock:
            entry = self._pending.get((scope, reply_id))
            if entry is None:
                return None
            self._remove(entry)
            return VoiceCancellationResult(
                voice_message_id=entry.claim.voice_message_id,
                status_message_id=entry.status_message_id,
            )

    def _entry_for_claim(self, claim: PendingVoiceClaim) -> _PendingVoice | None:
        entry = self._pending.get((claim.scope, claim.voice_message_id))
        if entry is None or entry.claim != claim:
            return None
        return entry

    def _remove(self, entry: _PendingVoice) -> None:
        voice_key = (entry.claim.scope, entry.claim.voice_message_id)
        if self._pending.get(voice_key) is entry:
            self._pending.pop(voice_key)
        if entry.status_message_id is None:
            return
        status_key = (entry.claim.scope, entry.status_message_id)
        if self._pending.get(status_key) is entry:
            self._pending.pop(status_key)
