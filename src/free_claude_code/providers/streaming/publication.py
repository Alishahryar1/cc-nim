"""Policy-free canonical event holdback and publication state."""

import time
from collections.abc import Callable

from free_claude_code.core.inference import (
    InferenceEvent,
    ResponseStarted,
    inference_event_size,
)

EARLY_HOLDBACK_SECONDS = 0.75
RECOVERY_BUFFER_MAX_BYTES = 65_536


class PublicationBuffer:
    """Briefly retain canonical output before its irreversible publication."""

    def __init__(
        self,
        *,
        holdback_seconds: float = EARLY_HOLDBACK_SECONDS,
        max_bytes: int = RECOVERY_BUFFER_MAX_BYTES,
        now: Callable[[], float] | None = None,
    ) -> None:
        if holdback_seconds < 0:
            raise ValueError("holdback_seconds must be >= 0")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._holdback_seconds = holdback_seconds
        self._max_bytes = max_bytes
        self._now = now or time.monotonic
        self._events: list[InferenceEvent] = []
        self._bytes = 0
        self._deadline: float | None = None
        self._published = False

    @property
    def published(self) -> bool:
        """Return whether canonical output has crossed the provider boundary."""
        return self._published

    @property
    def has_buffered(self) -> bool:
        return bool(self._events)

    @property
    def seconds_until_deadline(self) -> float | None:
        """Return the remaining active holdback delay, if one has started."""
        if self._published or self._deadline is None:
            return None
        return max(0.0, self._deadline - self._now())

    def push(self, event: InferenceEvent) -> list[InferenceEvent]:
        """Buffer one event or return events that are already publishable."""
        if self._published:
            return [event]
        if self._deadline is None and not isinstance(event, ResponseStarted):
            self._deadline = self._now() + self._holdback_seconds
        self._events.append(event)
        self._bytes += inference_event_size(event)
        if self._bytes >= self._max_bytes:
            return self.flush()
        return []

    def flush(self) -> list[InferenceEvent]:
        """Publish every held event exactly once."""
        if self._published:
            return []
        self._published = True
        events = self._events
        self._events = []
        self._bytes = 0
        self._deadline = None
        return events

    def discard(self) -> None:
        """Discard an unpublished attempt epoch and reset its holdback clock."""
        if self._published:
            return
        self._events.clear()
        self._bytes = 0
        self._deadline = None
