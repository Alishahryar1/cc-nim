"""Multi-credential rotation state shared by rotating provider wrappers."""

from __future__ import annotations

import asyncio
import time

import httpx
import openai

from free_claude_code.providers.failure_policy import (
    retryable_transient_status,
    retryable_upstream_transport_error,
)

ROTATION_BACKOFF_SECONDS = 60.0
ROTATION_POLICIES = frozenset({"single", "round_robin", "on_error"})


def error_justifies_rotation(error: BaseException) -> bool:
    """Return True when trying a different credential may resolve the failure.

    Rotating is worthwhile for authentication problems, rate limits, upstream
    5xx/overload responses, and transport errors. A plain 400 invalid request
    will fail identically with every key, so it is not rotated.
    """
    if isinstance(error, openai.AuthenticationError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code in (401, 403):
            return True
    if retryable_transient_status(error) is not None:
        return True
    return retryable_upstream_transport_error(error)


class CredentialRotationState:
    """Pick which credential serves each request under a rotation policy.

    Policies:
      - ``single``: always the first key.
      - ``round_robin``: advance to the next healthy key on every request.
      - ``on_error``: stick to the current key until a failure backs it off,
        then move to the next healthy key.
    """

    def __init__(
        self,
        key_count: int,
        policy: str = "single",
        *,
        backoff_seconds: float = ROTATION_BACKOFF_SECONDS,
    ) -> None:
        if key_count <= 0:
            raise ValueError("key_count must be > 0")
        self._key_count = key_count
        self._policy = policy if policy in ROTATION_POLICIES else "single"
        self._backoff_seconds = backoff_seconds
        self._current = 0
        self._round_robin_next = 0
        self._backoff_until = [0.0] * key_count
        self._lock = asyncio.Lock()

    @property
    def policy(self) -> str:
        return self._policy

    def _is_healthy(self, index: int, now: float) -> bool:
        return now >= self._backoff_until[index]

    async def acquire(self) -> int:
        """Return the index of the credential to use for one new request."""
        async with self._lock:
            now = time.monotonic()
            if self._policy == "single" or self._key_count == 1:
                return 0

            if self._policy == "round_robin":
                for _ in range(self._key_count):
                    index = self._round_robin_next
                    self._round_robin_next = (self._round_robin_next + 1) % self._key_count
                    if self._is_healthy(index, now):
                        self._current = index
                        return index
                # Every key is backed off; hand out the next one anyway.
                index = self._round_robin_next
                self._round_robin_next = (self._round_robin_next + 1) % self._key_count
                self._current = index
                return index

            # on_error
            if self._is_healthy(self._current, now):
                return self._current
            for offset in range(1, self._key_count + 1):
                index = (self._current + offset) % self._key_count
                if self._is_healthy(index, now):
                    self._current = index
                    return index
            return self._current

    async def report_failure(self, index: int, error: BaseException) -> bool:
        """Back off one credential; return whether rotation is worthwhile."""
        rotate = error_justifies_rotation(error)
        async with self._lock:
            if 0 <= index < self._key_count:
                self._backoff_until[index] = max(
                    self._backoff_until[index],
                    time.monotonic() + self._backoff_seconds,
                )
        return rotate

    async def report_success(self, index: int) -> None:
        """Clear any backoff for a credential that completed a request."""
        async with self._lock:
            if 0 <= index < self._key_count:
                self._backoff_until[index] = 0.0
