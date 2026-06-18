"""Provider API-key rotation helpers."""

from __future__ import annotations

import threading

import httpx
import openai

from config.api_key_rotation import ApiKeyRotationMode


class ApiKeyRotationPool:
    """Thread-safe API-key selector for a single provider instance."""

    def __init__(self, raw_value: str, mode: ApiKeyRotationMode | str) -> None:
        self._keys = tuple(
            part.strip() for part in raw_value.split(",") if part.strip()
        )
        if not self._keys:
            raise ValueError("ApiKeyRotationPool requires at least one non-empty key")
        self._mode = ApiKeyRotationMode(mode)
        self._next_index = 0
        self._lock = threading.Lock()

    @property
    def mode(self) -> ApiKeyRotationMode:
        return self._mode

    @property
    def size(self) -> int:
        return len(self._keys)

    def key_for_new_request(self) -> str:
        """Return the key to use when opening a new upstream request."""
        if self._mode == ApiKeyRotationMode.FAILOVER_ON_LIMIT:
            return self._keys[0]
        with self._lock:
            key = self._keys[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._keys)
            return key

    def next_key_after_limit(self, current_key: str) -> str | None:
        """Return a failover key after ``current_key`` hits a rate limit."""
        try:
            index = self._keys.index(current_key)
        except ValueError:
            return self._keys[0]
        next_index = index + 1
        if next_index >= len(self._keys):
            return None
        return self._keys[next_index]


def is_limit_error(error: BaseException) -> bool:
    """Return whether an upstream exception represents an API-key rate limit."""
    if isinstance(error, openai.RateLimitError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429
    return False
