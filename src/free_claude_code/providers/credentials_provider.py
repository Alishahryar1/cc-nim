"""Rotating credential provider with sequential/round-robin strategies, per-key rate tracking, and backoff.

This provider supports a sequential fallback mode (preferred key order: try key1, then key2, ...)
and a round_robin mode. It also enforces a simple per-key rate window (requests per window)
so we can proactively avoid hitting provider RPM limits and fall back when a key is exhausted.

The implementation is synchronous and uses threading.Lock for concurrency safety. Async
callers should wrap calls with asyncio.to_thread when necessary.
"""

import time
import random
import threading
from typing import Sequence, Optional, Dict


class RotatingCredentialProvider:
    """Provide API keys in a fallback-aware way and allow marking keys as failed.

    Key selection strategies:
    - "sequential": always prefer the lowest-index key that is enabled and has quota.
      (This implements the 1 -> 2 -> 3 -> 4 behavior.)
    - "round_robin": rotate among enabled keys.
    - "random": pick a random enabled key.

    Rate limiting:
    - If rate_limit is provided (requests per rate_window seconds), the provider tracks
      requests per-key inside the current window. When a key exhausts its quota it is
      disabled until the window end (or until an explicit Retry-After is applied).

    Failure handling:
    - mark_failure(key, retry_after_seconds) disables the key for that duration. If
      retry_after_seconds is None an exponential backoff is used per-key.
    - mark_success(key) clears failure state for the key.

    Note: selecting a key reserves one request slot (atomic under the provider lock).
    """

    def __init__(
        self,
        keys: Sequence[str],
        *,
        strategy: str = "sequential",
        rate_limit: Optional[int] = None,
        rate_window: int = 60,
        max_backoff: int = 3600,
        jitter_fraction: float = 0.1,
    ) -> None:
        self._keys = [k for k in keys if k]
        self._n = len(self._keys)
        self._strategy = strategy
        # pointer used for round_robin selection
        self._idx = 0
        self._lock = threading.Lock()

        # disabled_until: key -> unix timestamp until which the key is disabled
        self._disabled_until: Dict[str, float] = {}
        # failure counts to grow backoff per-key
        self._fail_counts: Dict[str, int] = {}

        # simple per-key request counters for rate limiting
        # map: key -> (count:int, window_start: float)
        self._counters: Dict[str, tuple[int, float]] = {}
        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)

        # backoff parameters
        self._max_backoff = int(max_backoff)
        self._jitter_fraction = float(jitter_fraction)

    def _now(self) -> float:
        return time.time()

    def _is_enabled(self, key: str) -> bool:
        until = self._disabled_until.get(key)
        return not until or self._now() >= until

    def _has_quota_and_reserve(self, key: str) -> bool:
        """Check and reserve a request token for the given key.

        Returns True and increments the counter if quota is available. If quota is
        exhausted, the key is disabled until the window end and False is returned.
        """
        if self._rate_limit is None:
            return True
        now = self._now()
        count, window_start = self._counters.get(key, (0, now))
        # reset window if expired
        if now - window_start >= self._rate_window:
            count = 0
            window_start = now
        if count < self._rate_limit:
            # reserve one slot
            count += 1
            self._counters[key] = (count, window_start)
            return True
        # no quota -> disable until end of window
        disable_until = window_start + self._rate_window
        self._disabled_until[key] = disable_until
        return False

    def next_key(self) -> Optional[str]:
        """Return the next enabled key according to the configured strategy.

        This method atomically selects and reserves one request slot for the returned
        key. If no keys are available it returns None.
        """
        with self._lock:
            if not self._keys:
                return None

            now = self._now()

            # helper: yield candidate keys in the order for the selected strategy
            def candidates():
                if self._strategy == "sequential":
                    yield from self._keys
                    return
                if self._strategy == "round_robin":
                    n = self._n
                    for i in range(n):
                        yield self._keys[(self._idx + i) % n]
                    # advance base index for next call
                    self._idx = (self._idx + 1) % n
                    return
                # random
                random.shuffle(self._keys)
                yield from self._keys

            for key in candidates():
                # skip disabled keys
                if not self._is_enabled(key):
                    continue
                # check quota and reserve
                if self._has_quota_and_reserve(key):
                    return key
                # if _has_quota_and_reserve returned False it has set disabled_until
                # so next iteration will skip it.
            return None

    def mark_failure(self, key: str, retry_after_seconds: Optional[int] = None) -> None:
        """Mark a key as failed. If retry_after_seconds is None an exponential backoff is used."""
        with self._lock:
            count = self._fail_counts.get(key, 0) + 1
            self._fail_counts[key] = count
            if retry_after_seconds is None:
                base = min(self._max_backoff, 2 ** min(count, 10))
                jitter = random.uniform(0, base * self._jitter_fraction)
                retry_after_seconds = int(base + jitter)
            self._disabled_until[key] = self._now() + float(max(1, int(retry_after_seconds)))

    def mark_success(self, key: str) -> None:
        with self._lock:
            self._fail_counts.pop(key, None)
            self._disabled_until.pop(key, None)
            # Do not reset counters here; quota usage is real and should remain counted

    def available_keys(self) -> list[str]:
        """Return a snapshot list of keys that appear usable now (not guaranteed until reservation)."""
        with self._lock:
            now = self._now()
            good = [k for k in self._keys if self._is_enabled(k) and (self._rate_limit is None or self._counters.get(k, (0, now))[0] < self._rate_limit)]
            return good

    def stats(self) -> dict[str, dict[str, float]]:
        """Return diagnostic stats for each key (for telemetry/debugging)."""
        with self._lock:
            now = self._now()
            out: Dict[str, dict[str, float]] = {}
            for k in self._keys:
                count, start = self._counters.get(k, (0, now))
                disabled_until = self._disabled_until.get(k)
                out[k] = {
                    "count": float(count),
                    "window_start": float(start),
                    "disabled_until": float(disabled_until) if disabled_until else 0.0,
                }
            return out
