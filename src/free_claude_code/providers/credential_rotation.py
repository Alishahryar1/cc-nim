"""Multi-credential rotation engine shared by rotating provider wrappers.

Health model (aligned with the multi-provider proxy's key pool):
  - HEALTHY: serving requests.
  - COOLDOWN: briefly benched after an error (tiered 10s -> 30s -> 60s -> 120s).
  - CIRCUIT_OPEN: 3+ consecutive failures; benched until cooldown elapses.
  - HALF_OPEN: recovering; a single probe request is allowed through.
  - LOCKED_OUT: auth failure (401/403); escalating lockout 5min -> 1h -> 24h,
    then a half-open probe before full reuse.

Policies:
  - ``single``: always the first key.
  - ``round_robin``: spread requests across healthy keys in turn.
  - ``least_used``: healthy key with the fewest requests goes first.
  - ``failover`` (alias ``on_error``): stick to the first healthy key until it
    fails, then move to the next.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import openai

from free_claude_code.providers.failure_policy import (
    retryable_transient_status,
    retryable_upstream_transport_error,
)

ROTATION_POLICIES = frozenset({"single", "round_robin", "least_used", "failover", "on_error"})

COOLDOWN_TIERS_SECONDS = (10.0, 30.0, 60.0, 120.0)
AUTH_LOCKOUT_TIERS_SECONDS = (300.0, 3600.0, 86400.0)
CIRCUIT_OPEN_THRESHOLD = 3

STATE_HEALTHY = "HEALTHY"
STATE_COOLDOWN = "COOLDOWN"
STATE_CIRCUIT_OPEN = "CIRCUIT_OPEN"
STATE_HALF_OPEN = "HALF_OPEN"
STATE_LOCKED_OUT = "LOCKED_OUT"


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


def _status_from_error(error: BaseException) -> int | None:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    if isinstance(error, openai.APIStatusError):
        return error.status_code
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


@dataclass
class KeyHealth:
    """Health and usage metrics for one credential."""

    state: str = STATE_HEALTHY
    request_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    tier: int = 0
    cooldown_until: float = 0.0
    lockout_until: float = 0.0
    last_used: float = 0.0
    is_probing: bool = False

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "state": self.state,
            "request_count": self.request_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "tier": self.tier,
            "cooldown_remaining": max(0.0, self.cooldown_until - now),
            "lockout_remaining": max(0.0, self.lockout_until - now),
            "is_probing": self.is_probing,
        }


class CredentialRotationState:
    """Pick which credential serves each request under a rotation policy."""

    def __init__(self, key_count: int, policy: str = "single") -> None:
        if key_count <= 0:
            raise ValueError("key_count must be > 0")
        if policy == "on_error":
            policy = "failover"
        self._key_count = key_count
        self._policy = policy if policy in ROTATION_POLICIES else "single"
        self._health = [KeyHealth() for _ in range(key_count)]
        self._rr_index = 0
        self._lock = asyncio.Lock()

    @property
    def policy(self) -> str:
        return self._policy

    def _update_states(self, now: float) -> None:
        for health in self._health:
            if health.state == STATE_LOCKED_OUT:
                if health.lockout_until > 0 and now >= health.lockout_until:
                    # After lockout, require a successful probe before full use.
                    health.state = STATE_HALF_OPEN
                    health.is_probing = False
            elif health.state in (STATE_COOLDOWN, STATE_CIRCUIT_OPEN):
                if health.cooldown_until > 0 and now >= health.cooldown_until:
                    health.state = STATE_HALF_OPEN
                    health.is_probing = False

    @staticmethod
    def _selectable(health: KeyHealth) -> bool:
        return health.state == STATE_HEALTHY or (
            health.state == STATE_HALF_OPEN and not health.is_probing
        )

    async def acquire(self) -> int:
        """Return the index of the credential to use for one new request."""
        async with self._lock:
            now = time.monotonic()
            self._update_states(now)

            if self._policy == "single" or self._key_count == 1:
                return 0

            selected: int | None = None
            if self._policy == "round_robin":
                for i in range(self._key_count):
                    index = (self._rr_index + i) % self._key_count
                    if self._selectable(self._health[index]):
                        selected = index
                        self._rr_index = (index + 1) % self._key_count
                        break
            elif self._policy == "least_used":
                candidates = [
                    (h.request_count, h.last_used, i)
                    for i, h in enumerate(self._health)
                    if self._selectable(h)
                ]
                if candidates:
                    selected = min(candidates)[2]
            else:  # failover
                for i, health in enumerate(self._health):
                    if self._selectable(health):
                        selected = i
                        break

            if selected is None:
                return -1

            health = self._health[selected]
            health.request_count += 1
            health.last_used = now
            if health.state == STATE_HALF_OPEN:
                health.is_probing = True
            return selected

    async def report_success(self, index: int) -> None:
        """Mark a credential as healthy after a successful request."""
        async with self._lock:
            if 0 <= index < self._key_count:
                health = self._health[index]
                health.state = STATE_HEALTHY
                health.consecutive_failures = 0
                health.tier = 0
                health.cooldown_until = 0.0
                health.lockout_until = 0.0
                health.is_probing = False

    async def report_failure(self, index: int, error: BaseException) -> bool:
        """Record a failure for one credential; return whether to rotate.

        The return value tells the caller whether trying the next credential
        could resolve this request (auth/rate-limit/5xx/transport errors),
        as opposed to a plain 400 that would fail identically on every key.
        """
        rotate = error_justifies_rotation(error)
        status = _status_from_error(error)

        async with self._lock:
            if not (0 <= index < self._key_count):
                return rotate
            health = self._health[index]
            now = time.monotonic()
            health.failure_count += 1
            health.is_probing = False

            if status in (401, 403):
                health.consecutive_failures += 1
                tier_index = min(health.consecutive_failures, len(AUTH_LOCKOUT_TIERS_SECONDS)) - 1
                health.state = STATE_LOCKED_OUT
                health.lockout_until = now + AUTH_LOCKOUT_TIERS_SECONDS[tier_index]
            elif status == 429:
                health.tier = min(health.tier + 1, len(COOLDOWN_TIERS_SECONDS))
                health.cooldown_until = now + COOLDOWN_TIERS_SECONDS[health.tier - 1]
                health.state = STATE_COOLDOWN
            else:
                health.consecutive_failures += 1
                health.tier = min(health.tier + 1, len(COOLDOWN_TIERS_SECONDS))
                health.cooldown_until = now + COOLDOWN_TIERS_SECONDS[health.tier - 1]
                health.state = (
                    STATE_CIRCUIT_OPEN
                    if health.consecutive_failures >= CIRCUIT_OPEN_THRESHOLD
                    else STATE_COOLDOWN
                )
        return rotate

    async def report_rate_limit(self, index: int) -> None:
        """Bump the escalation tier without changing health state."""
        async with self._lock:
            if 0 <= index < self._key_count:
                health = self._health[index]
                health.tier = min(health.tier + 1, len(COOLDOWN_TIERS_SECONDS))

    async def reset_key(self, index: int) -> bool:
        """Manually restore one credential to HEALTHY."""
        async with self._lock:
            if not (0 <= index < self._key_count):
                return False
            health = self._health[index]
            health.state = STATE_HEALTHY
            health.consecutive_failures = 0
            health.tier = 0
            health.cooldown_until = 0.0
            health.lockout_until = 0.0
            health.is_probing = False
            return True

    async def reset_all(self) -> int:
        """Restore every non-healthy credential to HEALTHY."""
        count = 0
        async with self._lock:
            for health in self._health:
                if health.state != STATE_HEALTHY:
                    health.state = STATE_HEALTHY
                    health.consecutive_failures = 0
                    health.tier = 0
                    health.cooldown_until = 0.0
                    health.lockout_until = 0.0
                    health.is_probing = False
                    count += 1
        return count

    async def shortest_cooldown_remaining(self) -> float:
        """Seconds until the soonest non-healthy credential may serve again."""
        async with self._lock:
            now = time.monotonic()
            self._update_states(now)
            remaining = [
                max(h.cooldown_until, h.lockout_until) - now
                for h in self._health
                if h.state != STATE_HEALTHY
            ]
            positives = [value for value in remaining if value > 0]
            return min(positives) if positives else 0.0

    def get_metrics(self) -> list[dict[str, Any]]:
        """Return per-credential health snapshots for dashboards."""
        return [health.snapshot() for health in self._health]
