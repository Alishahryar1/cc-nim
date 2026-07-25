"""Rate limit and fallback handling for multi-key/multi-provider strategy.

When multiple API keys or fallback providers are configured, the proxy
disables artificial rate limiting and instead uses 429 responses as signals
to rotate to the next credential/provider.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker state for credentials."""

    CLOSED = "closed"  # Ready to use
    OPEN = "open"  # Recently returned 429, waiting before retry
    HALF_OPEN = "half_open"  # Testing if credential is back


@dataclass
class CredentialCircuitBreaker:
    """Per-credential circuit breaker to track exhaustion and backoff."""

    credential_id: str  # api_key or provider_id
    state: CircuitState = CircuitState.CLOSED
    last_429_at: float = 0.0
    backoff_seconds: float = 5.0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5

    @property
    def is_available(self) -> bool:
        """Check if credential is available now."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            # Allow one attempt to test recovery
            return True
        # OPEN: check if backoff expired
        elapsed = time.time() - self.last_429_at
        if elapsed >= self.backoff_seconds:
            self.state = CircuitState.HALF_OPEN
            return True
        return False

    def record_429(self) -> None:
        """Record a rate limit error."""
        self.state = CircuitState.OPEN
        self.last_429_at = time.time()
        self.consecutive_failures += 1
        # Exponential backoff: 5s, 10s, 20s, 40s, 80s
        self.backoff_seconds = min(5.0 * (2 ** (self.consecutive_failures - 1)), 300.0)

    def record_success(self) -> None:
        """Record a successful request."""
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.backoff_seconds = 5.0

    def is_permanently_exhausted(self) -> bool:
        """Check if credential has exceeded max retries."""
        return self.consecutive_failures >= self.max_consecutive_failures


@dataclass
class RotationState:
    """Mutable state for credential/provider rotation."""

    current_index: int = 0
    circuit_breakers: dict[str, CredentialCircuitBreaker] = field(default_factory=dict)
    last_rotation_at: float = 0.0

    def should_rotate_on_429(
        self, credential_id: str, credentials: tuple[str, ...]
    ) -> bool:
        """Check if we should try the next credential after 429."""
        if not credentials or len(credentials) <= 1:
            # Single credential: no rotation possible
            return False
        # Only rotate if this credential is exhausted
        breaker = self.circuit_breakers.get(
            credential_id, CredentialCircuitBreaker(credential_id)
        )
        return breaker.consecutive_failures > 0

    def get_next_available_credential(
        self, credentials: tuple[str, ...], strategy: str
    ) -> str | None:
        """Find the next available credential based on strategy.

        Returns None if all credentials are permanently exhausted.
        """
        if not credentials:
            return None

        available = [
            cred
            for cred in credentials
            if not self.circuit_breakers.get(
                cred, CredentialCircuitBreaker(cred)
            ).is_permanently_exhausted()
        ]

        if not available:
            return None

        if strategy == "round_robin":
            # Rotate to next available
            for offset in range(1, len(credentials)):
                idx = (self.current_index + offset) % len(credentials)
                cred = credentials[idx]
                if cred in available:
                    self.current_index = idx
                    self.last_rotation_at = time.time()
                    return cred
        elif strategy == "random":
            import random

            return random.choice(available)
        else:  # sequential (default)
            # Prefer first available in order
            for cred in available:
                return cred

        return available[0] if available else None

    def record_429_for_credential(self, credential_id: str) -> None:
        """Mark credential as rate-limited."""
        if credential_id not in self.circuit_breakers:
            self.circuit_breakers[credential_id] = CredentialCircuitBreaker(
                credential_id
            )
        self.circuit_breakers[credential_id].record_429()

    def record_success_for_credential(self, credential_id: str) -> None:
        """Clear rate-limit flag for credential."""
        if credential_id not in self.circuit_breakers:
            self.circuit_breakers[credential_id] = CredentialCircuitBreaker(
                credential_id
            )
        self.circuit_breakers[credential_id].record_success()


def should_disable_rate_limiting(
    api_keys: tuple[str, ...], fallback_providers: tuple[str, ...]
) -> bool:
    """Determine if artificial rate limiting should be disabled.

    When multi-key or multi-provider fallback is configured, disable
    artificial rate limiting and let 429 responses drive rotation.
    """
    has_multiple_keys = len(api_keys) > 1
    has_fallback = len(fallback_providers) > 1
    return has_multiple_keys or has_fallback
