"""Global rate limiter for API requests."""

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar, TypeVar

from loguru import logger

from core.anthropic.streaming import retryable_transient_status
from core.rate_limit import StrictSlidingWindowLimiter
from core.trace import trace_event

T = TypeVar("T")

UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_MAX_RETRIES = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS - 1


def _upstream_http_retryable(code: int) -> bool:
    """True for rate limit / upstream server failures that should backoff-retry."""
    return code == 429 or 500 <= code <= 599


def retryable_upstream_status(exc: BaseException) -> int | None:
    """Return HTTP-like status codes that qualify for reactive backoff retries.

    ``429`` plus any upstream ``5xx`` use the same exponential backoff and scoped
    limiter blocking semantics as today's rate-limit path.
    """
    status = retryable_transient_status(exc)
    if status is not None and _upstream_http_retryable(status):
        return status
    return None


class GlobalRateLimiter:
    """
    Global singleton rate limiter that blocks all requests
    when a rate limit error is encountered (reactive) and
    throttles requests (proactive) using a strict rolling window.

    Optionally enforces a max_concurrency cap: at most N provider streams
    may be open simultaneously, independent of the sliding window.

    Proactive limits - throttles requests to stay within API limits.
    Reactive limits - pauses all requests when a 429 or 5xx retry backoff is active.
    Concurrency limit - caps simultaneously open streams.
    """

    _instance: ClassVar[GlobalRateLimiter | None] = None
    _scoped_instances: ClassVar[dict[str, GlobalRateLimiter]] = {}

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        # Prevent re-initialization on singleton reuse
        if hasattr(self, "_initialized"):
            return

        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        self._max_concurrency = max_concurrency
        self._proactive_limiter = StrictSlidingWindowLimiter(
            self._rate_limit, self._rate_window
        )
        self._blocked_until: float = 0
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        self._initialized = True

        logger.info(
            f"GlobalRateLimiter (Provider) initialized ({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    @classmethod
    def get_instance(
        cls,
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
    ) -> GlobalRateLimiter:
        """Get or create the singleton instance.

        Args:
            rate_limit: Requests per window (only used on first creation)
            rate_window: Window in seconds (only used on first creation)
            max_concurrency: Max simultaneous open streams (only used on first creation)
        """
        if cls._instance is None:
            cls._instance = cls(
                rate_limit=rate_limit or 40,
                rate_window=rate_window or 60.0,
                max_concurrency=max_concurrency,
            )
        return cls._instance

    @classmethod
    def get_scoped_instance(
        cls,
        scope: str,
        *,
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
    ) -> GlobalRateLimiter:
        """Get or create a provider-scoped limiter instance."""
        if not scope:
            raise ValueError("scope must be non-empty")
        desired_rate_limit = rate_limit or 40
        desired_rate_window = float(rate_window or 60.0)
        existing = cls._scoped_instances.get(scope)
        if existing and existing.matches_config(
            desired_rate_limit, desired_rate_window, max_concurrency
        ):
            return existing
        if existing:
            logger.info(
                "Rebuilding provider rate limiter for updated scope '{}'", scope
            )
        cls._scoped_instances[scope] = cls(
            rate_limit=desired_rate_limit,
            rate_window=desired_rate_window,
            max_concurrency=max_concurrency,
        )
        return cls._scoped_instances[scope]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._scoped_instances = {}

    async def wait_if_blocked(self) -> bool:
        """
        Wait if currently rate limited or throttle to meet quota.

        Returns:
            True if was reactively blocked and waited, False otherwise.
        """
        # 1. Reactive check: Wait if someone hit a reactive backoff (429/5xx retries)
        waited_reactively = False
        now = time.monotonic()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            logger.warning(
                f"Global provider rate limit active (reactive), waiting {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)
            waited_reactively = True

        # 2. Proactive check: strict rolling window (no bursts beyond N in last W seconds)
        await self._acquire_proactive_slot()
        return waited_reactively

    async def _acquire_proactive_slot(self) -> None:
        """
        Acquire a proactive slot enforcing a strict rolling window.

        Guarantees: at most `self._rate_limit` acquisitions in any interval of length
        `self._rate_window` (seconds).
        """
        await self._proactive_limiter.acquire()

    def set_blocked(self, seconds: float = 60) -> None:
        """
        Set global block for specified seconds (reactive).

        Args:
            seconds: How long to block (default 60s)
        """
        self._blocked_until = time.monotonic() + seconds
        logger.warning(f"Global provider rate limit set for {seconds:.1f}s (reactive)")

    def is_blocked(self) -> bool:
        """Check if currently reactively blocked."""
        return time.monotonic() < self._blocked_until

    def matches_config(
        self, rate_limit: int, rate_window: float, max_concurrency: int
    ) -> bool:
        """Return whether this limiter matches the requested runtime config."""
        return (
            self._rate_limit == rate_limit
            and self._rate_window == float(rate_window)
            and self._max_concurrency == max_concurrency
        )

    def remaining_wait(self) -> float:
        """Get remaining reactive wait time in seconds."""
        return max(0.0, self._blocked_until - time.monotonic())

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        """Async context manager that holds one concurrency slot for a stream.

        Blocks until a slot is available (controlled by max_concurrency).
        """
        await self._concurrency_sem.acquire()
        try:
            yield
        finally:
            self._concurrency_sem.release()

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = DEFAULT_UPSTREAM_MAX_RETRIES,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with rate limiting and retry on transient limits.

        Waits for the proactive limiter before each attempt. On ``429`` (rate limit)
        or upstream ``5xx`` server errors, applies exponential backoff with jitter
        and sets the reactive block before retrying.

        Args:
            fn: Async callable to execute.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay cap in seconds.
            jitter: Maximum random jitter in seconds added to each delay.

        Returns:
            The result of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            await self.wait_if_blocked()

            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                status = retryable_upstream_status(e)
                if status is None:
                    raise

                label = (
                    "Rate limited (429)"
                    if status == 429
                    else f"Upstream server error ({status})"
                )
                last_exc = e
                if attempt >= max_retries:
                    logger.warning(
                        "{} retry exhausted after {} retries (attempts={})",
                        label,
                        max_retries,
                        total_attempts,
                    )
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)
                attempt_no = attempt + 1
                logger.warning(
                    "{}, attempt {}/{}. Retrying in {:.1f}s...",
                    label,
                    attempt_no,
                    total_attempts,
                    delay,
                )
                trace_event(
                    stage="provider",
                    event="provider.retry.scheduled",
                    source="provider",
                    status_code=status,
                    attempt=attempt_no,
                    max_attempts=total_attempts,
                    delay_s=round(delay, 3),
                )
                self.set_blocked(delay)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
