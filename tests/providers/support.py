"""Provider test doubles with explicit limiter ownership."""

from collections.abc import Callable
from typing import Any

from free_claude_code.providers.rate_limit import ProviderRateLimiter


class PassthroughProviderRateLimiter(ProviderRateLimiter):
    """Skip retry timing while retaining the real concurrency context manager."""

    def __init__(self) -> None:
        super().__init__(
            rate_limit=1_000_000,
            rate_window=1.0,
            max_concurrency=1_000,
        )

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await fn(*args, **kwargs)


def passthrough_rate_limiter() -> ProviderRateLimiter:
    """Return a fresh limiter test double for one provider instance."""
    return PassthroughProviderRateLimiter()


def retrying_rate_limiter() -> ProviderRateLimiter:
    """Return a fresh real limiter for provider retry-policy tests."""
    return ProviderRateLimiter(
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
    )
