"""Provider test doubles with explicit limiter ownership."""

from collections.abc import Callable
from typing import Any

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
    create_openai_chat_provider,
)
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
        kwargs.pop("provider_failure_override", None)
        return await fn(*args, **kwargs)


def passthrough_rate_limiter() -> ProviderRateLimiter:
    """Return a fresh limiter test double for one provider instance."""
    return PassthroughProviderRateLimiter()


def profiled_provider(
    provider_id: str,
    config: ProviderConfig,
    *,
    rate_limiter: ProviderRateLimiter | None = None,
) -> OpenAIChatProvider:
    """Construct one declarative provider for a focused behavior test."""
    return create_openai_chat_provider(
        provider_id,
        config,
        rate_limiter or passthrough_rate_limiter(),
    )


def retrying_rate_limiter() -> ProviderRateLimiter:
    """Return a fresh real limiter for provider retry-policy tests."""
    return ProviderRateLimiter(
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
    )
