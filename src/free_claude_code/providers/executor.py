"""Multi-provider executor with 429-driven fallback and credential rotation."""

from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.rate_limiter import (
    RotationState,
    should_disable_rate_limiting,
)


class MultiProviderExecutor:
    """Execute requests across multiple providers with 429-driven fallback.

    Coordinates:
    1. Multiple API keys per provider (credential rotation)
    2. Fallback to other providers
    3. Circuit breaker state tracking
    4. Exponential backoff on rate limits
    """

    def __init__(
        self,
        primary_provider: BaseProvider,
        primary_config: ProviderConfig,
        fallback_providers: list[tuple[BaseProvider, ProviderConfig]] | None = None,
        fallback_provider_ids: tuple[str, ...] = (),
    ):
        """Initialize multi-provider executor.

        Args:
            primary_provider: Main provider instance
            primary_config: Primary provider config with api_keys tuple
            fallback_providers: List of (provider, config) tuples for fallback
            fallback_provider_ids: Comma-separated provider IDs to try in order
        """
        self.primary_provider = primary_provider
        self.primary_config = primary_config
        self.fallback_providers = fallback_providers or []
        self.fallback_provider_ids = fallback_provider_ids
        self.rotation_state = RotationState()
        self.disable_rate_limiting = should_disable_rate_limiting(
            primary_config.api_keys, fallback_provider_ids
        )

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream response with fallback on 429."""

        providers_to_try = self._build_provider_chain()
        last_error = None

        for provider, config in providers_to_try:
            if not self._is_credential_available(config):
                continue

            try:
                trace_event(
                    stage="provider",
                    event="provider.attempt",
                    source="executor",
                    provider=config.base_url,
                    request_id=request_id,
                    attempt_index=len(providers_to_try),
                )

                # Ensure the provider gets the updated configuration and key per iteration
                provider._config = config
                if hasattr(provider, "_api_key"):
                    object.__setattr__(provider, "_api_key", config.api_key)

                async for chunk in provider.stream_response(
                    request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    reasoning=reasoning,
                ):
                    yield chunk

                # Success: clear circuit breaker
                self.rotation_state.record_success_for_credential(config.api_key)
                return

            except Exception as e:
                last_error = e
                http_status = getattr(getattr(e, "response", None), "status_code", None)

                if http_status == 429:
                    # Rate limited: mark and try next
                    logger.warning(
                        "Provider returned 429, rotating to next credential: "
                        "provider={} consecutive_failures={}",
                        config.base_url,
                        (
                            (
                                cb := self.rotation_state.circuit_breakers.get(
                                    config.api_key
                                )
                            )
                            is not None
                            and cb.consecutive_failures
                        )
                        if config.api_key in self.rotation_state.circuit_breakers
                        else 0,
                    )
                    self.rotation_state.record_429_for_credential(config.api_key)
                    trace_event(
                        stage="provider",
                        event="provider.rate_limited",
                        source="executor",
                        provider=config.base_url,
                        request_id=request_id,
                    )
                    continue

                # Other errors: attempt fallback before giving up
                logger.warning(
                    "Provider request failed, attempting fallback: "
                    "provider={} error={} http_status={}",
                    config.base_url,
                    type(e).__name__,
                    http_status,
                )
                trace_event(
                    stage="provider",
                    event="provider.error",
                    source="executor",
                    provider=config.base_url,
                    request_id=request_id,
                    exc_type=type(e).__name__,
                    http_status=http_status,
                )
                continue

        # All providers exhausted
        if last_error:
            logger.error(
                "All providers exhausted after fallback chain. Last error: {} ({})",
                type(last_error).__name__,
                getattr(getattr(last_error, "response", None), "status_code", "unknown")
                if hasattr(last_error, "response")
                else "unknown",
            )
            raise last_error

        raise ApplicationUnavailableError(
            "All providers are rate-limited or unavailable. Try again in a few moments."
        )

    def _build_provider_chain(
        self,
    ) -> list[tuple[BaseProvider, ProviderConfig]]:
        """Build ordered list of (provider, config) to try.

        1. Primary provider with rotated credentials
        2. Fallback providers in order
        """
        chain: list[tuple[BaseProvider, ProviderConfig]] = []

        # Primary provider with all credentials
        if self.primary_config.api_keys:
            for api_key in self.primary_config.api_keys:
                config = self._make_credential_config(self.primary_config, api_key)
                chain.append((self.primary_provider, config))
        else:
            chain.append((self.primary_provider, self.primary_config))

        # Fallback providers
        for provider, config in self.fallback_providers:
            if config.api_keys:
                for api_key in config.api_keys:
                    fallback_config = self._make_credential_config(config, api_key)
                    chain.append((provider, fallback_config))
            else:
                chain.append((provider, config))

        return chain

    def _make_credential_config(
        self, base_config: ProviderConfig, api_key: str
    ) -> ProviderConfig:
        """Create a config with a specific API key."""
        return ProviderConfig(
            api_key=api_key,
            api_keys=(api_key,),  # Single key for this attempt
            credential_strategy=base_config.credential_strategy,
            base_url=base_config.base_url,
            rate_limit=None if self.disable_rate_limiting else base_config.rate_limit,
            rate_window=base_config.rate_window,
            max_concurrency=base_config.max_concurrency,
            http_read_timeout=base_config.http_read_timeout,
            http_write_timeout=base_config.http_write_timeout,
            http_connect_timeout=base_config.http_connect_timeout,
            proxy=base_config.proxy,
            log_raw_sse_events=base_config.log_raw_sse_events,
            log_api_error_tracebacks=base_config.log_api_error_tracebacks,
        )

    def _is_credential_available(self, config: ProviderConfig) -> bool:
        """Check if this credential/provider is available now."""
        breaker = self.rotation_state.circuit_breakers.get(config.api_key)
        if breaker is None:
            return True
        return breaker.is_available

    async def cleanup(self) -> None:
        """Clean up all provider instances."""
        await self.primary_provider.cleanup()
        for provider, _ in self.fallback_providers:
            await provider.cleanup()

    def get_circuit_state_summary(self) -> dict[str, Any]:
        """Get debug summary of circuit breaker states."""
        return {
            breaker.credential_id: {
                "state": breaker.state.value,
                "consecutive_failures": breaker.consecutive_failures,
                "backoff_seconds": breaker.backoff_seconds,
                "is_permanently_exhausted": breaker.is_permanently_exhausted(),
            }
            for breaker in self.rotation_state.circuit_breakers.values()
        }
