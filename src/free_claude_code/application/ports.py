"""Typed capabilities consumed by application use cases."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy

from .model_metadata import ProviderModelInfo


class ProviderPort(Protocol):
    """Minimal provider capability required to execute one request."""

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None: ...

    def stream_response(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]: ...


ProviderResolver = Callable[[str], ProviderPort]


class RequestRuntimeLease(Protocol):
    """One provider generation retained for a complete API response."""

    @property
    def generation_id(self) -> int: ...

    @property
    def settings(self) -> Settings: ...

    def is_provider_cached(self, provider_id: str) -> bool: ...

    def resolve_provider(self, provider_id: str) -> ProviderPort: ...

    async def release(self) -> None: ...


class RequestRuntimePort(Protocol):
    """Provider generation and model metadata required by application requests."""

    async def acquire(self) -> RequestRuntimeLease: ...

    def current_settings(self) -> Settings: ...

    def cached_model_supports_thinking(
        self, provider_id: str, model_id: str
    ) -> bool | None: ...

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]: ...


@dataclass(frozen=True, slots=True)
class StopResult:
    """Implementation-neutral result retaining the existing ``/stop`` variants."""

    cancelled_count: int | None = None
    source: str | None = None


class TaskController(Protocol):
    """Stop managed work without exposing messaging or CLI resources."""

    async def stop_all(self) -> StopResult | None: ...
