"""Runtime capabilities consumed by the HTTP API adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import ProviderModelInfo


class RequestRuntimeLease(Protocol):
    """One provider generation retained for a complete API response."""

    @property
    def generation_id(self) -> int: ...

    @property
    def settings(self) -> Settings: ...

    def is_provider_cached(self, provider_id: str) -> bool: ...

    def resolve_provider(self, provider_id: str) -> BaseProvider: ...

    async def release(self) -> None: ...


class RequestRuntimePort(Protocol):
    """Provider generation and model-catalog access required by API requests."""

    async def acquire(self) -> RequestRuntimeLease: ...

    def current_settings(self) -> Settings: ...

    def cached_model_ids(self) -> dict[str, frozenset[str]]: ...

    def cached_model_supports_thinking(
        self, provider_id: str, model_id: str
    ) -> bool | None: ...

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]: ...


class AdminRuntimePort(Protocol):
    """Runtime operations exposed by the local Admin API."""

    async def apply_admin_config(
        self, updates: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def admin_status(self) -> dict[str, Any]: ...

    async def test_provider(self, provider_id: str) -> dict[str, Any]: ...

    async def refresh_models(self) -> dict[str, Any]: ...

    async def request_restart(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StopResult:
    """Implementation-neutral result retaining the existing `/stop` variants."""

    cancelled_count: int | None = None
    source: str | None = None


class SessionControlPort(Protocol):
    """Stop managed work without exposing messaging or CLI resources."""

    async def stop_all(self) -> StopResult | None: ...


@dataclass(frozen=True, slots=True)
class ApiServices:
    """Complete runtime boundary required to construct the API application."""

    requests: RequestRuntimePort
    admin: AdminRuntimePort
    sessions: SessionControlPort
