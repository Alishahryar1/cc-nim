"""Application-owned model metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderModelInfo:
    """Provider model metadata used to shape the application model catalog."""

    model_id: str
    supports_thinking: bool | None = None


@dataclass(frozen=True, slots=True)
class ProviderModelRefreshResult:
    """Per-provider outcome of one model-catalog refresh."""

    refreshed_provider_ids: tuple[str, ...] = ()
    failed_provider_ids: tuple[str, ...] = ()
