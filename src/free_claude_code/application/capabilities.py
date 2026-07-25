"""Model capability metadata and a small in-memory registry."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock

from free_claude_code.application.model_metadata import ProviderModelInfo


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capability snapshot for one provider/model pair."""

    provider_id: str
    model_id: str
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    supports_streaming: bool | None = None
    supports_json_mode: bool | None = None
    detected: bool = False
    detected_at: datetime | None = None
    source: str = "unknown"

    def with_detection_time(
        self, detected_at: datetime | None = None
    ) -> ModelCapabilities:
        """Return a copy stamped with a detection time when one is not set."""

        return replace(
            self,
            detected_at=_normalize_datetime(
                detected_at or self.detected_at or datetime.now(UTC)
            ),
            detected=True,
        )

    def merge(self, other: ModelCapabilities) -> ModelCapabilities:
        """Merge two snapshots, keeping explicit values from ``other`` first."""

        if self.provider_id != other.provider_id or self.model_id != other.model_id:
            raise ValueError(
                "Capability snapshots can only be merged for the same model."
            )
        return ModelCapabilities(
            provider_id=self.provider_id,
            model_id=self.model_id,
            max_context_tokens=other.max_context_tokens
            if other.max_context_tokens is not None
            else self.max_context_tokens,
            max_output_tokens=other.max_output_tokens
            if other.max_output_tokens is not None
            else self.max_output_tokens,
            supports_tools=_prefer_bool(other.supports_tools, self.supports_tools),
            supports_vision=_prefer_bool(other.supports_vision, self.supports_vision),
            supports_reasoning=_prefer_bool(
                other.supports_reasoning, self.supports_reasoning
            ),
            supports_streaming=_prefer_bool(
                other.supports_streaming, self.supports_streaming
            ),
            supports_json_mode=_prefer_bool(
                other.supports_json_mode, self.supports_json_mode
            ),
            detected=other.detected or self.detected,
            detected_at=_normalize_datetime(other.detected_at or self.detected_at),
            source=other.source if other.source != "unknown" else self.source,
        )


class CapabilityRegistry:
    """Thread-safe in-memory cache of model capabilities."""

    def __init__(self, *, ttl_seconds: float = 24 * 60 * 60) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._ttl_seconds = float(ttl_seconds)
        self._lock = RLock()
        self._capabilities: dict[tuple[str, str], ModelCapabilities] = {}

    def get(
        self,
        provider_id: str,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> ModelCapabilities | None:
        """Return a fresh cached snapshot, or ``None`` if absent or stale."""

        with self._lock:
            capability = self._capabilities.get((provider_id, model_id))
            if capability is None:
                return None
            if self._is_stale(capability, now=now):
                return None
            return capability

    def peek(self, provider_id: str, model_id: str) -> ModelCapabilities | None:
        """Return the cached snapshot even when it is stale."""

        with self._lock:
            return self._capabilities.get((provider_id, model_id))

    def set(self, capability: ModelCapabilities) -> ModelCapabilities:
        """Store a capability snapshot and return the normalized copy."""

        normalized = capability.with_detection_time()
        with self._lock:
            self._capabilities[(normalized.provider_id, normalized.model_id)] = (
                normalized
            )
        return normalized

    def update(self, capability: ModelCapabilities) -> ModelCapabilities:
        """Merge a new snapshot into the cache and return the stored result."""

        with self._lock:
            existing = self._capabilities.get(
                (capability.provider_id, capability.model_id)
            )
            merged = capability if existing is None else existing.merge(capability)
            normalized = merged.with_detection_time(capability.detected_at)
            self._capabilities[(normalized.provider_id, normalized.model_id)] = (
                normalized
            )
            return normalized

    def register_model_info(
        self,
        provider_id: str,
        model_info: ProviderModelInfo,
        *,
        source: str = "model-list",
    ) -> ModelCapabilities:
        """Store one provider model-list entry as a capability snapshot."""

        capability = ModelCapabilities(
            provider_id=provider_id,
            model_id=model_info.model_id,
            max_context_tokens=model_info.max_context_tokens,
            max_output_tokens=model_info.max_output_tokens,
            supports_tools=model_info.supports_tools,
            supports_vision=model_info.supports_vision,
            supports_reasoning=_prefer_bool(
                model_info.supports_reasoning, model_info.supports_thinking
            ),
            supports_streaming=model_info.supports_streaming,
            supports_json_mode=model_info.supports_json_mode,
            detected=True,
            source=source,
        )
        return self.update(capability)

    def register_model_infos(
        self,
        provider_id: str,
        model_infos: tuple[ProviderModelInfo, ...] | list[ProviderModelInfo],
        *,
        source: str = "model-list",
    ) -> tuple[ModelCapabilities, ...]:
        """Store many provider model-list entries and return the normalized snapshots."""

        return tuple(
            self.register_model_info(provider_id, model_info, source=source)
            for model_info in model_infos
        )

    def invalidate(self, provider_id: str, model_id: str) -> None:
        """Remove one cached capability snapshot."""

        with self._lock:
            self._capabilities.pop((provider_id, model_id), None)

    def clear(self) -> None:
        """Drop every cached snapshot."""

        with self._lock:
            self._capabilities.clear()

    def items(self) -> tuple[ModelCapabilities, ...]:
        """Return a stable snapshot of all cached entries."""

        with self._lock:
            return tuple(self._capabilities.values())

    def _is_stale(
        self, capability: ModelCapabilities, *, now: datetime | None = None
    ) -> bool:
        if capability.detected_at is None:
            return False
        current = _normalize_datetime(now or datetime.now(UTC))
        detected_at = _normalize_datetime(capability.detected_at)
        if current is None or detected_at is None:
            return False
        return (current - detected_at).total_seconds() > self._ttl_seconds


capability_registry = CapabilityRegistry()


def _prefer_bool(preferred: bool | None, fallback: bool | None) -> bool | None:
    if preferred is not None:
        return preferred
    return fallback


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
