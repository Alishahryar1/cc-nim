from datetime import UTC, datetime, timedelta

from free_claude_code.application.capabilities import (
    CapabilityRegistry,
    ModelCapabilities,
)


def test_registry_set_and_get_fresh_snapshot() -> None:
    registry = CapabilityRegistry(ttl_seconds=60)
    capability = ModelCapabilities(
        provider_id="gemini",
        model_id="gemini-2.5-pro",
        max_context_tokens=1_000_000,
        supports_tools=True,
    )

    stored = registry.set(capability)

    assert stored.detected is True
    assert stored.detected_at is not None
    assert registry.get("gemini", "gemini-2.5-pro") == stored


def test_registry_returns_none_for_stale_snapshot() -> None:
    registry = CapabilityRegistry(ttl_seconds=60)
    stale_time = datetime.now(UTC) - timedelta(seconds=120)
    registry.update(
        ModelCapabilities(
            provider_id="open_router",
            model_id="openrouter/free",
            max_context_tokens=256_000,
            detected=True,
            detected_at=stale_time,
        )
    )

    assert registry.get("open_router", "openrouter/free") is None
    assert registry.peek("open_router", "openrouter/free") is not None


def test_merge_prefers_new_values_without_dropping_existing_ones() -> None:
    registry = CapabilityRegistry(ttl_seconds=60)
    registry.set(
        ModelCapabilities(
            provider_id="siliconflow",
            model_id="siliconflow/Qwen/Qwen3-Coder-480B-A35B-Instruct",
            max_context_tokens=256_000,
            supports_tools=True,
            supports_streaming=True,
        )
    )

    merged = registry.update(
        ModelCapabilities(
            provider_id="siliconflow",
            model_id="siliconflow/Qwen/Qwen3-Coder-480B-A35B-Instruct",
            max_output_tokens=32_000,
            supports_vision=False,
        )
    )

    assert merged.max_context_tokens == 256_000
    assert merged.max_output_tokens == 32_000
    assert merged.supports_tools is True
    assert merged.supports_vision is False
    assert merged.supports_streaming is True
