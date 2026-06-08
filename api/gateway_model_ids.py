"""Gateway-safe model id encoding for Claude Code model discovery."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"


@dataclass(frozen=True, slots=True)
class DecodedGatewayModelId:
    provider_id: str
    provider_model: str
    force_thinking_enabled: bool | None = None


@lru_cache(maxsize=256)
def gateway_model_id(provider_model_ref: str) -> str:
    """Cached normal Claude model id (⚡ Bolt Optimization: 71-75)."""
    return f"{GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


@lru_cache(maxsize=256)
def no_thinking_gateway_model_id(provider_model_ref: str) -> str:
    """Cached no-thinking Claude model id (⚡ Bolt Optimization)."""
    return f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


@lru_cache(maxsize=512)
def decode_gateway_model_id(model_name: str) -> DecodedGatewayModelId | None:
    """Cached gateway model id decoder (⚡ Bolt Optimization)."""
    prefix, separator, remainder = model_name.partition("/")
    if not separator:
        return None

    force_thinking_enabled: bool | None
    if prefix == GATEWAY_MODEL_ID_PREFIX:
        force_thinking_enabled = None
    elif prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        force_thinking_enabled = False
    else:
        return None

    provider_id, provider_separator, provider_model = remainder.partition("/")
    if not provider_separator or not provider_model:
        return None

    return DecodedGatewayModelId(
        provider_id=provider_id,
        provider_model=provider_model,
        force_thinking_enabled=force_thinking_enabled,
    )
