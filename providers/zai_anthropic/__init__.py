"""Z.ai Anthropic provider exports (native Anthropic Messages endpoint for GLM 5.2)."""

from providers.defaults import ZAI_ANTHROPIC_DEFAULT_BASE

from .client import ZaiAnthropicProvider

__all__ = [
    "ZAI_ANTHROPIC_DEFAULT_BASE",
    "ZaiAnthropicProvider",
]
