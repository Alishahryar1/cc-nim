"""Generic Anthropic-compatible provider adapter."""

from free_claude_code.config.provider_catalog import ANTHROPIC_COMPATIBLE_DEFAULT_BASE

from .client import AnthropicCompatibleProvider

__all__ = ["ANTHROPIC_COMPATIBLE_DEFAULT_BASE", "AnthropicCompatibleProvider"]
