"""Generic OpenAI-compatible provider adapter."""

from free_claude_code.config.provider_catalog import OPENAI_COMPATIBLE_DEFAULT_BASE

from .client import OpenAICompatibleProvider

__all__ = ["OPENAI_COMPATIBLE_DEFAULT_BASE", "OpenAICompatibleProvider"]
