"""Command Code (OpenAI-compat) adapter."""

from free_claude_code.config.provider_catalog import COMMANDCODE_DEFAULT_BASE

from .client import CommandCodeProvider

__all__ = ["COMMANDCODE_DEFAULT_BASE", "CommandCodeProvider"]
