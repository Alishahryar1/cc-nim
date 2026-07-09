"""Kimi (Moonshot) provider exports."""

from free_claude_code.providers.defaults import KIMI_DEFAULT_BASE

from .client import KimiProvider

__all__ = [
    "KIMI_DEFAULT_BASE",
    "KimiProvider",
]
