"""Z.ai provider exports."""

from free_claude_code.providers.defaults import ZAI_DEFAULT_BASE

from .client import ZaiProvider

__all__ = [
    "ZAI_DEFAULT_BASE",
    "ZaiProvider",
]
