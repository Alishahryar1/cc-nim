"""OpenCode Zen provider exports."""

from free_claude_code.providers.defaults import (
    OPENCODE_DEFAULT_BASE,
    OPENCODE_GO_DEFAULT_BASE,
)

from .client import OpenCodeProvider

__all__ = [
    "OPENCODE_DEFAULT_BASE",
    "OPENCODE_GO_DEFAULT_BASE",
    "OpenCodeProvider",
]
