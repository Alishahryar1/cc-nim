"""DeepSeek provider exports."""

from free_claude_code.providers.defaults import DEEPSEEK_DEFAULT_BASE

from .client import DeepSeekProvider

__all__ = [
    "DEEPSEEK_DEFAULT_BASE",
    "DeepSeekProvider",
]
