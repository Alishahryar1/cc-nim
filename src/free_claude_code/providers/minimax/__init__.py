"""MiniMax provider exports."""

from free_claude_code.providers.defaults import MINIMAX_DEFAULT_BASE

from .client import MiniMaxProvider

__all__ = [
    "MINIMAX_DEFAULT_BASE",
    "MiniMaxProvider",
]
