"""MiniMax provider exports."""

from providers.defaults import MINIMAX_DEFAULT_BASE

from .client import MiniMaxProvider

__all__ = [
    "MINIMAX_DEFAULT_BASE",
    "MiniMaxProvider",
]
