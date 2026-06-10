"""MiniMax provider using native Anthropic-compatible Messages."""

from providers.defaults import MINIMAX_DEFAULT_BASE

from .client import MinimaxProvider

__all__ = ["MINIMAX_DEFAULT_BASE", "MinimaxProvider"]
