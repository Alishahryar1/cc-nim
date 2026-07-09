"""Vercel AI Gateway OpenAI-compatible adapter."""

from free_claude_code.providers.defaults import VERCEL_AI_GATEWAY_DEFAULT_BASE

from .client import VercelProvider

__all__ = ["VERCEL_AI_GATEWAY_DEFAULT_BASE", "VercelProvider"]
