"""Vercel AI Gateway OpenAI-compatible adapter."""

from providers.defaults import VERCEL_AI_GATEWAY_DEFAULT_BASE

from .client import VercelProvider

__all__ = ["VERCEL_AI_GATEWAY_DEFAULT_BASE", "VercelProvider"]
