"""Groq Cloud (OpenAI-compat) adapter."""

from free_claude_code.providers.defaults import GROQ_DEFAULT_BASE

from .client import GroqProvider

__all__ = ["GROQ_DEFAULT_BASE", "GroqProvider"]
