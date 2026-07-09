"""Google AI Studio Gemini (OpenAI-compat) adapter."""

from free_claude_code.providers.defaults import GEMINI_DEFAULT_BASE

from .client import GeminiProvider

__all__ = ["GEMINI_DEFAULT_BASE", "GeminiProvider"]
