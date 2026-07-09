"""Cohere Compatibility API OpenAI-compatible adapter."""

from free_claude_code.providers.defaults import COHERE_DEFAULT_BASE

from .client import CohereProvider

__all__ = ["COHERE_DEFAULT_BASE", "CohereProvider"]
