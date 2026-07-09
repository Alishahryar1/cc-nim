"""Ollama provider package."""

from free_claude_code.providers.defaults import OLLAMA_DEFAULT_BASE

from .client import OllamaProvider

__all__ = ["OLLAMA_DEFAULT_BASE", "OllamaProvider"]
