"""Ollama Cloud provider package."""

from providers.defaults import OLLAMA_CLOUD_DEFAULT_BASE

from .client import OllamaCloudProvider

__all__ = ["OLLAMA_CLOUD_DEFAULT_BASE", "OllamaCloudProvider"]
