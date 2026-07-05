"""Cohere Compatibility API OpenAI-compatible adapter."""

from providers.defaults import COHERE_DEFAULT_BASE

from .client import CohereProvider

__all__ = ["COHERE_DEFAULT_BASE", "CohereProvider"]
