"""Mistral La Plateforme provider exports."""

from free_claude_code.providers.defaults import MISTRAL_DEFAULT_BASE

from .client import MistralProvider

__all__ = ["MISTRAL_DEFAULT_BASE", "MistralProvider"]
