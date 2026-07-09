"""Mistral Codestral provider (codestral.mistral.ai) exports."""

from free_claude_code.providers.defaults import CODESTRAL_DEFAULT_BASE

from .client import CodestralProvider

__all__ = ["CODESTRAL_DEFAULT_BASE", "CodestralProvider"]
