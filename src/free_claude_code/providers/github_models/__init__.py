"""GitHub Models provider."""

from free_claude_code.providers.defaults import GITHUB_MODELS_DEFAULT_BASE

from .client import GitHubModelsProvider

__all__ = ["GITHUB_MODELS_DEFAULT_BASE", "GitHubModelsProvider"]
