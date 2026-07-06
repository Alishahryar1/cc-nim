"""GitHub Models provider."""

from providers.defaults import GITHUB_MODELS_DEFAULT_BASE

from .client import GitHubModelsProvider

__all__ = ["GITHUB_MODELS_DEFAULT_BASE", "GitHubModelsProvider"]
