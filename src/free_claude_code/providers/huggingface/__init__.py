"""Hugging Face Inference Providers OpenAI-compatible adapter."""

from free_claude_code.providers.defaults import HUGGINGFACE_DEFAULT_BASE

from .client import HuggingFaceProvider

__all__ = ["HUGGINGFACE_DEFAULT_BASE", "HuggingFaceProvider"]
