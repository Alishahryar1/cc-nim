"""Hugging Face Inference Providers OpenAI-compatible adapter."""

from providers.defaults import HUGGINGFACE_DEFAULT_BASE

from .client import HuggingFaceProvider

__all__ = ["HUGGINGFACE_DEFAULT_BASE", "HuggingFaceProvider"]
