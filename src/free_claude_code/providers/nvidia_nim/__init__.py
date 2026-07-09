"""NVIDIA NIM provider package."""

from free_claude_code.providers.defaults import NVIDIA_NIM_DEFAULT_BASE

from .client import NvidiaNimProvider

__all__ = ["NVIDIA_NIM_DEFAULT_BASE", "NvidiaNimProvider"]
