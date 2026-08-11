"""Alibaba DashScope (OpenAI-compat) adapter."""

from free_claude_code.config.provider_catalog import ALIBABA_DEFAULT_BASE

from .client import AlibabaProvider

__all__ = ["ALIBABA_DEFAULT_BASE", "AlibabaProvider"]
