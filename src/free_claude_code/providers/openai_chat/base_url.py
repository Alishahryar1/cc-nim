"""OpenAI-compatible API base URL policy.

Re-exports the shared URL normalization helper from :mod:`free_claude_code.core.urls`
so provider adapters and external consumers share one canonical implementation.
"""

from free_claude_code.core.urls import openai_v1_base_url

__all__ = ["openai_v1_base_url"]
