"""
Custom OpenAI-compatible provider adapter.

Allows connecting free-claude-code to any OpenAI-compatible API endpoint
(e.g., freellmapi, LM Studio, llama.cpp, vLLM, or any service that speaks
the OpenAI /v1/chat/completions format). The user configures the endpoint
URL and API key via the admin UI.
"""

from .client import CustomOpenAIProvider

__all__ = ["CustomOpenAIProvider"]
