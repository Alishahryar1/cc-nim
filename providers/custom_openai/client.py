"""Custom OpenAI-compatible provider implementation.

This provider adapts free-claude-code's Anthropic-format requests into
OpenAI /v1/chat/completions format and sends them to a user-configured
endpoint. It extends OpenAIChatTransport (which already handles Anthropic→OpenAI
conversion) and adds:
  - A custom request body builder for fine-grained conversion control.
  - Graceful fallback when the upstream /v1/models endpoint is unavailable,
    using the user's configured model refs instead.
"""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class CustomOpenAIProvider(OpenAIChatTransport):
    """Provider client for any OpenAI /v1/chat/completions endpoint.

    Use cases:
      - freellmapi (aggregates free LLM API tiers behind an OpenAI-compatible API).
      - Self-hosted models via llama.cpp, LM Studio, vLLM, Ollama (OpenAI compat).
      - Any third-party API that speaks the OpenAI chat completions format.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CUSTOM_OPENAI",
            base_url=config.base_url or "",
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Convert an internal Anthropic-format request to OpenAI chat completions format."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def list_model_ids(self) -> frozenset[str]:
        """Query the upstream /v1/models endpoint, falling back on failure.

        If the upstream models endpoint is unreachable (common for lightweight
        proxies that don't implement it), falls back to any custom_openai models
        the user has configured via MODEL env vars, or returns sensible defaults
        so the provider still works.
        """
        try:
            return await super().list_model_ids()
        except Exception:
            from config.settings import get_settings

            settings = get_settings()
            custom_model_ids = {
                ref.model_id
                for ref in settings.configured_chat_model_refs()
                if ref.provider_id == "custom_openai"
            }
            if not custom_model_ids:
                custom_model_ids.add("auto")
                custom_model_ids.add("gpt-4o-mini")
            return frozenset(custom_model_ids)
