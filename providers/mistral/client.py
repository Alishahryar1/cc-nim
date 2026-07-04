"""Mistral La Plateforme provider implementation (OpenAI-compatible chat completions)."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import MISTRAL_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(provider_name="MISTRAL")


class MistralProvider(OpenAIChatTransport):
    """Mistral API using ``https://api.mistral.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MISTRAL",
            base_url=config.base_url or MISTRAL_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
