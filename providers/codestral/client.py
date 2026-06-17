"""Mistral Codestral provider (OpenAI-compatible chat on codestral.mistral.ai)."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import CODESTRAL_DEFAULT_BASE
from providers.mistral.request import build_request_body
from providers.transports.openai_chat import OpenAIChatTransport


class CodestralProvider(OpenAIChatTransport):
    """Codestral host using ``https://codestral.mistral.ai/v1/chat/completions``.

    Uses a separate Codestral API key from La Plateforme (``MISTRAL_API_KEY``).
    Request shaping matches Mistral La Plateforme (shared ``build_request_body``).
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CODESTRAL",
            base_url=config.base_url or CODESTRAL_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
