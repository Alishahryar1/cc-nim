"""MiniMax provider using native Anthropic-compatible Messages."""

from __future__ import annotations

from typing import Any

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import MINIMAX_DEFAULT_BASE

from .request import build_request_body

# OpenAI-compat root for model discovery (separate from the Anthropic path).
_MINIMAX_OPENAI_MODELS_URL = "https://api.minimaxi.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"


class MinimaxProvider(AnthropicMessagesTransport):
    """MiniMax provider using Anthropic-compatible Messages at api.minimaxi.com."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MINIMAX",
            default_base_url=MINIMAX_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    async def _send_model_list_request(self) -> httpx.Response:
        """Models are listed from the OpenAI-compat root, not ``/anthropic/v1``."""
        return await self._client.get(
            _MINIMAX_OPENAI_MODELS_URL,
            headers=self._model_list_headers(),
        )

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
