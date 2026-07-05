"""Z.ai (GLM-5.2) provider — native Anthropic Messages transport.

Z.ai exposes a native Anthropic Messages-compatible endpoint at
``https://api.z.ai/api/anthropic``, so requests pass through unmodified.
"""

from __future__ import annotations

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig

ZAI_DEFAULT_BASE = "https://api.z.ai/api/anthropic"


class ZAiProvider(AnthropicMessagesTransport):
    """Z.ai provider for GLM-5.2 using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI",
            default_base_url=ZAI_DEFAULT_BASE,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

    async def _send_model_list_request(self) -> httpx.Response:
        """Z.ai may not expose a /models endpoint; always pass and let discovery fail gracefully."""
        url = str(
            httpx.URL(self._base_url).copy_with(
                path="/models", query=None, fragment=None
            )
        )
        return await self._client.get(url, headers=self._model_list_headers())

    def _model_list_headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}
