"""Lemonade provider implementation."""

from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import LEMONADE_DEFAULT_BASE
from providers.model_listing import extract_openai_model_ids
from providers.transports.anthropic_messages import AnthropicMessagesTransport


class LemonadeProvider(AnthropicMessagesTransport):
    """Lemonade provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LEMONADE",
            default_base_url=LEMONADE_DEFAULT_BASE,
        )
        self._api_key = config.api_key or "lemonade"

    def _is_thinking_enabled(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> bool:
        """Always disable thinking for Lemonade local models."""
        return False

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _send_model_list_request(self) -> httpx.Response:
        return await self._client.get(f"{self._base_url}/v1/models")

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        return extract_openai_model_ids(
            payload, provider_name=self._provider_name
        )
