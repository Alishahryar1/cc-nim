"""Lemonade provider implementation."""

import httpx

from providers.base import ProviderConfig
from providers.defaults import LEMONADE_DEFAULT_BASE
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

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming native Anthropic messages response."""
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)
