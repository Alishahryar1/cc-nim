"""Ollama provider implementation."""

import httpx

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import OLLAMA_DEFAULT_BASE
from free_claude_code.providers.model_listing import extract_ollama_model_ids
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
)


class OllamaProvider(AnthropicMessagesTransport):
    """Ollama provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OLLAMA",
            default_base_url=OLLAMA_DEFAULT_BASE,
        )
        self._api_key = config.api_key or "ollama"

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming native Anthropic messages response."""
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _send_model_list_request(self) -> httpx.Response:
        """Query Ollama's native local model-list endpoint."""
        return await self._client.get(f"{self._base_url}/api/tags")

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        return extract_ollama_model_ids(payload, provider_name=self._provider_name)
