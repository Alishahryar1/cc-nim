"""Ollama Cloud provider implementation."""

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import OLLAMA_CLOUD_DEFAULT_BASE
from providers.model_listing import extract_ollama_model_ids


class OllamaCloudProvider(AnthropicMessagesTransport):
    """Ollama Cloud provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OLLAMA_CLOUD",
            default_base_url=OLLAMA_CLOUD_DEFAULT_BASE,
        )
        self._api_key = config.api_key or ""

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

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
        """Query Ollama Cloud's model-list endpoint."""
        return await self._client.get(
            "/api/tags",
            headers=self._model_list_headers(),
        )

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        return extract_ollama_model_ids(payload, provider_name=self._provider_name)
