"""Ollama provider implementation - supports local and cloud modes."""

import httpx

from providers.base import ProviderConfig
from providers.defaults import OLLAMA_DEFAULT_BASE
from providers.model_listing import extract_ollama_model_ids
from providers.transports.anthropic_messages import AnthropicMessagesTransport


class OllamaProvider(AnthropicMessagesTransport):
    """Ollama provider supporting local Ollama and Ollama Cloud."""

    def __init__(self, config: ProviderConfig):
        # Check if cloud mode is enabled
        self.use_cloud = getattr(config, 'ollama_use_cloud', False)
        self.cloud_api_key = getattr(config, 'ollama_api_key', '')

        if self.use_cloud and self.cloud_api_key:
            # Cloud mode - same Anthropic endpoint, different base URL + auth
            super().__init__(
                config,
                provider_name="OLLAMA_CLOUD",
                default_base_url="https://ollama.com",
            )
            self._api_key = self.cloud_api_key
        else:
            # Local mode 
            super().__init__(
                config,
                provider_name="OLLAMA",
                default_base_url=OLLAMA_DEFAULT_BASE,
            )
            self._api_key = config.api_key or "ollama"

    def _request_headers(self) -> dict[str, str]:
        """Return headers for the request."""
        headers = {"Content-Type": "application/json"}
        if self.use_cloud and self.cloud_api_key:
            headers["Authorization"] = f"Bearer {self.cloud_api_key}"
        return headers

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
        """Query available models."""
        if self.use_cloud and self.cloud_api_key:
            # Cloud: OpenAI-compatible model list
            return await self._client.get(
                "https://ollama.com/v1/models",
                headers=self._request_headers(),
            )
        else:
            # Local: Ollama's native tags endpoint
            return await self._client.get(f"{self._base_url}/api/tags")

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        """Extract model IDs from response."""
        if self.use_cloud and self.cloud_api_key:
            # Parse OpenAI format: {"data": [{"id": "model-name"}]}
            model_ids = set()
            if isinstance(payload, dict) and "data" in payload:
                for model in payload["data"]:
                    if isinstance(model, dict) and "id" in model:
                        model_ids.add(model["id"])
            return frozenset(model_ids)
        else:
            # Parse Ollama local format
            return extract_ollama_model_ids(payload, provider_name="ollama")