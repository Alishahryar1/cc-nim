import httpx

from providers.base import ProviderConfig
from providers.defaults import OLLAMA_DEFAULT_BASE
from providers.local_provider import LocalAuthProvider
from providers.model_listing import extract_ollama_model_ids


class OllamaProvider(LocalAuthProvider):
    """Ollama provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config,
            provider_name="OLLAMA",
            default_base_url=OLLAMA_DEFAULT_BASE,
        )

    def _request_headers(self) -> dict[str, str]:
        """Return headers for the native messages request."""
        return {"Content-Type": "application/json"}

    async def _send_model_list_request(self) -> httpx.Response:
        """Query Ollama's native local model-list endpoint."""
        from httpx import AsyncClient

        async with AsyncClient(
            base_url=self._base_url, timeout=self._client.timeout
        ) as client:
            return await client.get("/api/tags")

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        return extract_ollama_model_ids(payload, provider_name=self._provider_name)
