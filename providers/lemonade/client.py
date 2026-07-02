"""Lemonade provider implementation (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import LEMONADE_DEFAULT_BASE
from providers.model_listing import extract_openai_model_ids
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="LEMONADE",
    include_extra_body=True,
)


class LemonadeProvider(OpenAIChatTransport):
    """Lemonade provider using OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LEMONADE",
            base_url=config.base_url or LEMONADE_DEFAULT_BASE,
            api_key=config.api_key or "lemonade",
        )
        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._http_client.aclose()
        await super().cleanup()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from Lemonade's /v1/models endpoint."""
        response = await self._http_client.get("/v1/models")
        response.raise_for_status()
        return extract_openai_model_ids(
            response.json(), provider_name=self._provider_name
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
