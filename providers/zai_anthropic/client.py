"""Z.ai Anthropic provider for GLM 5.2 (native Anthropic Messages endpoint).

Uses ``https://api.z.ai/api/anthropic`` — same protocol as DeepSeek's
Anthropic endpoint, preserving thinking blocks, tool use, and SSE streaming.
"""

from __future__ import annotations

from typing import Any

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import ZAI_ANTHROPIC_DEFAULT_BASE

from .request import build_request_body


class ZaiAnthropicProvider(AnthropicMessagesTransport):
    """Z.ai GLM 5.2 via native Anthropic Messages endpoint.

    Auth: ``x-api-key`` header with ``zai-...`` Coding Plan API key.
    Model: ``glm-5.2`` or ``glm-5.2[1m]`` (1M context window).
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI_ANTHROPIC",
            default_base_url=ZAI_ANTHROPIC_DEFAULT_BASE,
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
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

    async def _send_model_list_request(self) -> httpx.Response:
        """Z.ai lists models from the OpenAI-format /models, not /anthropic."""
        url = str(
            httpx.URL(self._base_url).copy_with(
                path="/models", query=None, fragment=None
            )
        )
        return await self._client.get(url, headers=self._model_list_headers())

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
