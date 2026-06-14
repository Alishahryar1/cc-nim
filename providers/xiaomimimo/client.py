"""Xiaomi MiMo provider using native Anthropic-compatible Messages.

Pay-As-You-Go plan endpoint: https://api.xiaomimimo.com/anthropic
Token Plan endpoint:          https://token-plan-cn.xiaomimimo.com/anthropic

API key obtained from: https://platform.xiaomimimo.com/console/api-keys
Docs: https://mimo.mi.com/docs/en-US/tokenplan/integration/tools-overview

NOTE on model listing:
  The Anthropic-compat endpoint is at /anthropic, but the OpenAI-format model
  list lives at /v1/models (root-level), NOT at /anthropic/models.  The default
  AnthropicMessagesTransport._send_model_list_request would build the wrong URL
  (https://api.xiaomimimo.com/anthropic/models → 404), so we override it with
  the same copy_with(path='/v1/models') trick used by DeepSeekProvider.
"""

from __future__ import annotations

from typing import Any

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import XIAOMIMIMO_DEFAULT_BASE

from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"


class XiaomiMiMoProvider(AnthropicMessagesTransport):
    """Xiaomi MiMo using native Anthropic-compatible Messages API.

    Supports models: mimo-v2.5-pro, mimo-v2.5, mimo-v2-flash (and legacy v2 aliases).
    The endpoint is a first-class Anthropic Messages implementation — no OpenAI
    protocol translation is required.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="XIAOMIMIMO",
            default_base_url=XIAOMIMIMO_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        if thinking_enabled is None:
            thinking_enabled = self._is_thinking_enabled(request)
        return build_request_body(
            request,
            thinking_enabled=thinking_enabled,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    async def _send_model_list_request(self) -> httpx.Response:
        """MiMo lists models from the OpenAI-format root (/v1/models), not /anthropic/models."""
        url = str(
            httpx.URL(self._base_url).copy_with(
                path="/v1/models", query=None, fragment=None
            )
        )
        return await self._client.get(url, headers=self._model_list_headers())

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
