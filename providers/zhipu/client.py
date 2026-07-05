"""Zhipu AI (GLM) provider implementation."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import ZHIPU_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class ZhipuProvider(OpenAIChatTransport):
    """Zhipu AI (GLM) provider using the OpenAI-compatible chat completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZHIPU",
            base_url=config.base_url or ZHIPU_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
