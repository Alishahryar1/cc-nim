"""Custom provider for any OpenAI-compatible chat gateway."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.exceptions import AuthenticationError
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class CustomProvider(OpenAIChatTransport):
    """OpenAI-compatible ``/v1/chat/completions`` endpoint (``CUSTOM_URL_PROVIDER``)."""

    def __init__(self, config: ProviderConfig):
        base_url = (config.base_url or "").strip().rstrip("/")
        if not base_url:
            raise AuthenticationError(
                "CUSTOM_URL_PROVIDER is not set. "
                "Add it in the Admin UI Providers section."
            )
        if not (config.api_key or "").strip():
            raise AuthenticationError(
                "CUSTOM_API_KEY is not set. Add it in the Admin UI Providers section."
            )
        super().__init__(
            config,
            provider_name="CUSTOM",
            base_url=base_url,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
