"""Fireworks AI provider using native Anthropic-compatible Messages."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_POLICY = NativeMessagesRequestPolicy(
    provider_name="FIREWORKS",
    extra_body="merge_validated",
)


class FireworksProvider(AnthropicMessagesTransport):
    """Fireworks AI using Anthropic-compatible Messages."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="FIREWORKS",
            default_base_url=FIREWORKS_BASE_URL,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
