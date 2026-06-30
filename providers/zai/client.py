"""Z.ai provider implementation (Anthropic-compatible Messages API)."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import ZAI_DEFAULT_BASE
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_POLICY = NativeMessagesRequestPolicy(
    provider_name="ZAI",
    extra_body="reject",
    reject_extra_body_message=(
        "Z.ai native Messages API does not support extra_body on requests."
    ),
)


class ZaiProvider(AnthropicMessagesTransport):
    """Z.ai using Anthropic-compatible Messages at api.z.ai/api/anthropic/v1."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI",
            default_base_url=ZAI_DEFAULT_BASE,
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
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
