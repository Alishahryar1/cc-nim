"""MiniMax provider implementation (Anthropic-compatible Messages API)."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import MINIMAX_DEFAULT_BASE
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_POLICY = NativeMessagesRequestPolicy(provider_name="MINIMAX")


class MiniMaxProvider(AnthropicMessagesTransport):
    """MiniMax using Anthropic-compatible Messages at api.minimax.io/anthropic/v1."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MINIMAX",
            default_base_url=MINIMAX_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        return build_native_messages_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
            postprocessors=(_apply_minimax_thinking_policy,),
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


def _apply_minimax_thinking_policy(
    body: dict[str, Any], _request: Any, thinking_enabled: bool
) -> None:
    """Use MiniMax's documented Anthropic thinking control values."""
    body["thinking"] = (
        {"type": "adaptive"} if thinking_enabled else {"type": "disabled"}
    )
