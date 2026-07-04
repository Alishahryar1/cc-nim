"""Wafer provider implementation (native Anthropic-compatible Messages)."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import WAFER_DEFAULT_BASE
from providers.transports.anthropic_messages import AnthropicMessagesTransport

_ANTHROPIC_VERSION = "2023-06-01"


class WaferProvider(AnthropicMessagesTransport):
    """Wafer using ``https://pass.wafer.ai/v1/messages``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="WAFER",
            default_base_url=WAFER_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build native body; Wafer rejects omitted thinking as ``reasoning_effort=none``."""
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        body = self._build_request_body_with_resolved_thinking(
            request,
            thinking_enabled=effective_thinking_enabled,
        )
        if "thinking" not in body:
            body["thinking"] = (
                {"type": "enabled"}
                if effective_thinking_enabled
                else {"type": "disabled"}
            )
        return body

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
