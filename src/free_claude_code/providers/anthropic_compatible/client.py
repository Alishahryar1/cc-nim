"""Generic Anthropic-compatible provider implementation."""

from free_claude_code.config.provider_catalog import ANTHROPIC_COMPATIBLE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

_REQUEST_POLICY = NativeMessagesRequestPolicy(
    provider_name="ANTHROPIC_COMPATIBLE",
)


class AnthropicCompatibleProvider(AnthropicMessagesTransport):
    """Generic Anthropic-compatible provider using configurable base_url."""

    def __init__(
        self,
        config: ProviderConfig,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(
            config,
            admission=admission,
            provider_name="ANTHROPIC_COMPATIBLE",
            default_base_url=config.base_url or ANTHROPIC_COMPATIBLE_DEFAULT_BASE,
        )

    def _is_thinking_enabled(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> bool:
        if thinking_enabled is not None:
            return thinking_enabled
        thinking = request.thinking
        if isinstance(thinking, dict):
            return thinking.get("type") == "enabled"
        return getattr(thinking, "type", None) == "enabled"

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
