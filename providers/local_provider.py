"""Shared non-fallback base for local providers that may accept an api key."""

from __future__ import annotations

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig


class LocalAuthProvider(AnthropicMessagesTransport):
    """Local-provider transport that always honors ``config.api_key``."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
    ) -> None:
        super().__init__(
            config,
            provider_name=provider_name,
            default_base_url=default_base_url,
        )
