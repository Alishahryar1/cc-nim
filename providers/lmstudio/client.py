"""LM Studio provider implementation."""

from providers.base import ProviderConfig
from providers.defaults import LMSTUDIO_DEFAULT_BASE
from providers.transports.anthropic_messages import AnthropicMessagesTransport


class LMStudioProvider(AnthropicMessagesTransport):
    """LM Studio provider using native Anthropic Messages endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LMSTUDIO",
            default_base_url=LMSTUDIO_DEFAULT_BASE,
        )
