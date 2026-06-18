"""Llama.cpp provider implementation."""

from providers.base import ProviderConfig
from providers.defaults import LLAMACPP_DEFAULT_BASE
from providers.transports.anthropic_messages import AnthropicMessagesTransport


class LlamaCppProvider(AnthropicMessagesTransport):
    """Llama.cpp provider using native Anthropic Messages endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LLAMACPP",
            default_base_url=LLAMACPP_DEFAULT_BASE,
        )
