"""Llama.cpp provider implementation."""

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import LLAMACPP_DEFAULT_BASE
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
)


class LlamaCppProvider(AnthropicMessagesTransport):
    """Llama.cpp provider using native Anthropic Messages endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LLAMACPP",
            default_base_url=LLAMACPP_DEFAULT_BASE,
        )
