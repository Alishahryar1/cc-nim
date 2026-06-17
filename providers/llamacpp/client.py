from providers.base import ProviderConfig
from providers.defaults import LLAMACPP_DEFAULT_BASE
from providers.local_provider import LocalAuthProvider


class LlamaCppProvider(LocalAuthProvider):
    """Llama.cpp provider using native Anthropic Messages endpoint."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config,
            provider_name="LLAMACPP",
            default_base_url=LLAMACPP_DEFAULT_BASE,
        )
