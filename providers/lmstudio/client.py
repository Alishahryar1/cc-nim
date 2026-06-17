from providers.base import ProviderConfig
from providers.defaults import LMSTUDIO_DEFAULT_BASE
from providers.local_provider import LocalAuthProvider


class LMStudioProvider(LocalAuthProvider):
    """LM Studio provider using native Anthropic Messages endpoint."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config,
            provider_name="LMSTUDIO",
            default_base_url=LMSTUDIO_DEFAULT_BASE,
        )
