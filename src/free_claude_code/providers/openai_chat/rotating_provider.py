"""OpenAI-compatible provider variant with rotating credential fallback."""

from collections.abc import Mapping

from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig

from .provider import OpenAIChatProvider
from .key_rotation import RotatingOpenAIClient
from .profiles import OpenAIChatProfile


class RotatingOpenAIChatProvider(OpenAIChatProvider):
    """Drop-in OpenAI-chat provider that rotates through configured API keys."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile,
        admission: ProviderAdmissionController,
        default_headers: Mapping[str, str] | None = None,
        api_key_provider=None,
    ):
        super().__init__(
            config,
            profile=profile,
            admission=admission,
            default_headers=default_headers,
            api_key_provider=api_key_provider,
        )
        self._legacy_client = getattr(self, "_client", None)
        self._client = RotatingOpenAIClient(
            provider_name=self._provider_name,
            config=config,
            base_url=self._base_url,
            default_headers=default_headers,
        )

    async def cleanup(self) -> None:
        """Release both the original bootstrap client and the rotating client."""

        legacy_client = getattr(self, "_legacy_client", None)
        if legacy_client is not None:
            await legacy_client.close()
            self._legacy_client = None
        await super().cleanup()
