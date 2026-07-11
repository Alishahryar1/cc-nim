"""OpenCode Zen provider implementation (OpenAI-compatible Chat Completions)."""

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import OPENCODE_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)


class OpenCodeProvider(OpenAIChatTransport):
    """OpenCode Zen provider using ``https://opencode.ai/zen/v1/chat/completions``."""

    def __init__(
        self,
        config: ProviderConfig,
        provider_name: str = "OPENCODE",
        *,
        rate_limiter: ProviderRateLimiter,
    ):
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=config.base_url or OPENCODE_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )
        self._request_policy = OpenAIChatRequestPolicy(provider_name=provider_name)

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=self._request_policy,
        )
