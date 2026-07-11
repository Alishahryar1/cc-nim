"""Mistral Codestral provider (OpenAI-compatible chat on codestral.mistral.ai)."""

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import CODESTRAL_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(provider_name="CODESTRAL")


class CodestralProvider(OpenAIChatTransport):
    """Codestral host using ``https://codestral.mistral.ai/v1/chat/completions``.

    Uses a separate Codestral API key from La Plateforme (``MISTRAL_API_KEY``).
    Request shaping matches Mistral La Plateforme.
    """

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="CODESTRAL",
            base_url=config.base_url or CODESTRAL_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
