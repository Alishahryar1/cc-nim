"""Groq provider implementation (OpenAI-compatible chat completions)."""

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import GROQ_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="GROQ",
    include_extra_body=True,
    max_tokens_field="max_completion_tokens",
    strip_message_names=True,
    unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
    normalize_n_to_one=True,
)


class GroqProvider(OpenAIChatTransport):
    """Groq API using ``https://api.groq.com/openai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="GROQ",
            base_url=config.base_url or GROQ_DEFAULT_BASE,
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
