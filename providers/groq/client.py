"""Groq provider implementation (OpenAI-compatible chat completions)."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import GROQ_DEFAULT_BASE
from providers.transports.openai_chat import (
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

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="GROQ",
            base_url=config.base_url or GROQ_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
