"""Fireworks AI provider using OpenAI-compatible Chat Completions."""

from typing import Any

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import FIREWORKS_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)
from providers.transports.openai_chat.extra_body import (
    validate_extra_body_does_not_override_canonical_fields,
)

FIREWORKS_BASE_URL = FIREWORKS_DEFAULT_BASE

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="FIREWORKS",
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class FireworksProvider(OpenAIChatTransport):
    """Fireworks AI using ``https://api.fireworks.ai/inference/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="FIREWORKS",
            base_url=config.base_url or FIREWORKS_BASE_URL,
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
