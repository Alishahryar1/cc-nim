"""Cerebras Inference provider (OpenAI-compatible chat completions)."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import CEREBRAS_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="CEREBRAS",
    include_extra_body=True,
    max_tokens_field="max_completion_tokens",
)


class CerebrasProvider(OpenAIChatTransport):
    """Cerebras API at ``https://api.cerebras.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CEREBRAS",
            base_url=config.base_url or CEREBRAS_DEFAULT_BASE,
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
